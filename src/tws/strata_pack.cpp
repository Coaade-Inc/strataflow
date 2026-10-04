// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "tws/strata_pack.h"

#include "tws/async_io.h"      // kIoAlignment, align_up
#include "tws/strata_format.h"
#include "tws/stream_buft.h"   // parse_expert_tensor_name, ExpertTensorKind

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sf {
namespace {

// Zero-fill `f` from its current position so the next write starts at `target`.
bool pad_to(FILE *f, uint64_t target, uint64_t &pos) {
    static const uint8_t zeros[kIoAlignment] = {};
    while (pos < target) {
        const uint64_t chunk =
            target - pos < kIoAlignment ? target - pos : kIoAlignment;
        if (std::fwrite(zeros, 1, static_cast<size_t>(chunk), f) != chunk) {
            return false;
        }
        pos += chunk;
    }
    return true;
}

bool write_bytes(FILE *f, const void *data, uint64_t len, uint64_t &pos) {
    if (len == 0) return true;
    if (std::fwrite(data, 1, static_cast<size_t>(len), f) != len) return false;
    pos += len;
    return true;
}

// Copy `len` bytes from the source file at `src_off` into `f`, chunked.
bool copy_range(FILE *dst, FILE *src, uint64_t src_off, uint64_t len,
                uint64_t &pos) {
    if (std::fseek(src, static_cast<long>(src_off), SEEK_SET) != 0) return false;
    std::vector<uint8_t> buf(1u << 20);  // 1 MiB chunks
    uint64_t left = len;
    while (left > 0) {
        const size_t want =
            left < buf.size() ? static_cast<size_t>(left) : buf.size();
        if (std::fread(buf.data(), 1, want, src) != want) return false;
        if (std::fwrite(buf.data(), 1, want, dst) != want) return false;
        pos += want;
        left -= want;
    }
    return true;
}

struct TensorInfo {
    std::string name;
    uint64_t    src_offset = 0;  // absolute byte offset in the source GGUF
    uint64_t    size = 0;        // ggml_nbytes
    bool        is_expert = false;
    uint32_t    layer = 0;
    ExpertTensorKind kind = ExpertTensorKind::kGate;
    uint32_t    n_experts = 0;   // ne[2] for a stacked expert tensor
    uint64_t    nb02 = 0;        // per-expert stride (size / n_experts)
};

// Record `msg` in `*err` (if non-null) and return false, so callers can
// `return fail(err, "...")`.
bool fail(std::string *err, const std::string &msg) {
    if (err != nullptr) *err = msg;
    return false;
}

}  // namespace

bool pack_gguf_to_strata(const std::string &in_path,
                         const std::string &out_path, std::string *err) {
    if (err != nullptr) err->clear();

    // --- 1. Open the source GGUF metadata (no tensor data read) -------------
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context *gc = gguf_init_from_file(in_path.c_str(), gp);
    if (gc == nullptr) {
        return fail(err, "failed to parse source GGUF '" + in_path + "'");
    }

    const uint64_t data_offset = gguf_get_data_offset(gc);
    const int64_t  n_tensors   = gguf_get_n_tensors(gc);

    // Metadata region = everything before the tensor data blob. The verbatim
    // on-disk metadata size is exactly gguf_get_data_offset(): the file begins
    // with header+KV+tensor-info padded to the data offset. Copying [0,
    // data_offset) reproduces a standalone, parseable GGUF prefix.
    const uint64_t gguf_meta_size = data_offset;

    // --- Classify tensors ---------------------------------------------------
    std::vector<TensorInfo> trunk;
    std::vector<TensorInfo> experts;
    uint32_t max_layer = 0;
    uint32_t n_experts_global = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        t.name = gguf_get_tensor_name(gc, i);
        t.src_offset = data_offset + gguf_get_tensor_offset(gc, i);
        t.size = gguf_get_tensor_size(gc, i);

        ExpertTensorId id = parse_expert_tensor_name(t.name);
        if (id.valid) {
            const int64_t *ne = gguf_get_tensor_ne(gc, i);
            // Stacked expert tensors are 3-D with experts as the outer dim.
            const uint64_t n_exp = ne != nullptr && ne[2] > 0
                                       ? static_cast<uint64_t>(ne[2])
                                       : 0;
            if (n_exp == 0 || t.size % n_exp != 0) {
                gguf_free(gc);
                return fail(err, "expert tensor '" + t.name +
                                     "' has an unexpected shape");
            }
            t.is_expert = true;
            t.layer = id.layer;
            t.kind = id.kind;
            t.n_experts = static_cast<uint32_t>(n_exp);
            t.nb02 = t.size / n_exp;
            experts.push_back(t);
            max_layer = max_layer > id.layer ? max_layer : id.layer;
            n_experts_global = static_cast<uint32_t>(n_exp);
        } else {
            trunk.push_back(t);
        }
    }

    const uint32_t n_layers = experts.empty() ? 0 : (max_layer + 1);

    // Group the three (gate/up/down) kinds per (layer,expert). Index the
    // experts vector by {layer, kind} for quick slice lookup.
    auto find_kind = [&](uint32_t layer,
                         ExpertTensorKind kind) -> const TensorInfo * {
        for (const TensorInfo &t : experts) {
            if (t.layer == layer && t.kind == kind) return &t;
        }
        return nullptr;
    };

    // --- Open the files -----------------------------------------------------
    FILE *src = std::fopen(in_path.c_str(), "rb");
    if (src == nullptr) {
        gguf_free(gc);
        return fail(err, "cannot reopen source '" + in_path + "'");
    }
    FILE *out = std::fopen(out_path.c_str(), "wb");
    if (out == nullptr) {
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "cannot create '" + out_path + "'");
    }

    uint64_t pos = 0;  // current write offset in the output file

    // --- 2. Reserve the superblock (backfilled at the end) ------------------
    StrataSuperblock sb{};
    std::memcpy(sb.magic, kStrataMagic, sizeof(kStrataMagic));
    sb.version = kStrataVersion;
    sb.flags = kStrataFlagExpertsBundled | kStrataFlagIndexSorted;
    sb.align = static_cast<uint32_t>(kIoAlignment);
    sb.n_layers = n_layers;
    sb.n_experts = n_experts_global;
    if (!write_bytes(out, &sb, kStrataSuperblockSize, pos)) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "write superblock failed");
    }

    // --- Embedded GGUF metadata blob (verbatim [0, data_offset)) ------------
    sb.gguf_meta_offset = pos;
    sb.gguf_meta_size = gguf_meta_size;
    if (!copy_range(out, src, 0, gguf_meta_size, pos)) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "copy GGUF metadata failed");
    }

    // --- 3. Expert index (count known now; offsets backfilled below) --------
    // v1 writes experts in {layer-major, expert-minor} order.
    std::vector<ExpertIndexEntry> index;
    index.reserve(static_cast<size_t>(n_layers) * n_experts_global);
    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        const TensorInfo *g = find_kind(layer, ExpertTensorKind::kGate);
        const TensorInfo *gu = find_kind(layer, ExpertTensorKind::kGateUp);
        const uint32_t ne = g != nullptr ? g->n_experts
                                         : (gu != nullptr ? gu->n_experts : 0);
        for (uint32_t e = 0; e < ne; ++e) {
            ExpertIndexEntry entry{};
            entry.layer = layer;
            entry.expert = e;
            index.push_back(entry);
        }
    }

    sb.index_offset = pos;
    sb.index_count = index.size();
    const uint64_t index_bytes = index.size() * sizeof(ExpertIndexEntry);
    // Write a placeholder index now to reserve the file region; we rewrite it
    // with real offsets after the expert region is laid out.
    if (!write_bytes(out, index.data(), index_bytes, pos)) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "write index placeholder failed");
    }

    // --- 4. Trunk block (4 KiB-aligned start, tensors contiguous) -----------
    const uint64_t trunk_start = align_up(pos, kIoAlignment);
    if (!pad_to(out, trunk_start, pos)) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "pad to trunk failed");
    }
    sb.trunk_offset = trunk_start;
    for (const TensorInfo &t : trunk) {
        if (!copy_range(out, src, t.src_offset, t.size, pos)) {
            std::fclose(out); std::fclose(src); gguf_free(gc);
            return fail(err, "copy trunk tensor '" + t.name + "' failed");
        }
    }
    sb.trunk_size = pos - trunk_start;

    // --- 5. Expert region: one 4 KiB-aligned blob per (layer,expert) --------
    const uint64_t region_start = align_up(pos, kIoAlignment);
    if (!pad_to(out, region_start, pos)) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "pad to expert region failed");
    }
    sb.expert_region_offset = region_start;

    for (ExpertIndexEntry &entry : index) {
        const uint32_t layer = entry.layer;
        const uint32_t e = entry.expert;

        const TensorInfo *gate = find_kind(layer, ExpertTensorKind::kGate);
        const TensorInfo *up   = find_kind(layer, ExpertTensorKind::kUp);
        const TensorInfo *down = find_kind(layer, ExpertTensorKind::kDown);
        const TensorInfo *fused = find_kind(layer, ExpertTensorKind::kGateUp);

        // 4 KiB-align the blob START (only the start; the gate/up/down slices
        // are tightly packed inside so the reader derives slice lengths from
        // the *_rel gaps, and one read_at yields all three kinds).
        const uint64_t blob_start = align_up(pos, kIoAlignment);
        if (!pad_to(out, blob_start, pos)) {
            std::fclose(out); std::fclose(src); gguf_free(gc);
            return fail(err, "pad to expert blob failed");
        }
        entry.blob_offset = blob_start;

        uint64_t rel = 0;
        auto emit = [&](const TensorInfo *t, uint64_t &rel_out) -> bool {
            rel_out = rel;
            const uint64_t slice_off =
                t->src_offset + static_cast<uint64_t>(e) * t->nb02;
            if (!copy_range(out, src, slice_off, t->nb02, pos)) return false;
            rel += t->nb02;
            return true;
        };

        bool ok = true;
        if (fused != nullptr) {
            // Fused gate_up arch: bundle gate_up then down. gate_rel == up_rel
            // (one fused slice), down_rel follows.
            ok = ok && emit(fused, entry.gate_rel);
            entry.up_rel = entry.gate_rel;
            ok = ok && (down != nullptr) && emit(down, entry.down_rel);
        } else {
            if (gate == nullptr || up == nullptr || down == nullptr) {
                std::fclose(out); std::fclose(src); gguf_free(gc);
                return fail(err, "layer " + std::to_string(layer) +
                                     " is missing a gate/up/down expert tensor");
            }
            ok = ok && emit(gate, entry.gate_rel);
            ok = ok && emit(up, entry.up_rel);
            ok = ok && emit(down, entry.down_rel);
        }
        if (!ok) {
            std::fclose(out); std::fclose(src); gguf_free(gc);
            return fail(err, "copy expert blob (layer " +
                                 std::to_string(layer) + ", expert " +
                                 std::to_string(e) + ") failed");
        }
        entry.blob_length = pos - blob_start;
    }
    sb.expert_region_size = pos - region_start;

    // --- 6. Backfill the index with real offsets ---------------------------
    if (std::fseek(out, static_cast<long>(sb.index_offset), SEEK_SET) != 0) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "seek to index failed");
    }
    if (index_bytes > 0 &&
        std::fwrite(index.data(), 1, static_cast<size_t>(index_bytes), out) !=
            index_bytes) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "rewrite index failed");
    }

    // --- Backfill the superblock with real offsets + CRC --------------------
    sb.header_crc32 = strata_crc32(&sb, kStrataHeaderCrcOffset);
    if (std::fseek(out, 0, SEEK_SET) != 0) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "seek to superblock failed");
    }
    if (std::fwrite(&sb, 1, kStrataSuperblockSize, out) !=
        kStrataSuperblockSize) {
        std::fclose(out); std::fclose(src); gguf_free(gc);
        return fail(err, "rewrite superblock failed");
    }

    std::fclose(out);
    std::fclose(src);
    gguf_free(gc);
    return true;
}

}  // namespace sf

// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier:
// LicenseRef-Coaade-Source-Available-1.0
#include "tws/strata_pack.h"

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tws/async_io.h"  // kIoAlignment, align_up
#include "tws/strata_format.h"
#include "tws/stream_buft.h"  // parse_expert_tensor_name, ExpertTensorKind

namespace sf {
namespace {

// Zero-fill `f` from its current position so the next write starts at `target`.
bool pad_to(FILE *f, uint64_t target, uint64_t &pos) {
    static const uint8_t zeros[kIoAlignment] = {};
    while (pos < target) {
        const uint64_t chunk = target - pos < kIoAlignment ? target - pos : kIoAlignment;
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
bool copy_range(FILE *dst, FILE *src, uint64_t src_off, uint64_t len, uint64_t &pos) {
    if (std::fseek(src, static_cast<long>(src_off), SEEK_SET) != 0) return false;
    std::vector<uint8_t> buf(1u << 20);  // 1 MiB chunks
    uint64_t left = len;
    while (left > 0) {
        const size_t want = left < buf.size() ? static_cast<size_t>(left) : buf.size();
        if (std::fread(buf.data(), 1, want, src) != want) return false;
        if (std::fwrite(buf.data(), 1, want, dst) != want) return false;
        pos += want;
        left -= want;
    }
    return true;
}

// A classified source tensor. For a STACKED expert tensor `nb02` is the
// per-expert stride (size / n_experts) and the byte source for expert e is the
// nb02 slice at src_offset + e*nb02. For a LEGACY per-expert tensor the whole
// standalone tensor IS one expert's bytes, so `nb02 == size` and `expert`
// records which expert it is.
struct TensorInfo {
    std::string name;
    uint64_t src_offset = 0;  // absolute byte offset in the source GGUF
    uint64_t size = 0;        // ggml_nbytes
    bool is_expert = false;
    uint32_t layer = 0;
    ExpertTensorKind kind = ExpertTensorKind::kGate;
    uint32_t expert = 0;     // only meaningful for legacy per-expert tensors
    uint32_t n_experts = 0;  // ne[2] for a stacked expert tensor
    uint64_t nb02 = 0;       // per-expert stride (stacked) or full size (legacy)
    // Original ggml type + ne-shape of the SOURCE tensor, retained so the
    // legacy path can synthesize the stacked tensor-info metadata.
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
};

// Record `msg` in `*err` (if non-null) and return false, so callers can
// `return fail(err, "...")`.
bool fail(std::string *err, const std::string &msg) {
    if (err != nullptr) *err = msg;
    return false;
}

// Local parser for the LEGACY per-expert routed-FFN tensor names the real
// TheBloke Mixtral GGUFs use:  blk.<N>.ffn_{gate,down,up}.<E>[.weight].
// This recognition lives HERE (the raw-GGUF side) on purpose and is NOT added
// to parse_expert_tensor_name in stream_buft.*, because the engine reuses that
// matcher to read the .strata embedded metadata and must keep seeing ONLY the
// stacked _exps names (the metadata we synthesize below declares stacked
// tensors). Returns true and fills {layer, kind, expert} on a match.
struct LegacyExpertId {
    uint32_t layer = 0;
    ExpertTensorKind kind = ExpertTensorKind::kGate;
    uint32_t expert = 0;
};

bool parse_legacy_expert_name(const std::string &name, LegacyExpertId &out) {
    static const char kPrefix[] = "blk.";
    if (name.compare(0, 4, kPrefix) != 0) return false;

    size_t pos = 4;
    uint64_t layer = 0;
    size_t digits = 0;
    while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
        layer = layer * 10 + static_cast<uint64_t>(name[pos] - '0');
        ++pos;
        ++digits;
    }
    if (digits == 0) return false;
    if (pos >= name.size() || name[pos] != '.') return false;
    ++pos;  // skip the '.'

    std::string body = name.substr(pos);
    static const std::string kWeightSuffix = ".weight";
    if (body.size() > kWeightSuffix.size() &&
        body.compare(body.size() - kWeightSuffix.size(), kWeightSuffix.size(),
                     kWeightSuffix) == 0) {
        body = body.substr(0, body.size() - kWeightSuffix.size());
    }

    // body must now be "ffn_<kind>.<E>" with a trailing numeric expert index.
    const size_t dot = body.rfind('.');
    if (dot == std::string::npos) return false;
    const std::string kind_str = body.substr(0, dot);
    const std::string exp_str = body.substr(dot + 1);
    if (exp_str.empty()) return false;
    uint64_t expert = 0;
    for (char c : exp_str) {
        if (c < '0' || c > '9') return false;
        expert = expert * 10 + static_cast<uint64_t>(c - '0');
    }

    ExpertTensorKind kind;
    if (kind_str == "ffn_gate") {
        kind = ExpertTensorKind::kGate;
    } else if (kind_str == "ffn_down") {
        kind = ExpertTensorKind::kDown;
    } else if (kind_str == "ffn_up") {
        kind = ExpertTensorKind::kUp;
    } else {
        return false;  // ffn_gate_inp / ffn_norm / etc. are trunk, not experts
    }

    out.layer = static_cast<uint32_t>(layer);
    out.kind = kind;
    out.expert = static_cast<uint32_t>(expert);
    return true;
}

const char *kind_stacked_suffix(ExpertTensorKind kind) {
    switch (kind) {
        case ExpertTensorKind::kGate:
            return "ffn_gate_exps";
        case ExpertTensorKind::kDown:
            return "ffn_down_exps";
        case ExpertTensorKind::kUp:
            return "ffn_up_exps";
        case ExpertTensorKind::kGateUp:
            return "ffn_gate_up_exps";
    }
    return "ffn_gate_exps";
}

// Copy one KV from the source gguf context into the destination, preserving the
// gguf value type. Arrays are copied via the raw array data (and string arrays
// via the element getters). Returns false only on an unexpected/unknown type
// (none of which our fixtures or real Mixtral use).
bool copy_kv(gguf_context *dst, const gguf_context *src, int64_t kid) {
    const char *key = gguf_get_key(src, kid);
    const gguf_type t = gguf_get_kv_type(src, kid);
    switch (t) {
        case GGUF_TYPE_UINT8:
            gguf_set_val_u8(dst, key, gguf_get_val_u8(src, kid));
            return true;
        case GGUF_TYPE_INT8:
            gguf_set_val_i8(dst, key, gguf_get_val_i8(src, kid));
            return true;
        case GGUF_TYPE_UINT16:
            gguf_set_val_u16(dst, key, gguf_get_val_u16(src, kid));
            return true;
        case GGUF_TYPE_INT16:
            gguf_set_val_i16(dst, key, gguf_get_val_i16(src, kid));
            return true;
        case GGUF_TYPE_UINT32:
            gguf_set_val_u32(dst, key, gguf_get_val_u32(src, kid));
            return true;
        case GGUF_TYPE_INT32:
            gguf_set_val_i32(dst, key, gguf_get_val_i32(src, kid));
            return true;
        case GGUF_TYPE_FLOAT32:
            gguf_set_val_f32(dst, key, gguf_get_val_f32(src, kid));
            return true;
        case GGUF_TYPE_UINT64:
            gguf_set_val_u64(dst, key, gguf_get_val_u64(src, kid));
            return true;
        case GGUF_TYPE_INT64:
            gguf_set_val_i64(dst, key, gguf_get_val_i64(src, kid));
            return true;
        case GGUF_TYPE_FLOAT64:
            gguf_set_val_f64(dst, key, gguf_get_val_f64(src, kid));
            return true;
        case GGUF_TYPE_BOOL:
            gguf_set_val_bool(dst, key, gguf_get_val_bool(src, kid));
            return true;
        case GGUF_TYPE_STRING:
            gguf_set_val_str(dst, key, gguf_get_val_str(src, kid));
            return true;
        case GGUF_TYPE_ARRAY: {
            const gguf_type et = gguf_get_arr_type(src, kid);
            const size_t n = gguf_get_arr_n(src, kid);
            if (et == GGUF_TYPE_STRING) {
                std::vector<const char *> ptrs;
                ptrs.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    ptrs.push_back(gguf_get_arr_str(src, kid, i));
                }
                gguf_set_arr_str(dst, key, ptrs.data(), ptrs.size());
            } else {
                // Numeric arrays: the raw data pointer is a contiguous block of
                // `n` elements of the element type; gguf_set_arr_data copies it.
                const void *data = gguf_get_arr_data(src, kid);
                gguf_set_arr_data(dst, key, et, data, n);
            }
            return true;
        }
        default:
            return false;
    }
}

}  // namespace

bool pack_gguf_to_strata(const std::string &in_path, const std::string &out_path,
                         std::string *err) {
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
    const int64_t n_tensors = gguf_get_n_tensors(gc);

    // --- MoE metadata (authoritative) ---------------------------------------
    // The GGUF declares whether this is a MoE and how many experts/layers it
    // has via '<arch>.expert_count' / '<arch>.block_count' (mirroring how
    // src/model/ggml_model.cpp read_shape_from_gguf reads them; fall back to
    // the 'llama' prefix when general.architecture is absent). We treat these
    // as the ground truth and cross-check the name-based classifier against
    // them below, so a model that declares experts but whose expert tensors
    // all miss the name match is caught LOUDLY instead of silently producing a
    // 0-expert .strata the engine then loads fully resident (the Mixtral OOM).
    auto gguf_get_str = [&](const char *key) -> std::string {
        const int64_t id = gguf_find_key(gc, key);
        if (id < 0) return std::string();
        const char *v = gguf_get_val_str(gc, id);
        return v != nullptr ? std::string(v) : std::string();
    };
    auto gguf_get_u32 = [&](const std::string &key, uint32_t fallback) -> uint32_t {
        const int64_t id = gguf_find_key(gc, key.c_str());
        if (id < 0) return fallback;
        return gguf_get_val_u32(gc, id);
    };
    std::string arch = gguf_get_str("general.architecture");
    if (arch.empty()) arch = "llama";
    const uint32_t meta_expert_count = gguf_get_u32(arch + ".expert_count", 0);
    const uint32_t meta_block_count = gguf_get_u32(arch + ".block_count", 0);

    // Verbatim on-disk metadata size for the already-stacked fast-path: the
    // file begins with header+KV+tensor-info padded to the data offset, so
    // copying [0, data_offset) reproduces a standalone, parseable GGUF prefix.
    const uint64_t gguf_meta_size = data_offset;

    // --- Classify tensors ---------------------------------------------------
    // Three disjoint buckets:
    //   trunk   - non-expert tensors (copied verbatim; order preserved)
    //   experts - STACKED 3-D expert tensors (existing path), one per (layer,kind)
    //   legacy  - LEGACY per-expert 2-D tensors, one per (layer,kind,expert)
    std::vector<TensorInfo> trunk;
    std::vector<TensorInfo> experts;
    std::vector<TensorInfo> legacy;
    uint32_t max_layer = 0;
    uint32_t n_experts_global = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        t.name = gguf_get_tensor_name(gc, i);
        t.src_offset = data_offset + gguf_get_tensor_offset(gc, i);
        t.size = gguf_get_tensor_size(gc, i);
        t.type = gguf_get_tensor_type(gc, i);
        const int64_t *ne = gguf_get_tensor_ne(gc, i);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            t.ne[d] = ne != nullptr ? ne[d] : 1;
        }

        ExpertTensorId id = parse_expert_tensor_name(t.name);
        LegacyExpertId lid;
        if (id.valid) {
            // Stacked expert tensors are 3-D with experts as the outer dim.
            const uint64_t n_exp =
                ne != nullptr && ne[2] > 0 ? static_cast<uint64_t>(ne[2]) : 0;
            if (n_exp == 0 || t.size % n_exp != 0) {
                // Report the exact shape so a real-model shape surprise is
                // diagnosable: ne dims, ggml type, and byte size.
                const char *tn = ggml_type_name(t.type);
                std::string ne_str = "[";
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    if (d != 0) ne_str += ",";
                    ne_str += std::to_string(t.ne[d]);
                }
                ne_str += "]";
                gguf_free(gc);
                return fail(err, "expert tensor '" + t.name +
                                     "' has an unexpected shape: ne=" + ne_str +
                                     " type=" + (tn != nullptr ? tn : "?") +
                                     " size=" + std::to_string(t.size) +
                                     " (expected a 3-D stacked tensor whose "
                                     "ne[2] (expert count) divides the byte "
                                     "size)");
            }
            t.is_expert = true;
            t.layer = id.layer;
            t.kind = id.kind;
            t.n_experts = static_cast<uint32_t>(n_exp);
            t.nb02 = t.size / n_exp;
            experts.push_back(t);
            max_layer = max_layer > id.layer ? max_layer : id.layer;
            n_experts_global = static_cast<uint32_t>(n_exp);
        } else if (parse_legacy_expert_name(t.name, lid)) {
            // LEGACY per-expert 2-D tensor: the whole standalone tensor IS one
            // expert's bytes, so its full `size` is that expert's slice.
            t.is_expert = true;
            t.layer = lid.layer;
            t.kind = lid.kind;
            t.expert = lid.expert;
            t.nb02 = t.size;
            legacy.push_back(t);
            max_layer = max_layer > lid.layer ? max_layer : lid.layer;
        } else {
            trunk.push_back(t);
        }
    }

    // A GGUF must not mix the two expert layouts; that would be an ambiguous
    // source we have no faithful regroup for.
    if (!experts.empty() && !legacy.empty()) {
        gguf_free(gc);
        return fail(err,
                    "GGUF mixes stacked (_exps) and legacy per-expert FFN "
                    "tensors; refusing to pack an ambiguous expert layout");
    }

    const bool legacy_mode = experts.empty() && !legacy.empty();

    // --- Legacy grouping: require exactly meta_expert_count per (layer,kind) -
    // Determine n_expert from the authoritative metadata (fall back to the max
    // observed expert index + 1 when the metadata is silent), then verify every
    // (layer,kind) has exactly that many experts. Any gap/overcount fails
    // loudly: a partially-named legacy MoE is as dangerous as a totally-missed
    // one (it would stream wrong bytes), so we refuse rather than guess.
    if (legacy_mode) {
        uint32_t n_expert = meta_expert_count;
        if (n_expert == 0) {
            uint32_t max_e = 0;
            for (const TensorInfo &t : legacy) {
                max_e = max_e > t.expert ? max_e : t.expert;
            }
            n_expert = max_e + 1;
        }
        const uint32_t n_layers_legacy = max_layer + 1;
        // Count experts per (layer,kind). kinds: gate/down/up (3).
        for (uint32_t layer = 0; layer < n_layers_legacy; ++layer) {
            for (int ki = 0; ki < 3; ++ki) {
                const ExpertTensorKind kind =
                    ki == 0 ? ExpertTensorKind::kGate
                            : (ki == 1 ? ExpertTensorKind::kDown : ExpertTensorKind::kUp);
                std::vector<bool> seen(n_expert, false);
                uint32_t count = 0;
                for (const TensorInfo &t : legacy) {
                    if (t.layer == layer && t.kind == kind) {
                        if (t.expert >= n_expert) {
                            gguf_free(gc);
                            return fail(err,
                                        "legacy per-expert tensor '" + t.name +
                                            "' has expert index " +
                                            std::to_string(t.expert) + " >= declared " +
                                            arch +
                                            ".expert_count=" + std::to_string(n_expert));
                        }
                        if (!seen[t.expert]) {
                            seen[t.expert] = true;
                            ++count;
                        }
                    }
                }
                if (count != n_expert) {
                    const char *kn = kind == ExpertTensorKind::kGate   ? "ffn_gate"
                                     : kind == ExpertTensorKind::kDown ? "ffn_down"
                                                                       : "ffn_up";
                    gguf_free(gc);
                    return fail(err, "legacy per-expert MoE layer " +
                                         std::to_string(layer) + " kind " + kn + " has " +
                                         std::to_string(count) + " experts but " + arch +
                                         ".expert_count=" + std::to_string(n_expert) +
                                         " (every (layer,kind) must have exactly "
                                         "expert_count per-expert tensors)");
                }
            }
        }
        n_experts_global = n_expert;
    }

    const uint32_t n_layers = (experts.empty() && legacy.empty()) ? 0 : (max_layer + 1);

    // --- Metadata cross-check: FAIL LOUDLY on a misclassified MoE -----------
    // The classifier recognizes experts by tensor name only. On the real
    // Mixtral every expert tensor missed that match, so experts stayed empty
    // and the packer wrote a structurally-valid 0-expert .strata; the engine
    // then held the whole ~19 GB model resident -> SIGKILL (OOM). The GGUF
    // metadata is authoritative: if it DECLARES a MoE (expert_count > 0) but
    // the name-based classifier recognized ZERO streamable expert tensors
    // (NEITHER stacked NOR legacy), refuse to write an OOM-bomb .strata and
    // emit an actionable diagnostic that names the declared counts and lists a
    // few of the tensor names actually seen, so an operator can see WHY the
    // match failed.
    if (meta_expert_count > 0 && experts.empty() && legacy.empty()) {
        std::string sample;
        int shown = 0;
        for (const TensorInfo &t : trunk) {
            // Prefer the ffn_*/blk.* tensors an operator would expect to be
            // the experts, so the sample is maximally diagnostic.
            if (t.name.find("ffn_") != std::string::npos ||
                t.name.compare(0, 4, "blk.") == 0) {
                if (shown != 0) sample += ", ";
                sample += "'" + t.name + "'";
                if (++shown >= 8) break;
            }
        }
        if (shown == 0) {
            // No ffn_/blk. names at all; just list the first few tensors.
            for (const TensorInfo &t : trunk) {
                if (shown != 0) sample += ", ";
                sample += "'" + t.name + "'";
                if (++shown >= 8) break;
            }
        }
        gguf_free(gc);
        return fail(err, "GGUF metadata declares a MoE (" + arch +
                             ".expert_count=" + std::to_string(meta_expert_count) + ", " +
                             arch + ".block_count=" + std::to_string(meta_block_count) +
                             ") but zero streamable expert tensors were recognized "
                             "by name; refusing to write a 0-expert .strata that "
                             "would load fully resident (OOM). Tensor names seen: " +
                             (sample.empty() ? "(none)" : sample));
    }

    // Group the three (gate/up/down) kinds per (layer,expert). For the STACKED
    // path a single tensor per (layer,kind) holds all experts; for the LEGACY
    // path each (layer,kind,expert) is its own tensor.
    auto find_stacked = [&](uint32_t layer, ExpertTensorKind kind) -> const TensorInfo * {
        for (const TensorInfo &t : experts) {
            if (t.layer == layer && t.kind == kind) return &t;
        }
        return nullptr;
    };
    auto find_legacy = [&](uint32_t layer, ExpertTensorKind kind,
                           uint32_t e) -> const TensorInfo * {
        for (const TensorInfo &t : legacy) {
            if (t.layer == layer && t.kind == kind && t.expert == e) return &t;
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
        std::fclose(out);
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "write superblock failed");
    }

    // --- Embedded GGUF metadata blob ----------------------------------------
    // STACKED input (and non-MoE): copy [0, data_offset) VERBATIM so the
    // fast-path behavior is unchanged. LEGACY input: SYNTHESIZE a fresh
    // metadata blob that declares stacked _exps tensors in place of the
    // per-expert tensors, so engine::load_strata_weights and
    // ggml_model.cpp::load_strata_model (both parse this embedded GGUF) see
    // ONLY stacked names - the engine's mul_mat_id path is unchanged.
    std::vector<uint8_t> synth_meta;
    if (legacy_mode) {
        gguf_context *mc = gguf_init_empty();
        if (mc == nullptr) {
            std::fclose(out);
            std::fclose(src);
            gguf_free(gc);
            return fail(err, "synthesize metadata: gguf_init_empty failed");
        }
        // Copy ALL source KV metadata verbatim (architecture, tokenizer,
        // expert_count, etc.) by gguf type.
        const int64_t n_kv = gguf_get_n_kv(gc);
        for (int64_t k = 0; k < n_kv; ++k) {
            if (!copy_kv(mc, gc, k)) {
                const char *key = gguf_get_key(gc, k);
                gguf_free(mc);
                std::fclose(out);
                std::fclose(src);
                gguf_free(gc);
                return fail(err, "synthesize metadata: unsupported KV type for '" +
                                     std::string(key != nullptr ? key : "?") + "'");
            }
        }

        // A no_alloc ggml context to mint tensor-info records (shape/type/name)
        // without allocating any tensor data. gguf_add_tensor records the info.
        ggml_init_params tp{};
        tp.mem_size = ggml_tensor_overhead() * (static_cast<size_t>(trunk.size()) +
                                                static_cast<size_t>(n_layers) * 3u + 8u);
        tp.mem_buffer = nullptr;
        tp.no_alloc = true;
        ggml_context *tctx = ggml_init(tp);
        if (tctx == nullptr) {
            gguf_free(mc);
            std::fclose(out);
            std::fclose(src);
            gguf_free(gc);
            return fail(err, "synthesize metadata: ggml_init failed");
        }

        // (a) Every TRUNK tensor verbatim, in source order (the engine fills
        // the trunk block in metadata order, so this order must match the
        // trunk-block write order below, which is also `trunk` order).
        bool synth_ok = true;
        for (const TensorInfo &t : trunk) {
            ggml_tensor *tt = ggml_new_tensor(tctx, t.type, GGML_MAX_DIMS, t.ne);
            if (tt == nullptr) {
                synth_ok = false;
                break;
            }
            ggml_set_name(tt, t.name.c_str());
            gguf_add_tensor(mc, tt);
        }
        // (b) ONE synthesized stacked tensor per (layer,kind). The stacked ne
        // is {ne0, ne1, n_expert} where (ne0, ne1) come from any one of the
        // per-expert 2-D tensors of that (layer,kind) - they are identical
        // across experts. gate/up: {n_embd, n_ff}; down: {n_ff, n_embd}.
        for (uint32_t layer = 0; synth_ok && layer < n_layers; ++layer) {
            for (int ki = 0; ki < 3; ++ki) {
                const ExpertTensorKind kind =
                    ki == 0 ? ExpertTensorKind::kGate
                            : (ki == 1 ? ExpertTensorKind::kDown : ExpertTensorKind::kUp);
                const TensorInfo *any = find_legacy(layer, kind, 0);
                if (any == nullptr) {
                    synth_ok = false;
                    break;
                }
                int64_t ne3[GGML_MAX_DIMS] = {any->ne[0], any->ne[1],
                                              static_cast<int64_t>(n_experts_global), 1};
                ggml_tensor *tt = ggml_new_tensor(tctx, any->type, 3, ne3);
                if (tt == nullptr) {
                    synth_ok = false;
                    break;
                }
                char nm[64];
                std::snprintf(nm, sizeof(nm), "blk.%u.%s.weight", layer,
                              kind_stacked_suffix(kind));
                ggml_set_name(tt, nm);
                gguf_add_tensor(mc, tt);
            }
        }
        if (!synth_ok) {
            ggml_free(tctx);
            gguf_free(mc);
            std::fclose(out);
            std::fclose(src);
            gguf_free(gc);
            return fail(err, "synthesize metadata: tensor-info build failed");
        }

        // Serialize META-ONLY (no tensor data): the embedded blob the engine
        // and the vocab sidecar parse. The resulting data_offset + the stacked
        // tensor sizes stay self-consistent (gguf lays tensor data out
        // contiguously from the synthesized tensor-info), satisfying the
        // sidecar bounds check in load_strata_model.
        const size_t meta_size = gguf_get_meta_size(mc);
        synth_meta.resize(meta_size);
        gguf_get_meta_data(mc, synth_meta.data());

        ggml_free(tctx);
        gguf_free(mc);
    }

    sb.gguf_meta_offset = pos;
    if (legacy_mode) {
        sb.gguf_meta_size = synth_meta.size();
        if (!write_bytes(out, synth_meta.data(), synth_meta.size(), pos)) {
            std::fclose(out);
            std::fclose(src);
            gguf_free(gc);
            return fail(err, "write synthesized GGUF metadata failed");
        }
    } else {
        sb.gguf_meta_size = gguf_meta_size;
        if (!copy_range(out, src, 0, gguf_meta_size, pos)) {
            std::fclose(out);
            std::fclose(src);
            gguf_free(gc);
            return fail(err, "copy GGUF metadata failed");
        }
    }

    // --- 3. Expert index (count known now; offsets backfilled below) --------
    // v1 writes experts in {layer-major, expert-minor} order.
    std::vector<ExpertIndexEntry> index;
    index.reserve(static_cast<size_t>(n_layers) * n_experts_global);
    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        for (uint32_t e = 0; e < n_experts_global; ++e) {
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
        std::fclose(out);
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "write index placeholder failed");
    }

    // --- 4. Trunk block (4 KiB-aligned start, tensors contiguous) -----------
    const uint64_t trunk_start = align_up(pos, kIoAlignment);
    if (!pad_to(out, trunk_start, pos)) {
        std::fclose(out);
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "pad to trunk failed");
    }
    sb.trunk_offset = trunk_start;
    for (const TensorInfo &t : trunk) {
        if (!copy_range(out, src, t.src_offset, t.size, pos)) {
            std::fclose(out);
            std::fclose(src);
            gguf_free(gc);
            return fail(err, "copy trunk tensor '" + t.name + "' failed");
        }
    }
    sb.trunk_size = pos - trunk_start;

    // --- 5. Expert region: one 4 KiB-aligned blob per (layer,expert) --------
    const uint64_t region_start = align_up(pos, kIoAlignment);
    if (!pad_to(out, region_start, pos)) {
        std::fclose(out);
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "pad to expert region failed");
    }
    sb.expert_region_offset = region_start;

    for (ExpertIndexEntry &entry : index) {
        const uint32_t layer = entry.layer;
        const uint32_t e = entry.expert;

        // Resolve the byte source of each (layer,expert,kind) slice. For the
        // stacked path the slice is the tensor's nb02 chunk at offset e*nb02;
        // for the legacy path the slice is the whole standalone per-expert
        // tensor. We represent both as {abs_offset, length} so one emit() lambda
        // copies them identically (VERBATIM bytes, no requant/reshape).
        struct Slice {
            uint64_t off = 0;
            uint64_t len = 0;
            bool ok = false;
        };
        auto resolve = [&](ExpertTensorKind kind) -> Slice {
            Slice s;
            if (legacy_mode) {
                const TensorInfo *t = find_legacy(layer, kind, e);
                if (t != nullptr) {
                    s.off = t->src_offset;
                    s.len = t->size;
                    s.ok = true;
                }
            } else {
                const TensorInfo *t = find_stacked(layer, kind);
                if (t != nullptr) {
                    s.off = t->src_offset + static_cast<uint64_t>(e) * t->nb02;
                    s.len = t->nb02;
                    s.ok = true;
                }
            }
            return s;
        };

        const Slice gate = resolve(ExpertTensorKind::kGate);
        const Slice up = resolve(ExpertTensorKind::kUp);
        const Slice down = resolve(ExpertTensorKind::kDown);
        Slice fused;
        if (!legacy_mode) {
            const TensorInfo *t = find_stacked(layer, ExpertTensorKind::kGateUp);
            if (t != nullptr) {
                fused.off = t->src_offset + static_cast<uint64_t>(e) * t->nb02;
                fused.len = t->nb02;
                fused.ok = true;
            }
        }

        // 4 KiB-align the blob START (only the start; the gate/up/down slices
        // are tightly packed inside so the reader derives slice lengths from
        // the *_rel gaps, and one read_at yields all three kinds).
        const uint64_t blob_start = align_up(pos, kIoAlignment);
        if (!pad_to(out, blob_start, pos)) {
            std::fclose(out);
            std::fclose(src);
            gguf_free(gc);
            return fail(err, "pad to expert blob failed");
        }
        entry.blob_offset = blob_start;

        uint64_t rel = 0;
        auto emit = [&](const Slice &s, uint64_t &rel_out) -> bool {
            rel_out = rel;
            if (!copy_range(out, src, s.off, s.len, pos)) return false;
            rel += s.len;
            return true;
        };

        bool ok = true;
        if (fused.ok) {
            // Fused gate_up arch: bundle gate_up then down. gate_rel == up_rel
            // (one fused slice), down_rel follows.
            ok = ok && emit(fused, entry.gate_rel);
            entry.up_rel = entry.gate_rel;
            ok = ok && down.ok && emit(down, entry.down_rel);
        } else {
            if (!gate.ok || !up.ok || !down.ok) {
                std::fclose(out);
                std::fclose(src);
                gguf_free(gc);
                return fail(err, "layer " + std::to_string(layer) +
                                     " is missing a gate/up/down expert tensor");
            }
            ok = ok && emit(gate, entry.gate_rel);
            ok = ok && emit(up, entry.up_rel);
            ok = ok && emit(down, entry.down_rel);
        }
        if (!ok) {
            std::fclose(out);
            std::fclose(src);
            gguf_free(gc);
            return fail(err, "copy expert blob (layer " + std::to_string(layer) +
                                 ", expert " + std::to_string(e) + ") failed");
        }
        entry.blob_length = pos - blob_start;
    }
    sb.expert_region_size = pos - region_start;

    // --- 6. Backfill the index with real offsets ---------------------------
    if (std::fseek(out, static_cast<long>(sb.index_offset), SEEK_SET) != 0) {
        std::fclose(out);
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "seek to index failed");
    }
    if (index_bytes > 0 && std::fwrite(index.data(), 1, static_cast<size_t>(index_bytes),
                                       out) != index_bytes) {
        std::fclose(out);
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "rewrite index failed");
    }

    // --- Backfill the superblock with real offsets + CRC --------------------
    sb.header_crc32 = strata_crc32(&sb, kStrataHeaderCrcOffset);
    if (std::fseek(out, 0, SEEK_SET) != 0) {
        std::fclose(out);
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "seek to superblock failed");
    }
    if (std::fwrite(&sb, 1, kStrataSuperblockSize, out) != kStrataSuperblockSize) {
        std::fclose(out);
        std::fclose(src);
        gguf_free(gc);
        return fail(err, "rewrite superblock failed");
    }

    std::fclose(out);
    std::fclose(src);
    gguf_free(gc);
    return true;
}

}  // namespace sf

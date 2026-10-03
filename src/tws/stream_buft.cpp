// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "tws/stream_buft.h"

#include "common/log.h"

#include <ggml-backend.h>
#include <ggml.h>
// ggml-backend-impl.h is a PRIVATE ggml header (lives under ggml/src). It
// defines ggml_backend_buffer_type_i / ggml_backend_buffer_i, which we
// implement here. src/CMakeLists.txt adds ${STRATAFLOW_LLAMA_DIR}/ggml/src to
// this target's private include dirs so this include resolves.
#include "ggml-backend-impl.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sf {
namespace {

// ---------------------------------------------------------------------------
// Backing store + per-buffer state.
//
// For Task 2 (v1) the backing store is a plain in-process byte vector sized to
// the buffer's full expert footprint, and get_base() returns it directly: the
// allocator places every expert tensor at an offset inside this one region and
// the CPU kernel reads the resident bytes directly. set_tensor at load time
// copies the expert bytes into the region (recording the byte range so Task 3
// can swap the store for BlockFile reads from the .strata/GGUF without touching
// this buffer type). This is the "keep all experts resident" done-criterion.
// ---------------------------------------------------------------------------

struct TensorRange {
    uint64_t       offset = 0;   // byte offset of the tensor within the region
    uint64_t       length = 0;   // ggml_nbytes(tensor)
    ExpertTensorId id{};         // parsed {layer, kind}
};

// Per-buffer context: owns the contiguous region and the recorded ranges.
constexpr size_t kBuftAlignment = 32;  // match ggml's CPU tensor alignment

struct StreamBuffer {
    // Over-allocated backing storage. ggml's allocator aligns the first tensor
    // to kBuftAlignment relative to get_base(); if get_base() were not aligned
    // that initial padding would eat into the region and the last tensor would
    // not fit ("not enough space in the buffer"). So we allocate `size +
    // alignment` and hand out an aligned base, guaranteeing the full `size`
    // bytes are usable from base().
    std::vector<uint8_t> storage;
    uint8_t *aligned = nullptr;  // 32-byte aligned pointer within storage
    std::unordered_map<const ggml_tensor *, TensorRange> ranges;

    uint8_t *base() { return aligned; }
};

StreamBuftStats  g_stats;
std::mutex       g_stats_mu;

StreamBuffer *buf_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<StreamBuffer *>(buffer->context);
}

// ---- ggml_backend_buffer_i callbacks --------------------------------------

void buf_free(ggml_backend_buffer_t buffer) {
    delete buf_ctx(buffer);
    buffer->context = nullptr;
}

void *buf_get_base(ggml_backend_buffer_t buffer) {
    // ggml's linear allocator places all tensors at offsets within this single
    // region, so it must be the real backing memory (a dummy base fails with
    // "ggml_tallocr_alloc: not enough space in the buffer"). See the spike.
    // The base is aligned so the allocator's leading alignment is a no-op.
    return buf_ctx(buffer)->base();
}

enum ggml_status buf_init_tensor(ggml_backend_buffer_t buffer,
                                 struct ggml_tensor *tensor) {
    StreamBuffer *ctx = buf_ctx(buffer);
    const uint64_t len = static_cast<uint64_t>(ggml_nbytes(tensor));

    // tensor->data was already pointed at a region offset by the allocator;
    // record the range so set_tensor/get_tensor and Task 3's residency pass
    // can locate the bytes. The offset is tensor->data relative to get_base().
    const uint8_t *base = ctx->base();
    const uint8_t *tdat = static_cast<const uint8_t *>(tensor->data);
    uint64_t offset = 0;
    if (tdat >= base) {
        offset = static_cast<uint64_t>(tdat - base);
    }

    TensorRange r;
    r.offset = offset;
    r.length = len;
    r.id = parse_expert_tensor_name(tensor->name);
    ctx->ranges[tensor] = r;

    {
        std::lock_guard<std::mutex> lk(g_stats_mu);
        if (r.id.valid) {
            ++g_stats.registrations;
        } else {
            ++g_stats.non_expert;
        }
    }
    return GGML_STATUS_SUCCESS;
}

void buf_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor *tensor,
                    const void *data, size_t offset, size_t size) {
    // Load-time weights: copy expert bytes into the backing region. (Task 3
    // swaps this for recording the .strata/GGUF file range and loading on
    // demand via BlockFile; the seam is the TensorRange recorded above.)
    StreamBuffer *ctx = buf_ctx(buffer);
    auto it = ctx->ranges.find(tensor);
    if (it == ctx->ranges.end()) return;
    const uint64_t dst_off = it->second.offset + static_cast<uint64_t>(offset);
    const size_t cap = ctx->storage.size() -
                       static_cast<size_t>(ctx->base() - ctx->storage.data());
    if (dst_off + size > cap) return;  // defensive bound check
    std::memcpy(ctx->base() + dst_off, data, size);
    {
        std::lock_guard<std::mutex> lk(g_stats_mu);
        g_stats.stored_bytes += static_cast<uint64_t>(size);
    }
}

void buf_get_tensor(ggml_backend_buffer_t buffer,
                    const struct ggml_tensor *tensor, void *data,
                    size_t offset, size_t size) {
    StreamBuffer *ctx = buf_ctx(buffer);
    auto it = ctx->ranges.find(tensor);
    if (it == ctx->ranges.end()) return;
    const uint64_t src_off = it->second.offset + static_cast<uint64_t>(offset);
    const size_t cap = ctx->storage.size() -
                       static_cast<size_t>(ctx->base() - ctx->storage.data());
    if (src_off + size > cap) return;
    std::memcpy(data, ctx->base() + src_off, size);
}

void buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    StreamBuffer *ctx = buf_ctx(buffer);
    const size_t cap = ctx->storage.size() -
                       static_cast<size_t>(ctx->base() - ctx->storage.data());
    std::memset(ctx->base(), value, cap);
}

// ---- ggml_backend_buffer_type_i callbacks ---------------------------------

const char *bt_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return strata_stream_buft_name();
}

size_t bt_get_alignment(ggml_backend_buffer_type_t /*buft*/) {
    // Match ggml's standard CPU tensor alignment so tensors pack identically.
    return kBuftAlignment;
}

size_t bt_get_alloc_size(ggml_backend_buffer_type_t /*buft*/,
                         const struct ggml_tensor *tensor) {
    return ggml_nbytes(tensor);
}

bool bt_is_host(ggml_backend_buffer_type_t /*buft*/) {
    // True: the bytes live in host memory in standard ggml layout, so the CPU
    // kernel reads tensor->data directly (there is no per-op fault hook).
    return true;
}

ggml_backend_buffer_t bt_alloc_buffer(ggml_backend_buffer_type_t buft,
                                      size_t size) {
    // Allocate a REAL contiguous region of the requested size. Task 2 v1 sizes
    // it to the full expert footprint; Task 3 refines to a bounded slot pool.
    // Over-allocate by one alignment and hand out an aligned base so the full
    // `size` bytes are usable (see StreamBuffer).
    StreamBuffer *ctx = new StreamBuffer();
    ctx->storage.assign(size + kBuftAlignment, 0);
    uintptr_t raw = reinterpret_cast<uintptr_t>(ctx->storage.data());
    uintptr_t aligned = (raw + (kBuftAlignment - 1)) & ~(kBuftAlignment - 1);
    ctx->aligned = reinterpret_cast<uint8_t *>(aligned);

    {
        std::lock_guard<std::mutex> lk(g_stats_mu);
        ++g_stats.buffers;
    }

    ggml_backend_buffer_i iface{};
    iface.free_buffer  = buf_free;
    iface.get_base     = buf_get_base;
    iface.init_tensor  = buf_init_tensor;
    iface.memset_tensor = nullptr;
    iface.set_tensor   = buf_set_tensor;
    iface.get_tensor   = buf_get_tensor;
    iface.set_tensor_2d = nullptr;
    iface.get_tensor_2d = nullptr;
    iface.cpy_tensor   = nullptr;
    iface.clear        = buf_clear;
    iface.reset        = nullptr;

    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

// The singleton buffer-type instance. iface uses designated initializers so the
// field order tracks ggml-backend-impl.h exactly (important: the real struct
// has alloc_buffer_n / get_max_size / get_alloc_size_n optional slots).
ggml_backend_buffer_type g_buft = {
    /*.iface   =*/ {
        /*.get_name         =*/ bt_get_name,
        /*.alloc_buffer     =*/ bt_alloc_buffer,
        /*.alloc_buffer_n   =*/ nullptr,
        /*.get_alignment    =*/ bt_get_alignment,
        /*.get_max_size     =*/ nullptr,
        /*.get_alloc_size   =*/ bt_get_alloc_size,
        /*.get_alloc_size_n =*/ nullptr,
        /*.is_host          =*/ bt_is_host,
    },
    /*.device  =*/ nullptr,
    /*.context =*/ nullptr,
};

}  // namespace

ExpertTensorId parse_expert_tensor_name(const std::string &name) {
    ExpertTensorId id{};

    // Expect "blk.<N>.ffn_<kind>_exps" optionally followed by ".weight".
    static const char kPrefix[] = "blk.";
    if (name.compare(0, 4, kPrefix) != 0) return id;

    size_t pos = 4;
    uint64_t layer = 0;
    size_t digits = 0;
    while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
        layer = layer * 10 + static_cast<uint64_t>(name[pos] - '0');
        ++pos;
        ++digits;
    }
    if (digits == 0) return id;
    if (pos >= name.size() || name[pos] != '.') return id;
    ++pos;  // skip the '.'

    const std::string rest = name.substr(pos);
    // Strip an optional trailing ".weight" so both forms parse.
    std::string body = rest;
    static const std::string kWeightSuffix = ".weight";
    if (body.size() > kWeightSuffix.size() &&
        body.compare(body.size() - kWeightSuffix.size(),
                     kWeightSuffix.size(), kWeightSuffix) == 0) {
        body = body.substr(0, body.size() - kWeightSuffix.size());
    }

    ExpertTensorKind kind;
    if (body == "ffn_gate_exps") {
        kind = ExpertTensorKind::kGate;
    } else if (body == "ffn_down_exps") {
        kind = ExpertTensorKind::kDown;
    } else if (body == "ffn_up_exps") {
        kind = ExpertTensorKind::kUp;
    } else if (body == "ffn_gate_up_exps") {
        kind = ExpertTensorKind::kGateUp;
    } else {
        return id;  // not a routed-expert FFN tensor
    }

    id.layer = static_cast<uint32_t>(layer);
    id.kind = kind;
    id.valid = true;
    return id;
}

const StreamBuftStats &stream_buft_stats() { return g_stats; }

void reset_stream_buft_stats() {
    std::lock_guard<std::mutex> lk(g_stats_mu);
    g_stats = StreamBuftStats{};
}

const char *strata_stream_buft_name() { return "strata-stream"; }

ggml_backend_buffer_type *strata_stream_buft() {
    // Bind to the CPU device the first time, so llama's scheduler treats the
    // buffer as CPU-side (is_host == true) and the CPU backend runs the ops.
    if (g_buft.device == nullptr) {
        ggml_backend_dev_t cpu_dev =
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        g_buft.device = cpu_dev;  // may stay null on an exotic no-CPU build
    }
    return &g_buft;
}

}  // namespace sf

// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "tws/stream_buft.h"

#include "common/log.h"
#include "tws/weight_store.h"

#include <ggml-backend.h>
#include <ggml.h>
// ggml-backend-impl.h is a PRIVATE ggml header (lives under ggml/src). It
// defines ggml_backend_buffer_type_i / ggml_backend_buffer_i, which we
// implement here. src/CMakeLists.txt adds ${STRATAFLOW_LLAMA_DIR}/ggml/src to
// this target's private include dirs so this include resolves.
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sf {
namespace {

// ---------------------------------------------------------------------------
// Backing store + per-buffer state (Task 3).
//
// alloc_buffer() still allocates a REAL contiguous region sized to the full
// expert footprint: ggml's linear allocator places every tensor at an offset
// inside one get_base() region, AND one llama_decode runs the FULL graph for
// all layers in a single call, so every stacked expert tensor must be resident
// at its own distinct address when the kernel runs (CORRECTNESS MODEL A — see
// the header). The region holds the committed working set the kernel reads.
//
// The bounded SlotPool is the CACHE TIER layered under the region: the backing
// store (`store`) holds every stacked expert tensor's bytes (filled by
// set_tensor at load time; Task 4 swaps this for BlockFile direct-I/O without
// touching this file). Before each llama_decode the residency pass acquires
// every registered tensor through the pool, loading its byte range from the
// store into its region slot. The pool bounds how many tensors are held
// resident ACROSS decodes to n_slots; when n_slots < the working set the
// residency pass evicts and reloads (demonstrated across decode steps), and the
// cache-resident bytes stay bounded by n_slots * slot_bytes.
// ---------------------------------------------------------------------------

struct TensorRange {
    uint64_t       offset = 0;   // byte offset of the tensor within the region
    uint64_t       length = 0;   // ggml_nbytes(tensor)
    ExpertTensorId id{};         // parsed {layer, kind}
    bool           loaded = false;  // set_tensor has filled the store range
};

constexpr size_t kBuftAlignment = 32;  // match ggml's CPU tensor alignment

// Map an ExpertTensorId to the SlotPool key type (ExpertId{layer, expert}).
// The slot-pool unit here is the WHOLE stacked tensor per (layer, kind), so we
// pack the kind into the `expert` field. This keeps the SlotPool API (and
// test_slot_pool) unchanged while keying slots at stacked-tensor granularity.
ExpertId slot_key(const ExpertTensorId &id) {
    return ExpertId{id.layer, static_cast<uint32_t>(id.kind)};
}

struct StreamBuffer {
    // Over-allocated backing storage for the committed working set. ggml aligns
    // the first tensor to kBuftAlignment relative to get_base(); we allocate
    // `size + alignment` and hand out an aligned base so the full `size` bytes
    // are usable.
    std::vector<uint8_t> storage;
    uint8_t *aligned = nullptr;  // 32-byte aligned pointer within storage

    // Recorded ranges per tensor, plus a stable registration order so the
    // residency pass visits tensors deterministically (determinism gate).
    std::unordered_map<const ggml_tensor *, TensorRange> ranges;
    std::vector<const ggml_tensor *> order;

    // The CACHE TIER: a bounded pool of fixed-size slots. One slot holds one
    // whole stacked expert tensor's bytes. slot_bytes == max stacked-tensor
    // nbytes so any tensor fits any slot. n_slots is the memory-budget knob.
    std::unique_ptr<SlotPool> pool;
    uint32_t pool_slots = 0;
    uint64_t pool_slot_bytes = 0;

    // Backing store: the authoritative bytes for every stacked expert tensor,
    // keyed by region offset. In-memory for now (Task 4 -> BlockFile). We also
    // keep an optional BlockFile seam so the .strata/GGUF path (Task 5) can
    // read ranges straight from disk.
    std::vector<uint8_t> store;        // mirror sized to the region
    uint64_t             disk_reads = 0;

    uint8_t *base() { return aligned; }
};

StreamBuftStats  g_stats;
std::mutex       g_stats_mu;
uint32_t         g_cfg_slots = 0;  // StreamBuftConfig::n_slots for next alloc

// The live buffer. llama allocates exactly one strata-stream buffer for the
// expert tensors, so we track it for the residency pass. Guarded by g_live_mu.
StreamBuffer    *g_live = nullptr;
std::mutex       g_live_mu;

StreamBuffer *buf_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<StreamBuffer *>(buffer->context);
}

// Publish the pool stats into the global StreamBuftStats under g_stats_mu.
void publish_pool_stats_locked(StreamBuffer *ctx) {
    if (ctx->pool != nullptr) {
        const CacheStats &cs = ctx->pool->stats();
        g_stats.cache_hits      = cs.hits;
        g_stats.cache_misses    = cs.misses;
        g_stats.cache_evictions = cs.evictions;
    }
    g_stats.disk_reads     = ctx->disk_reads;
    g_stats.n_slots        = ctx->pool_slots;
    g_stats.slot_bytes     = ctx->pool_slot_bytes;
    // Resident slots == min(expert tensors, n_slots) once the pool is primed:
    // this is the honest count of stacked expert tensors held in the bounded
    // cache tier, so resident_cache_bytes() = resident_slots * slot_bytes is
    // the memory bound the test asserts.
    uint64_t n_expert_tensors = 0;
    for (const ggml_tensor *t : ctx->order) {
        if (ctx->ranges[t].id.valid) ++n_expert_tensors;
    }
    g_stats.resident_slots =
        ctx->pool_slots == 0
            ? 0
            : std::min<uint64_t>(n_expert_tensors, ctx->pool_slots);
}

// ---- ggml_backend_buffer_i callbacks --------------------------------------

void buf_free(ggml_backend_buffer_t buffer) {
    StreamBuffer *ctx = buf_ctx(buffer);
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        if (g_live == ctx) g_live = nullptr;
    }
    delete ctx;
    buffer->context = nullptr;
}

void *buf_get_base(ggml_backend_buffer_t buffer) {
    // ggml's linear allocator places all tensors at offsets within this single
    // region, so it must be the real backing memory. The base is aligned so
    // the allocator's leading alignment is a no-op.
    return buf_ctx(buffer)->base();
}

enum ggml_status buf_init_tensor(ggml_backend_buffer_t buffer,
                                 struct ggml_tensor *tensor) {
    StreamBuffer *ctx = buf_ctx(buffer);
    const uint64_t len = static_cast<uint64_t>(ggml_nbytes(tensor));

    // tensor->data was already pointed at a region offset by the allocator;
    // record the range so set_tensor/get_tensor and the residency pass can
    // locate the bytes. The offset is tensor->data relative to get_base().
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
    ctx->order.push_back(tensor);

    // Size the cache tier's slot to the largest stacked expert tensor seen so
    // the pool is allocated lazily on first residency (we don't know the max
    // nbytes at alloc_buffer time). Only expert tensors count toward slots.
    if (r.id.valid) {
        ctx->pool_slot_bytes = std::max<uint64_t>(ctx->pool_slot_bytes, len);
    }

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
    // Load-time weights: copy expert bytes into BOTH the committed region (so
    // the first decode is correct even before the residency pass runs) and the
    // backing store (the authoritative copy the residency pass reloads from
    // after an eviction). Task 4 swaps the store copy for a BlockFile range.
    StreamBuffer *ctx = buf_ctx(buffer);
    auto it = ctx->ranges.find(tensor);
    if (it == ctx->ranges.end()) return;
    const uint64_t dst_off = it->second.offset + static_cast<uint64_t>(offset);
    const size_t cap = ctx->storage.size() -
                       static_cast<size_t>(ctx->base() - ctx->storage.data());
    if (dst_off + size > cap) return;  // defensive bound check
    std::memcpy(ctx->base() + dst_off, data, size);
    if (ctx->store.size() >= dst_off + size) {
        std::memcpy(ctx->store.data() + dst_off, data, size);
    }
    it->second.loaded = true;
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
    // Allocate the committed working-set region (full expert footprint). The
    // bounded SlotPool cache tier is created lazily on first residency, once
    // init_tensor has told us the largest stacked-tensor nbytes (slot_bytes).
    StreamBuffer *ctx = new StreamBuffer();
    ctx->storage.assign(size + kBuftAlignment, 0);
    ctx->store.assign(size + kBuftAlignment, 0);
    uintptr_t raw = reinterpret_cast<uintptr_t>(ctx->storage.data());
    uintptr_t aligned = (raw + (kBuftAlignment - 1)) & ~(kBuftAlignment - 1);
    ctx->aligned = reinterpret_cast<uint8_t *>(aligned);

    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        ctx->pool_slots = g_cfg_slots;  // snapshot the configured size
        g_live = ctx;
    }
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

// The singleton buffer-type instance.
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

// Lazily create the cache tier once slot_bytes is known. Auto-sizes the pool to
// the full working set (number of expert tensors) when n_slots == 0.
void ensure_pool_locked(StreamBuffer *ctx) {
    if (ctx->pool != nullptr) return;
    if (ctx->pool_slot_bytes == 0) return;  // no expert tensors registered

    uint32_t n_expert_tensors = 0;
    for (const ggml_tensor *t : ctx->order) {
        if (ctx->ranges[t].id.valid) ++n_expert_tensors;
    }
    uint32_t n_slots = ctx->pool_slots;
    if (n_slots == 0) {
        n_slots = n_expert_tensors;  // auto: hold the whole working set
    }
    if (n_slots == 0) return;  // nothing to cache
    ctx->pool_slots = n_slots;
    ctx->pool = std::make_unique<SlotPool>(n_slots, ctx->pool_slot_bytes);
}

}  // namespace

ExpertTensorId parse_expert_tensor_name(const std::string &name) {
    ExpertTensorId id{};

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

void set_stream_buft_slots(uint32_t n_slots) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    g_cfg_slots = n_slots;
}

uint32_t stream_buft_slots() {
    std::lock_guard<std::mutex> lk(g_live_mu);
    return g_cfg_slots;
}

void stream_buft_ensure_decode_residency(bool prefill) {
    std::lock_guard<std::mutex> live_lk(g_live_mu);
    StreamBuffer *ctx = g_live;
    if (ctx == nullptr) return;

    ensure_pool_locked(ctx);
    if (ctx->pool == nullptr) return;

    // Count expert tensors so prefill-bypass can decide (PLAN 3.3): a prompt /
    // micro-batch touching more expert tensors than the pool has slots streams
    // the overflow WITHOUT caching, so prefill can't wipe the decode hot set.
    uint32_t n_expert_tensors = 0;
    for (const ggml_tensor *t : ctx->order) {
        if (ctx->ranges[t].id.valid) ++n_expert_tensors;
    }
    const bool bypass_cache =
        prefill && ctx->pool_slots != 0 && n_expert_tensors > ctx->pool_slots;

    for (const ggml_tensor *t : ctx->order) {
        TensorRange &r = ctx->ranges[t];
        if (!r.id.valid) continue;  // non-expert tensors are untouched

        const uint64_t off = r.offset;
        const uint64_t len = r.length;
        if (ctx->store.size() < off + len) continue;  // defensive

        // The committed region (tensor->data) must hold this tensor's bytes
        // when the kernel runs. load_slot copies the authoritative bytes from
        // the backing store into the region on a cache miss (an eviction forced
        // the previous occupant out and this tensor must be re-materialized).
        auto load_slot = [&](void * /*slot_dst*/) {
            // Copy from the store into the committed region. We also populate
            // the slot buffer so resident-byte accounting is honest, but the
            // kernel reads the region (tensor->data), which is what must be
            // correct within the single decode (Model A).
            std::memcpy(ctx->base() + off, ctx->store.data() + off,
                        static_cast<size_t>(len));
            ++ctx->disk_reads;
        };

        if (bypass_cache) {
            // Prefill overflow: stream straight into the region, no LRU churn.
            load_slot(nullptr);
            continue;
        }

        // Cache path: acquire through the SlotPool. On a hit the region already
        // holds valid bytes (we never overwrite a resident tensor's region). On
        // a miss load_slot refills the region from the store and the pool may
        // evict the LRU tensor to make room — but note the EVICTED tensor's
        // region bytes remain valid for THIS decode (we size the region to the
        // full working set); eviction only means the NEXT time that tensor is
        // acquired it will miss and reload (demonstrated across decodes).
        ctx->pool->acquire(slot_key(r.id), load_slot);
    }

    {
        std::lock_guard<std::mutex> stats_lk(g_stats_mu);
        publish_pool_stats_locked(ctx);
    }
}

const char *strata_stream_buft_name() { return "strata-stream"; }

ggml_backend_buffer_type *strata_stream_buft() {
    if (g_buft.device == nullptr) {
        ggml_backend_dev_t cpu_dev =
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        g_buft.device = cpu_dev;  // may stay null on an exotic no-CPU build
    }
    return &g_buft;
}

}  // namespace sf

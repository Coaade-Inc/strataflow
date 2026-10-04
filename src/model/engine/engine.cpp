// Engine core (EC-3) implementation: build and run OUR OWN ggml graphs for the
// llama-arch MoE forward on the ggml CPU backend, now in PER-LAYER SEGMENTS so
// only the top-k routed experts per layer are made resident (sections 3.2, 3.3,
// 3.5). The op math is lifted verbatim from docs/ENGINE_CORE_DESIGN.md section
// 6.3 and stays byte-identical to EC-1/EC-2 (the spike measured argmax match,
// max |logit delta| 5.3e-5 on the tiny MoE against the libllama oracle).
//
// Segmentation (EC-3): per layer we build+compute a ROUTER segment (attn +
// residual + ffn_norm + router + argsort top-k), read the top-k expert ids back
// to the host with ggml_backend_tensor_get, make EXACTLY those experts resident
// via a bounded SlotPool, then build+compute the EXPERT segment (mul_mat_id over
// the resident experts + weight-and-sum + residual). Because the router output
// is known on the host before the expert matmul is built, resident expert RAM
// is bounded to top-k per layer. The resident full-weight tensors are the
// backing store the pool loads from (the .strata source is EC-4).
//
// This is the ONLY engine TU that includes ggml/llama headers; everything above
// the sf::Model seam sees only model/engine/engine.h.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "model/engine/engine.h"

#include "common/log.h"
#include "kv/kv_store.h"
#include "predict/predictor.h"
#include "tws/strata_file.h"
#include "tws/weight_store.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace sf {
namespace engine {
namespace {

// Fetch a weight tensor by GGUF name from the resident weight context, or null.
ggml_tensor *get_weight(ggml_context *wctx, const std::string &name) {
    ggml_tensor *t = ggml_get_tensor(wctx, name.c_str());
    if (t == nullptr) {
        log_error("engine: tensor not found: " + name);
    }
    return t;
}

} // namespace

// Private state: the resident weight context (ggml owns + fills the tensors via
// gguf no_alloc=false, matching the spike's reliable path), the parsed hparams,
// and the engine-owned KV cache (section 3.4). The weight context is kept alive
// for the engine's lifetime so the graph leaves (weight tensors) stay backed
// across every forward() call.
//
// KV cache: one persistent, data-owning context (kvctx) holds a K and a V
// tensor per layer, each shaped [head_dim, n_head_kv, n_ctx] with real backing
// (allocated once, filled in-graph via ggml_cpy into a position view). These
// are graph LEAVES across forwards, so the per-call gallocr leaves their data
// untouched (same pattern as the resident weight tensors). KvStore is the
// bookkeeping owner of the current fill position and the n_ctx bound.
struct Engine::Impl {
    gguf_context *gc   = nullptr;
    ggml_context *wctx = nullptr;  // full weight tensors, resident (backing store)
    EngineHParams hp;
    bool is_moe = true;            // EC-7: false for a dense llama FFN

    ggml_context *kvctx = nullptr;              // KV cache tensors, resident
    ggml_backend_buffer_t kvbuf = nullptr;      // backing buffer for kvctx
    std::vector<ggml_tensor *> k_cache;         // per layer [head_dim, n_head_kv, n_ctx]
    std::vector<ggml_tensor *> v_cache;         // per layer [head_dim, n_head_kv, n_ctx]
    std::unique_ptr<KvStore> kv;                // position + capacity bookkeeping

    // EC-3 top-k residency. `estctx`/`estbuf` back the per-layer STAGING
    // stacked expert tensors fed to ggml_mul_mat_id: full ne-shape, but only
    // the routed experts' nb02 slices are populated each token (section 3.3 -
    // the kernel skips cne1==0 experts before dereferencing the slice). The
    // SlotPool holds the bounded resident working copies keyed by {layer,
    // expert}; a slot carries one expert's gate|up|down bytes concatenated. The
    // full wctx expert tensors are the backing store the pool loads from.
    ggml_context *estctx = nullptr;
    ggml_backend_buffer_t estbuf = nullptr;
    // ONE staging buffer reused across all layers (segments run sequentially,
    // so only the current layer's experts are ever live). This bounds staging
    // RAM to a single layer's expert footprint instead of n_layer x it - the
    // key to peak-RSS staying near the budget, not the model size.
    ggml_tensor *stage_gate = nullptr;          // ne = ffn_gate_exps (one layer)
    ggml_tensor *stage_up   = nullptr;          // ne = ffn_up_exps
    ggml_tensor *stage_down = nullptr;          // ne = ffn_down_exps
    std::unique_ptr<SlotPool> pool;
    uint64_t gate_expert_bytes = 0;             // bytes of one expert's gate slice
    uint64_t up_expert_bytes   = 0;
    uint64_t down_expert_bytes = 0;
    EngineCacheStats cstats;

    // EC-6 predictor-driven prefetch (section 3.5). After layer N's exact
    // router read-back we observe() the real experts and ask the predictor for
    // layer N+1's likely experts, then WARM those slots in the pool (make them
    // resident) before layer N+1 runs. This only affects LATENCY: layer N+1's
    // own router read-back is still authoritative, and a mispredicted prefetch
    // is corrected by the synchronous ensure_layer_experts_resident, so output
    // stays byte-identical. prefetch_* track how many prefetched experts were
    // actually used (the measurable EC-6 win).
    std::unique_ptr<ExpertPredictor> predictor;
    uint64_t prefetch_warmed = 0;   // experts warmed by a prefetch
    uint64_t prefetch_used   = 0;   // of those, later confirmed resident-hit
    // Experts warmed (by this token's prefetches) whose use we have not yet
    // scored. Cleared at the start of each forward().
    std::unordered_set<ExpertId, ExpertIdHash> warmed_this_token;

    // One CPU backend + gallocr reused across every segment and token (section
    // 3.5: reuse the allocator across tokens; also avoids re-initializing the
    // CPU backend per segment, which is fragile after libllama churns the
    // global ggml backend state in the oracle tests).
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc  = nullptr;

    // EC-4: when the model was loaded from a .strata single file, `reader` is
    // the byte source for the SlotPool: expert bundles are streamed from the
    // .strata expert region via StrataReader::read_blob instead of copied from
    // resident GGUF weight tensors. nullptr on the plain-GGUF path (experts are
    // served from the resident wctx expert tensors, EC-3). The trunk tensors
    // are resident in wctx on both paths; only the expert byte provenance
    // differs, which keeps decode byte-identical (same bytes, same ops).
    std::unique_ptr<StrataReader> reader;
    ggml_backend_buffer_t wbuf = nullptr;  // backing for wctx on the .strata path
    // .strata path only: expert tensors live here as METADATA ONLY (no_alloc,
    // never backed by a buffer), so the full expert footprint is NOT resident
    // in RAM - the whole point of streaming. The SlotPool streams each routed
    // expert from the .strata file via read_blob; these tensors exist only so
    // the staging-tensor sizing can read their ne-shape and nb[2] stride.
    ggml_context *ectx = nullptr;

    ~Impl() {
        if (galloc != nullptr) ggml_gallocr_free(galloc);
        if (backend != nullptr) ggml_backend_free(backend);
        if (estbuf != nullptr) ggml_backend_buffer_free(estbuf);
        if (estctx != nullptr) ggml_free(estctx);
        if (kvbuf != nullptr) ggml_backend_buffer_free(kvbuf);
        if (kvctx != nullptr) ggml_free(kvctx);
        if (wbuf != nullptr) ggml_backend_buffer_free(wbuf);
        if (wctx != nullptr) ggml_free(wctx);
        if (ectx != nullptr) ggml_free(ectx);  // no buffer: metadata only
        if (gc != nullptr) gguf_free(gc);
    }
};

Engine::Engine() : impl_(new Impl()) {}
Engine::~Engine() = default;

// EC-4 (section 5.1): open a .strata, parse its embedded GGUF metadata into
// im.gc, build the resident weight context im.wctx from that metadata (shapes/
// types/names), allocate it from the CPU buffer type, and fill the TRUNK
// tensors from the .strata trunk block. Expert tensors are created resident too
// (so the staging-tensor sizing and the mul_mat_id addressing are identical to
// the GGUF path) but are NOT filled here: on the .strata path the SlotPool
// streams each routed expert bundle from the retained StrataReader
// (im.reader) via read_blob, so the expert byte provenance is the .strata
// expert region, not these tensors. The trunk block is a verbatim contiguous
// copy of the non-expert tensors in GGUF order, so a tensor's trunk-relative
// offset is the running sum of the preceding non-expert tensor sizes.
bool Engine::load_strata_weights(Impl &im, const std::string &path) {
    std::unique_ptr<StrataReader> reader(new StrataReader());
    if (!reader->open(path)) {
        return false;  // StrataReader logged the specific failure
    }

    // Parse the embedded GGUF metadata blob (header+KV+tensor-info, verbatim).
    // no_alloc=true: metadata only, no tensor data (there is none in the blob).
    gguf_init_params mp{};
    mp.no_alloc = true;
    mp.ctx = nullptr;
    im.gc = gguf_init_from_buffer(reader->metadata(),
                                  static_cast<size_t>(reader->metadata_size()),
                                  mp);
    if (im.gc == nullptr) {
        log_warn("engine: failed to parse embedded GGUF metadata in '" + path +
                 "'");
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(im.gc);

    // Build a no_alloc context holding every tensor's metadata (name/shape/
    // type) mirrored from the embedded GGUF tensor-info. ggml_new_tensor copies
    // the ne-shape; we set the name so get_weight(name) resolves downstream.
    // Two contexts: wctx holds the TRUNK tensors (allocated resident); ectx
    // holds the EXPERT tensors as metadata only (no_alloc, never backed), so
    // the full expert footprint is NOT resident in RAM. This is the bounded-RAM
    // guarantee: a huge MoE allocates only trunk + the SlotPool here, and the
    // experts stream from the .strata file on demand.
    {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() *
                      (static_cast<size_t>(n_tensors) + 8);
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;  // trunk backing comes from one buffer below
        im.wctx = ggml_init(ip);
        ggml_init_params ep = ip;
        im.ectx = ggml_init(ep);
        if (im.wctx == nullptr || im.ectx == nullptr) {
            log_warn("engine: .strata weight context init failed");
            return false;
        }
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(im.gc, i);
        const ggml_type type = gguf_get_tensor_type(im.gc, i);
        // gguf_get_tensor_ne returns a GGML_MAX_DIMS array (ne[d]==1 for d >=
        // n_dims), the same inner-first order ggml uses, so pass it straight to
        // ggml_new_tensor with the full dim count; the trailing 1s are inert.
        const int64_t *src_ne = gguf_get_tensor_ne(im.gc, i);
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (int d = 0; d < GGML_MAX_DIMS; ++d) ne[d] = src_ne[d];
        // Expert tensors -> ectx (metadata only); everything else -> wctx.
        const bool is_expert = parse_expert_tensor_name(name).valid;
        ggml_context *into = is_expert ? im.ectx : im.wctx;
        ggml_tensor *t = ggml_new_tensor(into, type, GGML_MAX_DIMS, ne);
        if (t == nullptr) {
            log_warn("engine: .strata tensor create failed for '" +
                     std::string(name) + "'");
            return false;
        }
        ggml_set_name(t, name);
    }

    // Allocate ONLY the trunk tensors (wctx) resident. ectx (experts) is left
    // unbacked on purpose - experts stream from the .strata file.
    im.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(
        im.wctx, ggml_backend_cpu_buffer_type());
    if (im.wbuf == nullptr) {
        log_warn("engine: .strata weight buffer alloc failed");
        return false;
    }

    // Fill the TRUNK tensors from the trunk block. The trunk block packs the
    // non-expert tensors contiguously in GGUF tensor order (tools/strata-pack),
    // so iterate in that order and track the running trunk-relative offset.
    const uint64_t trunk_base = reader->superblock().trunk_offset;
    uint64_t trunk_rel = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(im.gc, i);
        const uint64_t sz = gguf_get_tensor_size(im.gc, i);
        // Expert tensors live in the expert region, not the trunk; skip them
        // here (they stream through the SlotPool from read_blob).
        if (parse_expert_tensor_name(name).valid) {
            continue;
        }
        ggml_tensor *t = ggml_get_tensor(im.wctx, name);
        if (t == nullptr) {
            log_warn("engine: .strata trunk tensor missing '" +
                     std::string(name) + "'");
            return false;
        }
        if (static_cast<uint64_t>(ggml_nbytes(t)) != sz) {
            log_warn("engine: .strata trunk tensor size mismatch '" +
                     std::string(name) + "'");
            return false;
        }
        const int64_t n = reader->read_at(t->data, sz, trunk_base + trunk_rel);
        if (n != static_cast<int64_t>(sz)) {
            log_warn("engine: .strata trunk read failed for '" +
                     std::string(name) + "'");
            return false;
        }
        trunk_rel += sz;
    }

    im.reader = std::move(reader);
    log_info("engine: loaded .strata trunk (" +
             std::to_string(im.reader->index_count()) + " expert blobs; "
             "experts stream via StrataReader)");
    return true;
}

const EngineHParams &Engine::hparams() const { return impl_->hp; }
const EngineCacheStats &Engine::cache_stats() const { return impl_->cstats; }

uint64_t Engine::resident_weight_bytes() const {
    // The .strata path backs only the trunk tensors in im.wbuf; experts live in
    // the unbacked ectx. The plain-GGUF path lets ggml/llama own the mapping,
    // so there is no single wbuf to measure -> report 0 (not applicable).
    if (impl_->wbuf == nullptr) return 0;
    return static_cast<uint64_t>(ggml_backend_buffer_get_size(impl_->wbuf));
}

uint64_t Engine::streamed_bytes() const {
    // Expert bytes streamed from disk on the .strata path; 0 on the plain-GGUF
    // path where there is no StrataReader (experts are resident).
    return impl_->reader ? impl_->reader->streamed_bytes() : 0;
}

std::unique_ptr<Engine> Engine::load(const std::string &path,
                                     uint32_t expert_slots) {
    std::unique_ptr<Engine> eng(new Engine());
    Impl &im = *eng->impl_;

    // EC-4: a .strata single file is our own split trunk/expert format. We read
    // its embedded GGUF metadata + trunk tensors directly here (NO
    // llama_model_load_from_file is ever handed a .strata), and stream experts
    // from the .strata expert region via the StrataReader (section 5.1). A
    // plain GGUF stays on the resident gguf_init_from_file path (section 5.2).
    // Both populate im.gc (metadata) and im.wctx (resident weight tensors) so
    // everything downstream (hparams, KV sizing, staging, forward) is shared.
    if (is_strata_file(path)) {
        if (!load_strata_weights(im, path)) {
            log_warn("engine: .strata load failed for '" + path + "'");
            return nullptr;
        }
    } else {
        // Read trunk + expert tensors resident. no_alloc=false lets ggml
        // allocate and fill every tensor so we can fetch each weight by its
        // GGUF name. On this path experts are served from these resident
        // tensors (EC-3's backing store).
        gguf_init_params gp{};
        gp.no_alloc = false;
        gp.ctx = &im.wctx;
        im.gc = gguf_init_from_file(path.c_str(), gp);
        if (im.gc == nullptr || im.wctx == nullptr) {
            log_warn("engine: gguf open failed for '" + path + "'");
            return nullptr;
        }
    }

    // Confirm this is the one architecture EC-1 handles. Anything else stays on
    // the libllama path (the arch-descriptor table is EC-7).
    {
        int64_t id = gguf_find_key(im.gc, "general.architecture");
        const char *arch = id >= 0 ? gguf_get_val_str(im.gc, id) : nullptr;
        if (arch == nullptr || std::string(arch) != "llama") {
            log_warn("engine: unsupported arch (EC-1 handles llama only); "
                     "staying on libllama path");
            return nullptr;
        }
    }

    // Derive hparams from the GGUF metadata keys, then validate against tensor
    // shapes (ne-order) so a mismatch fails loudly instead of silently.
    auto key_u32 = [&](const std::string &k, int fallback) -> int {
        int64_t id = gguf_find_key(im.gc, k.c_str());
        if (id < 0) return fallback;
        return static_cast<int>(gguf_get_val_u32(im.gc, id));
    };
    auto key_f32 = [&](const std::string &k, float fallback) -> float {
        int64_t id = gguf_find_key(im.gc, k.c_str());
        if (id < 0) return fallback;
        return gguf_get_val_f32(im.gc, id);
    };

    EngineHParams &hp = im.hp;
    hp.n_layer       = key_u32("llama.block_count", 0);
    hp.n_embd        = key_u32("llama.embedding_length", 0);
    hp.n_head        = key_u32("llama.attention.head_count", 0);
    hp.n_head_kv     = key_u32("llama.attention.head_count_kv", hp.n_head);
    hp.n_expert      = key_u32("llama.expert_count", 0);
    hp.n_expert_used = key_u32("llama.expert_used_count", 0);
    hp.n_ctx_orig    = key_u32("llama.context_length", 0);
    hp.rms_eps       = key_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    hp.freq_base     = key_f32("llama.rope.freq_base", 10000.0f);
    hp.freq_scale    = 1.0f;

    ggml_tensor *tok_embd = get_weight(im.wctx, "token_embd.weight");
    ggml_tensor *out_w    = get_weight(im.wctx, "output.weight");
    if (tok_embd == nullptr || out_w == nullptr) {
        return nullptr;
    }

    // EC-7: a dense llama model (expert_count 0/absent) has a plain FFN
    // (ffn_gate/up/down.weight); a MoE model has a router + stacked experts
    // (ffn_gate_exps.weight). Detect which by the layer-0 FFN tensors. Probe
    // with ggml_get_tensor directly (not get_weight) so an expected absence is
    // not logged as an error.
    // Expert tensors live in ectx on the .strata path (metadata only), in wctx
    // on the plain-GGUF path. Look in both.
    auto expert_meta = [&](const char *n) -> ggml_tensor * {
        ggml_tensor *t = ggml_get_tensor(im.wctx, n);
        if (t == nullptr && im.ectx != nullptr) t = ggml_get_tensor(im.ectx, n);
        return t;
    };
    ggml_tensor *ge0  = expert_meta("blk.0.ffn_gate_exps.weight");
    ggml_tensor *dg0  = ggml_get_tensor(im.wctx, "blk.0.ffn_gate.weight");
    im.is_moe = (ge0 != nullptr && hp.n_expert > 0);

    hp.n_vocab = static_cast<int>(tok_embd->ne[1]);
    if (hp.n_embd == 0) hp.n_embd = static_cast<int>(tok_embd->ne[0]);

    if (hp.n_head <= 0 || hp.n_layer <= 0 || hp.n_embd <= 0) {
        log_warn("engine: incomplete hparams; staying on libllama path");
        return nullptr;
    }
    hp.head_dim = hp.n_embd / hp.n_head;

    if (im.is_moe) {
        if (hp.n_expert_used <= 0) {
            log_warn("engine: MoE with no expert_used_count; staying on libllama path");
            return nullptr;
        }
        hp.n_ff = static_cast<int>(ge0->ne[1]);  // gate_exps ne = [n_embd, n_ff, n_expert]
        // Validate the expert-tensor ne-order (design 6.2) so a layout surprise
        // is caught here, not as a wrong logit later.
        if (ge0->ne[0] != hp.n_embd || ge0->ne[2] != hp.n_expert) {
            log_warn("engine: ffn_gate_exps shape mismatch; staying on libllama path");
            return nullptr;
        }
    } else {
        // Dense FFN: ffn_gate/up ne = [n_embd, n_ff], ffn_down ne = [n_ff, n_embd].
        if (dg0 == nullptr) {
            log_warn("engine: no dense FFN and no experts; staying on libllama path");
            return nullptr;
        }
        hp.n_ff = static_cast<int>(dg0->ne[1]);
        if (dg0->ne[0] != hp.n_embd) {
            log_warn("engine: ffn_gate shape mismatch; staying on libllama path");
            return nullptr;
        }
    }

    // Bound the KV cache by the trained context length (n_ctx). The fixture
    // reports 64; fall back to a safe minimum if the key was absent.
    const int n_ctx = hp.n_ctx_orig > 0 ? hp.n_ctx_orig : 64;

    // Allocate the engine-owned KV cache (section 3.4): a K and a V tensor per
    // layer, [head_dim, n_head_kv, n_ctx], in one data-owning context backed by
    // the CPU buffer type. These persist for the engine's lifetime and are the
    // store that each forward() appends to and attends over.
    {
        ggml_init_params kp{};
        kp.mem_size   = ggml_tensor_overhead() *
                        (static_cast<size_t>(hp.n_layer) * 2 + 8);
        kp.mem_buffer = nullptr;
        kp.no_alloc   = true;  // tensors get backing from a single buffer below
        im.kvctx = ggml_init(kp);
        if (im.kvctx == nullptr) {
            log_warn("engine: KV context init failed; staying on libllama path");
            return nullptr;
        }
        im.k_cache.resize(static_cast<size_t>(hp.n_layer));
        im.v_cache.resize(static_cast<size_t>(hp.n_layer));
        for (int il = 0; il < hp.n_layer; ++il) {
            ggml_tensor *kc = ggml_new_tensor_3d(im.kvctx, GGML_TYPE_F32,
                                                 hp.head_dim, hp.n_head_kv, n_ctx);
            ggml_tensor *vc = ggml_new_tensor_3d(im.kvctx, GGML_TYPE_F32,
                                                 hp.head_dim, hp.n_head_kv, n_ctx);
            ggml_set_name(kc, ("kv_k." + std::to_string(il)).c_str());
            ggml_set_name(vc, ("kv_v." + std::to_string(il)).c_str());
            im.k_cache[static_cast<size_t>(il)] = kc;
            im.v_cache[static_cast<size_t>(il)] = vc;
        }
        im.kvbuf = ggml_backend_alloc_ctx_tensors_from_buft(
            im.kvctx, ggml_backend_cpu_buffer_type());
        if (im.kvbuf == nullptr) {
            log_warn("engine: KV buffer alloc failed; staying on libllama path");
            return nullptr;
        }
    }

    // bytes_per_token across all layers: K and V, head_dim * n_head_kv F32 each.
    const uint64_t bytes_per_token =
        static_cast<uint64_t>(hp.n_layer) * 2 *
        static_cast<uint64_t>(hp.head_dim) *
        static_cast<uint64_t>(hp.n_head_kv) * sizeof(float);
    im.kv.reset(new KvStore(static_cast<uint32_t>(n_ctx), bytes_per_token));

    // EC-3: per-layer staging stacked expert tensors + the bounded SlotPool.
    // The staging tensors have the SAME ne-shape as the full expert tensors so
    // ggml_mul_mat_id addresses expert cur_a at the same cur_a*nb02 offset; we
    // populate only the routed slices each token. One slot of the pool holds
    // one expert's gate|up|down bytes, so a pool sized below n_expert bounds
    // resident expert RAM to that many bundles. Dense models (EC-7) have no
    // experts, so this whole block is MoE-only.
    if (im.is_moe) {
        ggml_tensor *g0 = expert_meta("blk.0.ffn_gate_exps.weight");
        ggml_tensor *u0 = expert_meta("blk.0.ffn_up_exps.weight");
        ggml_tensor *d0 = expert_meta("blk.0.ffn_down_exps.weight");
        if (g0 == nullptr || u0 == nullptr || d0 == nullptr) {
            log_warn("engine: missing expert tensor; staying on libllama path");
            return nullptr;
        }
        // Per-expert slice bytes = whole-tensor bytes / n_expert (nb02 stride).
        im.gate_expert_bytes = g0->nb[2];
        im.up_expert_bytes   = u0->nb[2];
        im.down_expert_bytes = d0->nb[2];
        const uint64_t slot_bytes =
            im.gate_expert_bytes + im.up_expert_bytes + im.down_expert_bytes;

        ggml_init_params ep{};
        ep.mem_size   = ggml_tensor_overhead() *
                        (static_cast<size_t>(hp.n_layer) * 3 + 8);
        ep.mem_buffer = nullptr;
        ep.no_alloc   = true;  // backing from one buffer below
        im.estctx = ggml_init(ep);
        if (im.estctx == nullptr) {
            log_warn("engine: staging context init failed; libllama path");
            return nullptr;
        }
        // One layer's worth of staging, reused every layer. Use the SOURCE
        // expert tensor's real ggml type (F32 or a quant type like
        // Q4_K/Q6_K/Q8_0): ggml_mul_mat_id handles quantized weights natively.
        im.stage_gate = ggml_new_tensor_3d(im.estctx, g0->type,
                                           g0->ne[0], g0->ne[1], g0->ne[2]);
        im.stage_up   = ggml_new_tensor_3d(im.estctx, u0->type,
                                           u0->ne[0], u0->ne[1], u0->ne[2]);
        im.stage_down = ggml_new_tensor_3d(im.estctx, d0->type,
                                           d0->ne[0], d0->ne[1], d0->ne[2]);
        ggml_set_name(im.stage_gate, "stg_gate");
        ggml_set_name(im.stage_up,   "stg_up");
        ggml_set_name(im.stage_down, "stg_down");
        im.estbuf = ggml_backend_alloc_ctx_tensors_from_buft(
            im.estctx, ggml_backend_cpu_buffer_type());
        if (im.estbuf == nullptr) {
            log_warn("engine: staging buffer alloc failed; libllama path");
            return nullptr;
        }
        // Zero the staging buffer so non-routed expert slices are well-defined
        // (the kernel skips cne1==0 experts, but defined zeros avoid any inf/nan
        // in uninitialized memory leaking into an assert on debug builds).
        std::memset(im.stage_gate->data, 0, ggml_nbytes(im.stage_gate));
        std::memset(im.stage_up->data,   0, ggml_nbytes(im.stage_up));
        std::memset(im.stage_down->data, 0, ggml_nbytes(im.stage_down));

        // Pool size: auto (0) holds the whole working set (n_layer*n_expert, no
        // eviction); a positive request is clamped up to n_expert_used so a
        // single layer's top-k always fits. A test passes a value below
        // n_expert to force eviction/reload.
        const uint32_t full_slots =
            static_cast<uint32_t>(hp.n_layer) * static_cast<uint32_t>(hp.n_expert);
        uint32_t n_slots = expert_slots == 0 ? full_slots : expert_slots;
        if (n_slots < static_cast<uint32_t>(hp.n_expert_used)) {
            n_slots = static_cast<uint32_t>(hp.n_expert_used);
        }
        im.pool.reset(new SlotPool(n_slots, slot_bytes));
        im.cstats.n_slots    = n_slots;
        im.cstats.slot_bytes = slot_bytes;
        im.cstats.full_bytes = static_cast<uint64_t>(full_slots) * slot_bytes;
    }

    // EC-6: statistical expert predictor for prefetch warming (section 3.5).
    // MoE-only (a dense model routes nothing).
    if (im.is_moe) {
        im.predictor.reset(new ExpertPredictor(static_cast<uint32_t>(hp.n_layer),
                                               static_cast<uint32_t>(hp.n_expert)));
    }

    // One CPU backend + gallocr for the engine's lifetime (1 thread for the
    // greedy-determinism contract). Reused across every segment and token.
    im.backend = ggml_backend_cpu_init();
    if (im.backend == nullptr) {
        log_warn("engine: cpu backend init failed; staying on libllama path");
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(im.backend, 1);
    im.galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (im.galloc == nullptr) {
        log_warn("engine: gallocr init failed; staying on libllama path");
        return nullptr;
    }

    if (im.is_moe) {
        log_info("engine: loaded llama-arch MoE (" + std::to_string(hp.n_layer) +
                 " layers, " + std::to_string(hp.n_expert) + "x" +
                 std::to_string(hp.n_expert_used) + " experts, n_embd=" +
                 std::to_string(hp.n_embd) + ", n_vocab=" +
                 std::to_string(hp.n_vocab) + ", n_ctx=" + std::to_string(n_ctx) +
                 ")");
    } else {
        log_info("engine: loaded llama-arch dense (" + std::to_string(hp.n_layer) +
                 " layers, n_embd=" + std::to_string(hp.n_embd) + ", n_ff=" +
                 std::to_string(hp.n_ff) + ", n_vocab=" +
                 std::to_string(hp.n_vocab) + ", n_ctx=" + std::to_string(n_ctx) +
                 ")");
    }
    return eng;
}

void Engine::reset_kv() {
    if (impl_->kv != nullptr) impl_->kv->reset();
}

namespace {

// Allocate and run one segment graph `gf` (rooted at `roots`) on a fresh CPU
// backend, calling `set_inputs()` after allocation to fill the segment's input
// tensors. 1 thread for the greedy-determinism contract. Returns true on a
// successful compute. The caller owns `gctx` and reads outputs afterwards.
template <typename SetInputs>
bool run_segment(ggml_backend_t backend, ggml_gallocr_t galloc,
                 ggml_context *gctx, const std::vector<ggml_tensor *> &roots,
                 SetInputs &&set_inputs) {
    ggml_cgraph *gf = ggml_new_graph(gctx);
    for (ggml_tensor *r : roots) {
        ggml_build_forward_expand(gf, r);
    }

    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        log_error("engine: gallocr alloc failed");
        return false;
    }

    set_inputs();  // fill inputs AFTER allocation

    const bool compute_ok =
        ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (!compute_ok) log_error("engine: graph compute failed");
    return compute_ok;
}

// EC-7: build one DENSE llama layer as a single segment (no router/experts, so
// no mid-layer host read-back): attn + residual + ffn_norm + plain gate/up/down
// FFN (swiglu) + residual. The attention block is intentionally identical to
// build_router_segment's (duplicated rather than shared, to keep the proven MoE
// path untouched). Returns the layer output [n_embd, n_tokens] (set as output);
// KV stores are appended to `graph_stores` so run_segment roots them.
ggml_tensor *build_dense_layer_segment(ggml_context *gctx, ggml_tensor *inpL,
                                       ggml_tensor *inp_pos, int il,
                                       ggml_context *wctx, const EngineHParams &hp,
                                       ggml_tensor *k_cache, ggml_tensor *v_cache,
                                       ggml_tensor *kq_mask, int64_t pos,
                                       int64_t n_kv,
                                       std::vector<ggml_tensor *> &graph_stores) {
    const std::string p = "blk." + std::to_string(il) + ".";
    ggml_tensor *attn_norm_w = ggml_get_tensor(wctx, (p + "attn_norm.weight").c_str());
    ggml_tensor *wq          = ggml_get_tensor(wctx, (p + "attn_q.weight").c_str());
    ggml_tensor *wk          = ggml_get_tensor(wctx, (p + "attn_k.weight").c_str());
    ggml_tensor *wv          = ggml_get_tensor(wctx, (p + "attn_v.weight").c_str());
    ggml_tensor *wo          = ggml_get_tensor(wctx, (p + "attn_output.weight").c_str());
    ggml_tensor *ffn_norm_w  = ggml_get_tensor(wctx, (p + "ffn_norm.weight").c_str());
    ggml_tensor *w_gate      = ggml_get_tensor(wctx, (p + "ffn_gate.weight").c_str());
    ggml_tensor *w_up        = ggml_get_tensor(wctx, (p + "ffn_up.weight").c_str());
    ggml_tensor *w_down      = ggml_get_tensor(wctx, (p + "ffn_down.weight").c_str());

    const int64_t n_tokens = inpL->ne[1];
    ggml_tensor *inpSA = inpL;

    ggml_tensor *cur = ggml_rms_norm(gctx, inpL, hp.rms_eps);
    cur = ggml_mul(gctx, cur, attn_norm_w);

    ggml_tensor *Qcur = ggml_mul_mat(gctx, wq, cur);
    ggml_tensor *Kcur = ggml_mul_mat(gctx, wk, cur);
    ggml_tensor *Vcur = ggml_mul_mat(gctx, wv, cur);
    Qcur = ggml_reshape_3d(gctx, Qcur, hp.head_dim, hp.n_head,    n_tokens);
    Kcur = ggml_reshape_3d(gctx, Kcur, hp.head_dim, hp.n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(gctx, Vcur, hp.head_dim, hp.n_head_kv, n_tokens);
    Qcur = ggml_rope_ext(gctx, Qcur, inp_pos, nullptr, hp.head_dim,
                         GGML_ROPE_TYPE_NORMAL, hp.n_ctx_orig, hp.freq_base,
                         hp.freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(gctx, Kcur, inp_pos, nullptr, hp.head_dim,
                         GGML_ROPE_TYPE_NORMAL, hp.n_ctx_orig, hp.freq_base,
                         hp.freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

    const size_t kv_col_nb = k_cache->nb[2];
    ggml_tensor *k_dst = ggml_view_3d(gctx, k_cache, hp.head_dim, hp.n_head_kv,
                                      n_tokens, k_cache->nb[1], k_cache->nb[2],
                                      static_cast<size_t>(pos) * kv_col_nb);
    ggml_tensor *v_dst = ggml_view_3d(gctx, v_cache, hp.head_dim, hp.n_head_kv,
                                      n_tokens, v_cache->nb[1], v_cache->nb[2],
                                      static_cast<size_t>(pos) * v_cache->nb[2]);
    graph_stores.push_back(ggml_cpy(gctx, Kcur, k_dst));
    graph_stores.push_back(ggml_cpy(gctx, Vcur, v_dst));

    ggml_tensor *k_hist = ggml_view_3d(gctx, k_cache, hp.head_dim, hp.n_head_kv,
                                       n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    ggml_tensor *v_hist = ggml_view_3d(gctx, v_cache, hp.head_dim, hp.n_head_kv,
                                       n_kv, v_cache->nb[1], v_cache->nb[2], 0);
    ggml_tensor *q = ggml_permute(gctx, Qcur, 0, 2, 1, 3);
    ggml_tensor *k = ggml_cont(gctx, ggml_permute(gctx, k_hist, 0, 2, 1, 3));
    ggml_tensor *v = ggml_cont(gctx, ggml_permute(gctx, v_hist, 0, 2, 1, 3));
    ggml_tensor *kq = ggml_mul_mat(gctx, k, q);
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(hp.head_dim));
    kq = ggml_soft_max_ext(gctx, kq, kq_mask, kq_scale, 0.0f);
    ggml_tensor *vt = ggml_cont(gctx, ggml_transpose(gctx, v));
    ggml_tensor *kqv = ggml_mul_mat(gctx, vt, kq);
    kqv = ggml_permute(gctx, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(gctx, kqv, hp.head_dim * hp.n_head, n_tokens);
    cur = ggml_mul_mat(gctx, wo, cur);
    ggml_tensor *ffn_inp = ggml_add(gctx, cur, inpSA);

    cur = ggml_rms_norm(gctx, ffn_inp, hp.rms_eps);
    cur = ggml_mul(gctx, cur, ffn_norm_w);

    // Dense FFN: swiglu(gate(cur), up(cur)) -> down, then residual.
    ggml_tensor *gate = ggml_mul_mat(gctx, w_gate, cur);
    ggml_tensor *up   = ggml_mul_mat(gctx, w_up, cur);
    ggml_tensor *act  = ggml_swiglu_split(gctx, gate, up);
    ggml_tensor *down = ggml_mul_mat(gctx, w_down, act);
    ggml_tensor *layer_out = ggml_cont(gctx, ggml_add(gctx, down, ffn_inp));
    ggml_set_output(layer_out);
    return layer_out;
}

// Build the ROUTER segment for one layer (section 3.2 step 1): attn + residual
// + ffn_norm + router + argsort top-k. `inpL` is the segment input [n_embd, 1]
// (an input tensor filled from the host). Outputs needed by the host and the
// expert segment are returned via the out-params and marked ggml_set_output:
//   `cur`      - ffn-normed input to the experts [n_embd, 1]
//   `ffn_inp`  - the pre-FFN residual branch   [n_embd, 1]
//   `weights`  - normalized top-k gate weights [n_expert_used]
//   `selected` - the top-k expert ids          [n_expert_used]
// KV stores for this layer are appended to `graph_stores` so run_segment roots
// them (attention reads the cache tensor, not the cpy node).
void build_router_segment(ggml_context *gctx, ggml_tensor *inpL,
                          ggml_tensor *inp_pos, int il, ggml_context *wctx,
                          const EngineHParams &hp, ggml_tensor *k_cache,
                          ggml_tensor *v_cache, ggml_tensor *kq_mask,
                          int64_t pos, int64_t n_kv,
                          std::vector<ggml_tensor *> &graph_stores,
                          ggml_tensor **out_cur, ggml_tensor **out_ffn_inp,
                          ggml_tensor **out_weights, ggml_tensor **out_selected) {
    const std::string p = "blk." + std::to_string(il) + ".";

    ggml_tensor *attn_norm_w = ggml_get_tensor(wctx, (p + "attn_norm.weight").c_str());
    ggml_tensor *wq          = ggml_get_tensor(wctx, (p + "attn_q.weight").c_str());
    ggml_tensor *wk          = ggml_get_tensor(wctx, (p + "attn_k.weight").c_str());
    ggml_tensor *wv          = ggml_get_tensor(wctx, (p + "attn_v.weight").c_str());
    ggml_tensor *wo          = ggml_get_tensor(wctx, (p + "attn_output.weight").c_str());
    ggml_tensor *ffn_norm_w  = ggml_get_tensor(wctx, (p + "ffn_norm.weight").c_str());
    ggml_tensor *gate_inp    = ggml_get_tensor(wctx, (p + "ffn_gate_inp.weight").c_str());

    const int64_t n_tokens = inpL->ne[1];
    ggml_tensor *inpSA = inpL;

    // attn_norm: RMSNorm then elementwise mul by weight.
    ggml_tensor *cur = ggml_rms_norm(gctx, inpL, hp.rms_eps);
    cur = ggml_mul(gctx, cur, attn_norm_w);

    // Q,K,V projections.
    ggml_tensor *Qcur = ggml_mul_mat(gctx, wq, cur);
    ggml_tensor *Kcur = ggml_mul_mat(gctx, wk, cur);
    ggml_tensor *Vcur = ggml_mul_mat(gctx, wv, cur);

    // Reshape to heads: [head_dim, n_head(_kv), n_tokens].
    Qcur = ggml_reshape_3d(gctx, Qcur, hp.head_dim, hp.n_head,    n_tokens);
    Kcur = ggml_reshape_3d(gctx, Kcur, hp.head_dim, hp.n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(gctx, Vcur, hp.head_dim, hp.n_head_kv, n_tokens);

    // NORMAL rope on Q and K (mode 0), n_dims = head_dim.
    Qcur = ggml_rope_ext(gctx, Qcur, inp_pos, nullptr, hp.head_dim,
                         GGML_ROPE_TYPE_NORMAL, hp.n_ctx_orig, hp.freq_base,
                         hp.freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(gctx, Kcur, inp_pos, nullptr, hp.head_dim,
                         GGML_ROPE_TYPE_NORMAL, hp.n_ctx_orig, hp.freq_base,
                         hp.freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

    // Append this token's K and V to the per-layer KV cache at column `pos`.
    const size_t kv_col_nb = k_cache->nb[2];  // bytes per token-column
    ggml_tensor *k_dst = ggml_view_3d(gctx, k_cache, hp.head_dim, hp.n_head_kv,
                                      n_tokens, k_cache->nb[1], k_cache->nb[2],
                                      static_cast<size_t>(pos) * kv_col_nb);
    ggml_tensor *v_dst = ggml_view_3d(gctx, v_cache, hp.head_dim, hp.n_head_kv,
                                      n_tokens, v_cache->nb[1], v_cache->nb[2],
                                      static_cast<size_t>(pos) * v_cache->nb[2]);
    graph_stores.push_back(ggml_cpy(gctx, Kcur, k_dst));
    graph_stores.push_back(ggml_cpy(gctx, Vcur, v_dst));

    // Attention over KV history (explicit non-flash branch of build_attn_mha).
    ggml_tensor *k_hist = ggml_view_3d(gctx, k_cache, hp.head_dim, hp.n_head_kv,
                                       n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    ggml_tensor *v_hist = ggml_view_3d(gctx, v_cache, hp.head_dim, hp.n_head_kv,
                                       n_kv, v_cache->nb[1], v_cache->nb[2], 0);

    ggml_tensor *q = ggml_permute(gctx, Qcur, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head]
    ggml_tensor *k = ggml_cont(gctx, ggml_permute(gctx, k_hist, 0, 2, 1, 3));  // [head_dim, n_kv, n_head_kv]
    ggml_tensor *v = ggml_cont(gctx, ggml_permute(gctx, v_hist, 0, 2, 1, 3));  // [head_dim, n_kv, n_head_kv]

    ggml_tensor *kq = ggml_mul_mat(gctx, k, q);             // [n_kv, n_tokens, n_head]
    // The reference forces GGML_PREC_F32 accumulation on kq; match it.
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(hp.head_dim));
    kq = ggml_soft_max_ext(gctx, kq, kq_mask, kq_scale, 0.0f);

    ggml_tensor *vt = ggml_cont(gctx, ggml_transpose(gctx, v));  // [n_kv, head_dim, n_head_kv]
    ggml_tensor *kqv = ggml_mul_mat(gctx, vt, kq);          // [head_dim, n_tokens, n_head]
    kqv = ggml_permute(gctx, kqv, 0, 2, 1, 3);              // [head_dim, n_head, n_tokens]
    cur = ggml_cont_2d(gctx, kqv, hp.head_dim * hp.n_head, n_tokens);

    // Output projection + residual.
    cur = ggml_mul_mat(gctx, wo, cur);
    ggml_tensor *ffn_inp = ggml_add(gctx, cur, inpSA);

    // ffn_norm.
    cur = ggml_rms_norm(gctx, ffn_inp, hp.rms_eps);
    cur = ggml_mul(gctx, cur, ffn_norm_w);

    // Router (build_moe_ffn: SOFTMAX gating, norm_w=true, w_scale=1).
    ggml_tensor *router   = ggml_mul_mat(gctx, gate_inp, cur);   // [n_expert, n_tokens]
    ggml_tensor *probs    = ggml_soft_max(gctx, router);         // [n_expert, n_tokens]
    ggml_tensor *selected = ggml_argsort_top_k(gctx, probs, hp.n_expert_used);

    ggml_tensor *probs3  = ggml_reshape_3d(gctx, probs, 1, hp.n_expert, n_tokens);
    ggml_tensor *weights = ggml_get_rows(gctx, probs3, selected);  // [1, n_used, n_tokens]

    // Normalize weights by their clamped sum (clamp constant from build_moe_ffn).
    weights = ggml_reshape_2d(gctx, weights, hp.n_expert_used, n_tokens);
    ggml_tensor *wsum = ggml_sum_rows(gctx, weights);           // [1, n_tokens]
    wsum = ggml_clamp(gctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(gctx, weights, wsum);                    // [n_expert_used, n_tokens]

    // argsort_top_k returns a strided VIEW into the full argsort row; make the
    // top-k ids contiguous so the host read-back gets exactly the k ids.
    ggml_tensor *sel_c = ggml_cont(gctx, selected);

    // Materialize the host-read-back tensors as distinct contiguous copies so
    // the graph allocator cannot reuse their storage for a later op after the
    // producing op runs (cur is consumed only by the router matmul, so without
    // this the allocator may recycle its block and the read-back sees stale
    // zeros). These copies are explicit graph leaves marked as outputs.
    ggml_tensor *cur_o = ggml_cont(gctx, cur);
    ggml_tensor *ffn_o = ggml_cont(gctx, ffn_inp);
    ggml_tensor *w_o   = ggml_cont(gctx, weights);

    ggml_set_output(cur_o);
    ggml_set_output(ffn_o);
    ggml_set_output(w_o);
    ggml_set_output(sel_c);
    *out_cur      = cur_o;
    *out_ffn_inp  = ffn_o;
    *out_weights  = w_o;
    *out_selected = sel_c;
}

// Build the EXPERT segment for one layer (section 3.2 step 3) over the now-
// resident staging stacked tensors. Inputs are filled from the host router
// read-back: `cur_in` [n_embd,1], `ids_in` [n_expert_used] (I32), `w_in`
// [n_expert_used]. `ffn_inp_in` [n_embd,1] is the residual branch. The staging
// tensors (`stg_*`) carry valid bytes only in the routed experts' nb02 slices;
// ggml_mul_mat_id skips the cne1==0 experts so the non-routed slices are never
// read (section 3.3). Returns the layer output [n_embd, 1] (set as output).
ggml_tensor *build_expert_segment(ggml_context *gctx, const EngineHParams &hp,
                                  ggml_tensor *cur_in, ggml_tensor *ffn_inp_in,
                                  ggml_tensor *ids_in, ggml_tensor *w_in,
                                  ggml_tensor *stg_gate, ggml_tensor *stg_up,
                                  ggml_tensor *stg_down) {
    const int64_t n_tokens = 1;
    ggml_tensor *ids = ggml_reshape_2d(gctx, ids_in, hp.n_expert_used, n_tokens);
    ggml_tensor *weights =
        ggml_reshape_3d(gctx, w_in, 1, hp.n_expert_used, n_tokens);

    ggml_tensor *cur3 = ggml_reshape_3d(gctx, cur_in, hp.n_embd, 1, n_tokens);
    ggml_tensor *up   = ggml_mul_mat_id(gctx, stg_up,   cur3, ids);
    ggml_tensor *gate = ggml_mul_mat_id(gctx, stg_gate, cur3, ids);
    ggml_tensor *act  = ggml_swiglu_split(gctx, gate, up);
    ggml_tensor *experts = ggml_mul_mat_id(gctx, stg_down, act, ids);

    // Weight each expert output and sum over the n_expert_used dim.
    experts = ggml_mul(gctx, experts, weights);                // [n_embd, n_used, n_tokens]
    ggml_tensor *moe_out = ggml_view_2d(gctx, experts, hp.n_embd, n_tokens,
                                        experts->nb[2], 0);
    for (int i = 1; i < hp.n_expert_used; ++i) {
        ggml_tensor *ei = ggml_view_2d(gctx, experts, hp.n_embd, n_tokens,
                                       experts->nb[2],
                                       static_cast<size_t>(i) * experts->nb[1]);
        moe_out = ggml_add(gctx, moe_out, ei);
    }

    // Materialize a distinct contiguous copy for the host read-back so the
    // graph allocator does not recycle the residual-add block (same reasoning
    // as the router segment outputs).
    ggml_tensor *layer_out = ggml_cont(gctx, ggml_add(gctx, moe_out, ffn_inp_in));
    ggml_set_output(layer_out);
    return layer_out;
}

// New no_alloc=true build context sized for one segment's node count.
ggml_context *new_segment_ctx() {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    return ggml_init(ip);
}

} // namespace

// Make the routed top-k experts of layer `il` resident in the staging stacked
// tensors (section 3.2 step 2). For each distinct selected expert id, acquire
// its {layer,expert} bundle from the bounded SlotPool (loading gate|up|down
// from the full weight tensors on a miss), then copy the slot bytes into that
// expert's nb02 slice of the staging tensors. Only these slices hold valid
// bytes; the kernel never reads the others. Returns false on a lookup failure.
bool Engine::ensure_layer_experts_resident(int il, const int32_t *ids,
                                           int n_ids) {
    Impl &im = *impl_;
    const std::string p = "blk." + std::to_string(il) + ".";
    // Plain-GGUF path: experts are served from the resident wctx tensors. On
    // the .strata path (im.reader != nullptr) they stream from the .strata
    // expert region via StrataReader::read_blob, so these resident tensors are
    // not read (and were never filled). Only look them up on the GGUF path.
    ggml_tensor *src_gate = nullptr;
    ggml_tensor *src_up   = nullptr;
    ggml_tensor *src_down = nullptr;
    if (im.reader == nullptr) {
        src_gate = ggml_get_tensor(im.wctx, (p + "ffn_gate_exps.weight").c_str());
        src_up   = ggml_get_tensor(im.wctx, (p + "ffn_up_exps.weight").c_str());
        src_down = ggml_get_tensor(im.wctx, (p + "ffn_down_exps.weight").c_str());
        if (src_gate == nullptr || src_up == nullptr || src_down == nullptr) {
            log_error("engine: missing expert source tensor at layer " +
                      std::to_string(il));
            return false;
        }
    }
    ggml_tensor *dst_gate = im.stage_gate;
    ggml_tensor *dst_up   = im.stage_up;
    ggml_tensor *dst_down = im.stage_down;

    const uint64_t gb = im.gate_expert_bytes;
    const uint64_t ub = im.up_expert_bytes;
    const uint64_t db = im.down_expert_bytes;

    for (int i = 0; i < n_ids; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= hp_().n_expert) {
            log_error("engine: routed expert id out of range");
            return false;
        }
        // Skip duplicates within this token's top-k (acquire is idempotent, but
        // this avoids a redundant copy).
        bool dup = false;
        for (int j = 0; j < i; ++j) {
            if (ids[j] == e) { dup = true; break; }
        }
        if (dup) continue;

        ExpertId id{static_cast<uint32_t>(il), static_cast<uint32_t>(e)};
        // Load this expert bundle's gate|up|down bytes into its slot on a miss.
        // Byte source (EC-4): the .strata expert region via StrataReader when a
        // .strata was loaded, else the per-expert nb02 slices of the resident
        // GGUF weight tensors. Either source yields the SAME verbatim bytes
        // (the packer copies bytes), so the slot contents - and the decode -
        // are byte-identical regardless of provenance.
        bool load_ok = true;
        const uint32_t slot = im.pool->acquire(id, [&](void *dst) {
            uint8_t *d = static_cast<uint8_t *>(dst);
            if (im.reader != nullptr) {
                const int64_t ng = im.reader->read_blob(
                    static_cast<uint32_t>(il), static_cast<uint32_t>(e),
                    ExpertTensorKind::kGate, d);
                const int64_t nu = im.reader->read_blob(
                    static_cast<uint32_t>(il), static_cast<uint32_t>(e),
                    ExpertTensorKind::kUp, d + gb);
                const int64_t nd = im.reader->read_blob(
                    static_cast<uint32_t>(il), static_cast<uint32_t>(e),
                    ExpertTensorKind::kDown, d + gb + ub);
                if (ng != static_cast<int64_t>(gb) ||
                    nu != static_cast<int64_t>(ub) ||
                    nd != static_cast<int64_t>(db)) {
                    load_ok = false;
                }
            } else {
                std::memcpy(d,
                            static_cast<const uint8_t *>(src_gate->data) +
                                static_cast<size_t>(e) * gb,
                            static_cast<size_t>(gb));
                std::memcpy(d + gb,
                            static_cast<const uint8_t *>(src_up->data) +
                                static_cast<size_t>(e) * ub,
                            static_cast<size_t>(ub));
                std::memcpy(d + gb + ub,
                            static_cast<const uint8_t *>(src_down->data) +
                                static_cast<size_t>(e) * db,
                            static_cast<size_t>(db));
            }
        });
        if (slot == UINT32_MAX) {
            log_error("engine: slot pool returned no slot");
            return false;
        }
        if (!load_ok) {
            log_error("engine: .strata expert read failed (layer " +
                      std::to_string(il) + ", expert " + std::to_string(e) +
                      ")");
            return false;
        }

        // Copy the resident slot bytes into the staging tensors' nb02 slices for
        // this expert, so ggml_mul_mat_id reads correct bytes at cur_a*nb02.
        const uint8_t *s = static_cast<const uint8_t *>(im.pool->slot_data(slot));
        std::memcpy(static_cast<uint8_t *>(dst_gate->data) +
                        static_cast<size_t>(e) * gb,
                    s, static_cast<size_t>(gb));
        std::memcpy(static_cast<uint8_t *>(dst_up->data) +
                        static_cast<size_t>(e) * ub,
                    s + gb, static_cast<size_t>(ub));
        std::memcpy(static_cast<uint8_t *>(dst_down->data) +
                        static_cast<size_t>(e) * db,
                    s + gb + ub, static_cast<size_t>(db));
    }
    return true;
}

// EC-6: warm one (layer, expert) bundle into the pool (prefetch). Same byte
// source and load as ensure_layer_experts_resident, but no staging copy - it
// only makes the slot resident so a later authoritative acquire hits. Any
// failure is swallowed: correctness never depends on a prefetch.
void Engine::warm_expert(int il, int expert) {
    Impl &im = *impl_;
    if (il < 0 || il >= hp_().n_layer) return;
    if (expert < 0 || expert >= hp_().n_expert) return;

    const std::string p = "blk." + std::to_string(il) + ".";
    ggml_tensor *src_gate = nullptr;
    ggml_tensor *src_up   = nullptr;
    ggml_tensor *src_down = nullptr;
    if (im.reader == nullptr) {
        src_gate = ggml_get_tensor(im.wctx, (p + "ffn_gate_exps.weight").c_str());
        src_up   = ggml_get_tensor(im.wctx, (p + "ffn_up_exps.weight").c_str());
        src_down = ggml_get_tensor(im.wctx, (p + "ffn_down_exps.weight").c_str());
        if (src_gate == nullptr || src_up == nullptr || src_down == nullptr) return;
    }
    const uint64_t gb = im.gate_expert_bytes;
    const uint64_t ub = im.up_expert_bytes;
    const uint64_t db = im.down_expert_bytes;
    const int32_t e = expert;

    ExpertId id{static_cast<uint32_t>(il), static_cast<uint32_t>(e)};
    im.pool->acquire(id, [&](void *dst) {
        uint8_t *d = static_cast<uint8_t *>(dst);
        if (im.reader != nullptr) {
            im.reader->read_blob(static_cast<uint32_t>(il),
                                 static_cast<uint32_t>(e),
                                 ExpertTensorKind::kGate, d);
            im.reader->read_blob(static_cast<uint32_t>(il),
                                 static_cast<uint32_t>(e),
                                 ExpertTensorKind::kUp, d + gb);
            im.reader->read_blob(static_cast<uint32_t>(il),
                                 static_cast<uint32_t>(e),
                                 ExpertTensorKind::kDown, d + gb + ub);
        } else {
            std::memcpy(d,
                        static_cast<const uint8_t *>(src_gate->data) +
                            static_cast<size_t>(e) * gb,
                        static_cast<size_t>(gb));
            std::memcpy(d + gb,
                        static_cast<const uint8_t *>(src_up->data) +
                            static_cast<size_t>(e) * ub,
                        static_cast<size_t>(ub));
            std::memcpy(d + gb + ub,
                        static_cast<const uint8_t *>(src_down->data) +
                            static_cast<size_t>(e) * db,
                        static_cast<size_t>(db));
        }
    });
}

const EngineHParams &Engine::hp_() const { return impl_->hp; }

bool Engine::forward(int32_t token, int32_t pos, std::vector<float> &out) {
    Impl &im = *impl_;
    const EngineHParams &hp = im.hp;

    im.warmed_this_token.clear();  // EC-6: fresh prefetch bookkeeping per token

    ggml_tensor *tok_embd = ggml_get_tensor(im.wctx, "token_embd.weight");
    ggml_tensor *out_norm = ggml_get_tensor(im.wctx, "output_norm.weight");
    ggml_tensor *lm_head  = ggml_get_tensor(im.wctx, "output.weight");
    if (tok_embd == nullptr || out_norm == nullptr || lm_head == nullptr) {
        log_error("engine: missing trunk tensor at forward");
        return false;
    }

    if (pos < 0 || im.kv == nullptr ||
        pos >= static_cast<int32_t>(im.kv->capacity())) {
        log_error("engine: position " + std::to_string(pos) +
                  " out of KV bounds (capacity " +
                  std::to_string(im.kv != nullptr ? im.kv->capacity() : 0) + ")");
        return false;
    }
    const int64_t n_kv = static_cast<int64_t>(pos) + 1;  // attend over 0..pos
    const size_t embd = static_cast<size_t>(hp.n_embd);
    const size_t n_used = static_cast<size_t>(hp.n_expert_used);

    // The layer-boundary hidden state carried on the host between segments.
    // Segment 0's embedding lookup is folded into the first router segment's
    // input tensor, so compute the embedding here into `hidden`.
    std::vector<float> hidden(embd, 0.0f);
    {
        ggml_context *gctx = new_segment_ctx();
        if (gctx == nullptr) { log_error("engine: ggml_init failed"); return false; }
        ggml_tensor *inp_tok = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
        ggml_set_input(inp_tok);
        ggml_tensor *emb = ggml_get_rows(gctx, tok_embd, inp_tok);  // [n_embd, 1]
        ggml_set_output(emb);
        const bool ok = run_segment(im.backend, im.galloc, gctx, {emb}, [&]() {
            int32_t t = token;
            ggml_backend_tensor_set(inp_tok, &t, 0, sizeof(t));
        });
        if (ok) {
            ggml_backend_tensor_get(emb, hidden.data(), 0, embd * sizeof(float));
        }
        ggml_free(gctx);
        if (!ok) return false;
    }

    // Per-layer SEGMENTED execution (section 3.2): router segment -> host read
    // back top-k ids -> make resident -> expert segment.
    for (int il = 0; il < hp.n_layer; ++il) {
        // --- EC-7: dense layer runs as ONE segment (no router read-back) ---
        if (!im.is_moe) {
            ggml_context *gctx = new_segment_ctx();
            if (gctx == nullptr) { log_error("engine: ggml_init failed"); return false; }
            ggml_tensor *inpL = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hp.n_embd, 1);
            ggml_set_input(inpL);
            ggml_tensor *inp_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
            ggml_set_input(inp_pos);
            ggml_tensor *kq_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_kv, 1);
            ggml_set_input(kq_mask);

            std::vector<ggml_tensor *> stores;
            ggml_tensor *layer_out = build_dense_layer_segment(
                gctx, inpL, inp_pos, il, im.wctx, hp,
                im.k_cache[static_cast<size_t>(il)],
                im.v_cache[static_cast<size_t>(il)], kq_mask, pos, n_kv, stores);

            std::vector<ggml_tensor *> roots = stores;
            roots.push_back(layer_out);
            const bool ok = run_segment(im.backend, im.galloc, gctx, roots, [&]() {
                ggml_backend_tensor_set(inpL, hidden.data(), 0, embd * sizeof(float));
                int32_t pi = pos;
                ggml_backend_tensor_set(inp_pos, &pi, 0, sizeof(pi));
                std::vector<float> mask(static_cast<size_t>(n_kv), 0.0f);
                ggml_backend_tensor_set(kq_mask, mask.data(), 0,
                                        mask.size() * sizeof(float));
            });
            if (ok) {
                ggml_backend_tensor_get(layer_out, hidden.data(), 0, embd * sizeof(float));
            }
            ggml_free(gctx);
            if (!ok) return false;
            continue;
        }

        std::vector<float> cur_host(embd, 0.0f);
        std::vector<float> ffn_inp_host(embd, 0.0f);
        std::vector<float> w_host(n_used, 0.0f);
        std::vector<int32_t> ids_host(n_used, 0);

        // --- Router segment ---
        {
            ggml_context *gctx = new_segment_ctx();
            if (gctx == nullptr) { log_error("engine: ggml_init failed"); return false; }

            ggml_tensor *inpL = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hp.n_embd, 1);
            ggml_set_input(inpL);
            ggml_tensor *inp_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
            ggml_set_input(inp_pos);
            ggml_tensor *kq_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_kv, 1);
            ggml_set_input(kq_mask);

            std::vector<ggml_tensor *> stores;
            ggml_tensor *r_cur = nullptr, *r_ffn = nullptr, *r_w = nullptr,
                        *r_sel = nullptr;
            build_router_segment(gctx, inpL, inp_pos, il, im.wctx, hp,
                                 im.k_cache[static_cast<size_t>(il)],
                                 im.v_cache[static_cast<size_t>(il)], kq_mask,
                                 pos, n_kv, stores, &r_cur, &r_ffn, &r_w, &r_sel);

            std::vector<ggml_tensor *> roots = stores;
            roots.push_back(r_cur);
            roots.push_back(r_ffn);
            roots.push_back(r_w);
            roots.push_back(r_sel);
            const bool ok = run_segment(im.backend, im.galloc, gctx, roots, [&]() {
                ggml_backend_tensor_set(inpL, hidden.data(), 0, embd * sizeof(float));
                int32_t pi = pos;
                ggml_backend_tensor_set(inp_pos, &pi, 0, sizeof(pi));
                std::vector<float> mask(static_cast<size_t>(n_kv), 0.0f);
                ggml_backend_tensor_set(kq_mask, mask.data(), 0,
                                        mask.size() * sizeof(float));
            });
            if (ok) {
                ggml_backend_tensor_get(r_cur, cur_host.data(), 0, embd * sizeof(float));
                ggml_backend_tensor_get(r_ffn, ffn_inp_host.data(), 0, embd * sizeof(float));
                ggml_backend_tensor_get(r_w, w_host.data(), 0, n_used * sizeof(float));
                ggml_backend_tensor_get(r_sel, ids_host.data(), 0, n_used * sizeof(int32_t));
            }
            ggml_free(gctx);
            if (!ok) return false;
        }

        // --- EC-6: measure prefetch effectiveness + feed the predictor ---
        // Before the authoritative residency pass, count how many of this
        // layer's actually-routed experts a prior prefetch had already made
        // resident (a prefetch "hit"). Then observe the real routing so the
        // predictor learns. This read is pure bookkeeping; it does not change
        // what gets computed.
        if (im.predictor != nullptr) {
            for (size_t i = 0; i < n_used; ++i) {
                const int32_t e = ids_host[i];
                if (e < 0 || e >= hp.n_expert) continue;
                ExpertId id{static_cast<uint32_t>(il), static_cast<uint32_t>(e)};
                if (im.warmed_this_token.count(id) != 0) {
                    ++im.prefetch_used;
                }
                im.predictor->observe(id);
            }
        }

        // --- Make the routed top-k experts resident (bounded SlotPool) ---
        if (!ensure_layer_experts_resident(il, ids_host.data(),
                                           static_cast<int>(n_used))) {
            return false;
        }

        // --- EC-6: prefetch the NEXT layer's predicted experts ---
        // Warm layer il+1's likely experts into the pool now, while we are
        // between layers, so its authoritative residency pass tends to hit.
        // Correctness is unaffected: il+1's own router read-back is still
        // authoritative and re-acquires exactly what it selects.
        if (im.predictor != nullptr && il + 1 < hp.n_layer) {
            const uint32_t k = static_cast<uint32_t>(hp.n_expert_used);
            std::vector<uint32_t> pred = im.predictor->predict(
                static_cast<uint32_t>(il + 1), k);
            for (uint32_t pe : pred) {
                ExpertId id{static_cast<uint32_t>(il + 1), pe};
                if (im.pool->resident(id)) continue;  // already warm
                warm_expert(il + 1, static_cast<int>(pe));
                im.warmed_this_token.insert(id);
                ++im.prefetch_warmed;
            }
        }

        // --- Expert segment ---
        {
            ggml_context *gctx = new_segment_ctx();
            if (gctx == nullptr) { log_error("engine: ggml_init failed"); return false; }

            ggml_tensor *cur_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hp.n_embd, 1);
            ggml_set_input(cur_in);
            ggml_tensor *ffn_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hp.n_embd, 1);
            ggml_set_input(ffn_in);
            ggml_tensor *ids_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, hp.n_expert_used);
            ggml_set_input(ids_in);
            ggml_tensor *w_in = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, hp.n_expert_used);
            ggml_set_input(w_in);

            ggml_tensor *layer_out = build_expert_segment(
                gctx, hp, cur_in, ffn_in, ids_in, w_in, im.stage_gate,
                im.stage_up, im.stage_down);

            const bool ok = run_segment(im.backend, im.galloc, gctx, {layer_out}, [&]() {
                ggml_backend_tensor_set(cur_in, cur_host.data(), 0, embd * sizeof(float));
                ggml_backend_tensor_set(ffn_in, ffn_inp_host.data(), 0, embd * sizeof(float));
                ggml_backend_tensor_set(ids_in, ids_host.data(), 0, n_used * sizeof(int32_t));
                ggml_backend_tensor_set(w_in, w_host.data(), 0, n_used * sizeof(float));
            });
            if (ok) {
                ggml_backend_tensor_get(layer_out, hidden.data(), 0, embd * sizeof(float));
            }
            ggml_free(gctx);
            if (!ok) return false;
        }
    }

    // Final norm + lm_head segment.
    {
        ggml_context *gctx = new_segment_ctx();
        if (gctx == nullptr) { log_error("engine: ggml_init failed"); return false; }
        ggml_tensor *inpL = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hp.n_embd, 1);
        ggml_set_input(inpL);
        ggml_tensor *cur = ggml_rms_norm(gctx, inpL, hp.rms_eps);
        cur = ggml_mul(gctx, cur, out_norm);
        cur = ggml_mul_mat(gctx, lm_head, cur);  // [n_vocab, 1]
        ggml_set_output(cur);

        const bool ok = run_segment(im.backend, im.galloc, gctx, {cur}, [&]() {
            ggml_backend_tensor_set(inpL, hidden.data(), 0, embd * sizeof(float));
        });
        if (ok) {
            out.resize(static_cast<size_t>(hp.n_vocab));
            ggml_backend_tensor_get(cur, out.data(), 0,
                                    static_cast<size_t>(hp.n_vocab) * sizeof(float));
        }
        ggml_free(gctx);
        if (!ok) return false;
    }

    // The token at `pos` is now committed to the KV cache; advance bookkeeping.
    im.kv->advance(1);

    // Surface the pool's cache stats (hits/misses/evictions) to the engine.
    // Dense models (EC-7) have no expert pool, so there is nothing to surface.
    if (im.pool != nullptr) {
        const CacheStats &cs = im.pool->stats();
        im.cstats.hits      = cs.hits;
        im.cstats.misses    = cs.misses;
        im.cstats.evictions = cs.evictions;
        im.cstats.prefetch_warmed = im.prefetch_warmed;  // EC-6
        im.cstats.prefetch_used   = im.prefetch_used;
    }
    return true;
}

// ---- oracle + vocab helpers (test support) -------------------------------

bool run_oracle_single_token(const std::string &path, int32_t token,
                             std::vector<float> &out) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        log_error("engine oracle: model load failed");
        return false;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(llama_model_n_ctx_train(model));
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        log_error("engine oracle: context creation failed");
        llama_model_free(model);
        return false;
    }

    llama_token tok = token;
    llama_batch batch = llama_batch_get_one(&tok, 1);
    bool ok = false;
    if (llama_decode(ctx, batch) == 0) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        if (logits != nullptr) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            out.assign(logits, logits + n_vocab);
            ok = true;
        } else {
            log_error("engine oracle: null logits");
        }
    } else {
        log_error("engine oracle: llama_decode failed");
    }

    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

int32_t vocab_bos_token(const std::string &path) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) return -1;
    const llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t bos = llama_vocab_bos(vocab);
    llama_model_free(model);
    return bos;
}

namespace {

// Tokenize `prompt` with `vocab` (add_special=true, parse_special=true),
// returning the ids. Mirrors GgmlModel::tokenize's two-pass sizing.
std::vector<int32_t> tokenize_with_vocab(const llama_vocab *vocab,
                                         const std::string &prompt) {
    std::vector<int32_t> ids;
    const int32_t n_max = static_cast<int32_t>(prompt.size()) + 8;
    std::vector<llama_token> toks(static_cast<size_t>(n_max));
    int32_t n = llama_tokenize(vocab, prompt.c_str(),
                               static_cast<int32_t>(prompt.size()), toks.data(),
                               n_max, /*add_special=*/true,
                               /*parse_special=*/true);
    if (n < 0) {
        toks.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, prompt.c_str(),
                           static_cast<int32_t>(prompt.size()), toks.data(), -n,
                           /*add_special=*/true, /*parse_special=*/true);
    }
    if (n > 0) {
        ids.assign(toks.begin(), toks.begin() + n);
    }
    if (ids.empty()) ids.push_back(llama_vocab_bos(vocab));
    return ids;
}

int32_t argmax_logits(const float *v, int n) {
    int32_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        if (v[i] > best_v) {
            best_v = v[i];
            best = i;
        }
    }
    return best;
}

} // namespace

bool vocab_tokenize(const std::string &path, const std::string &prompt,
                    std::vector<int32_t> &out) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        log_error("engine oracle: tokenize model load failed");
        return false;
    }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    out = tokenize_with_vocab(vocab, prompt);
    llama_model_free(model);
    return !out.empty();
}

bool run_oracle_sequence(const std::string &path, const std::string &prompt,
                         int n_generate, std::vector<int32_t> &out_tokens,
                         std::vector<std::vector<float>> *out_step_logits) {
    out_tokens.clear();
    if (out_step_logits != nullptr) out_step_logits->clear();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        log_error("engine oracle: model load failed");
        return false;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(llama_model_n_ctx_train(model));
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        log_error("engine oracle: context creation failed");
        llama_model_free(model);
        return false;
    }

    std::vector<int32_t> prompt_ids = tokenize_with_vocab(vocab, prompt);

    bool ok = true;
    // Prefill: decode the prompt one token at a time so the per-step logits
    // line up with the engine's one-token-per-call path. Only the final prompt
    // token's logits drive the first generated token.
    const float *logits = nullptr;
    for (size_t i = 0; i < prompt_ids.size() && ok; ++i) {
        llama_token tok = prompt_ids[i];
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            log_error("engine oracle: prompt decode failed");
            ok = false;
            break;
        }
        logits = llama_get_logits_ith(ctx, -1);
    }

    for (int g = 0; g < n_generate && ok; ++g) {
        if (logits == nullptr) {
            log_error("engine oracle: null logits");
            ok = false;
            break;
        }
        if (out_step_logits != nullptr) {
            out_step_logits->emplace_back(logits, logits + n_vocab);
        }
        const int32_t next = argmax_logits(logits, n_vocab);
        out_tokens.push_back(next);

        llama_token tok = next;
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            log_error("engine oracle: generate decode failed");
            ok = false;
            break;
        }
        logits = llama_get_logits_ith(ctx, -1);
    }

    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

} // namespace engine
} // namespace sf

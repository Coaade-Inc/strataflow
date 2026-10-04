// Engine core (EC-1) implementation: build and run OUR OWN ggml_cgraph for the
// llama-arch MoE single-token forward on the ggml CPU backend. This is the
// production form of the proven spike spike/engine_core/spike_engine.cpp; the
// op sequence is lifted verbatim from docs/ENGINE_CORE_DESIGN.md section 6.3,
// which the spike measured against the libllama oracle (argmax match, max
// |logit delta| 5.3e-5 on the tiny MoE).
//
// This is the ONLY engine TU that includes ggml/llama headers; everything above
// the sf::Model seam sees only model/engine/engine.h.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "model/engine/engine.h"

#include "common/log.h"
#include "kv/kv_store.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
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
    ggml_context *wctx = nullptr;  // weight tensors, resident
    EngineHParams hp;

    ggml_context *kvctx = nullptr;              // KV cache tensors, resident
    ggml_backend_buffer_t kvbuf = nullptr;      // backing buffer for kvctx
    std::vector<ggml_tensor *> k_cache;         // per layer [head_dim, n_head_kv, n_ctx]
    std::vector<ggml_tensor *> v_cache;         // per layer [head_dim, n_head_kv, n_ctx]
    std::unique_ptr<KvStore> kv;                // position + capacity bookkeeping

    ~Impl() {
        if (kvbuf != nullptr) ggml_backend_buffer_free(kvbuf);
        if (kvctx != nullptr) ggml_free(kvctx);
        if (wctx != nullptr) ggml_free(wctx);
        if (gc != nullptr) gguf_free(gc);
    }
};

Engine::Engine() : impl_(new Impl()) {}
Engine::~Engine() = default;

const EngineHParams &Engine::hparams() const { return impl_->hp; }

std::unique_ptr<Engine> Engine::load(const std::string &path) {
    std::unique_ptr<Engine> eng(new Engine());
    Impl &im = *eng->impl_;

    // Read trunk + expert tensors resident. no_alloc=false lets ggml allocate
    // and fill every tensor so we can fetch each weight by its GGUF name. All
    // experts resident (no streaming) is EC-1 scope; EC-3 adds residency.
    gguf_init_params gp{};
    gp.no_alloc = false;
    gp.ctx = &im.wctx;
    im.gc = gguf_init_from_file(path.c_str(), gp);
    if (im.gc == nullptr || im.wctx == nullptr) {
        log_warn("engine: gguf open failed for '" + path + "'");
        return nullptr;
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
    ggml_tensor *ge0      = get_weight(im.wctx, "blk.0.ffn_gate_exps.weight");
    if (tok_embd == nullptr || out_w == nullptr || ge0 == nullptr) {
        return nullptr;
    }

    hp.n_vocab = static_cast<int>(tok_embd->ne[1]);
    if (hp.n_embd == 0) hp.n_embd = static_cast<int>(tok_embd->ne[0]);
    hp.n_ff    = static_cast<int>(ge0->ne[1]);  // gate_exps ne = [n_embd, n_ff, n_expert]

    if (hp.n_head <= 0 || hp.n_layer <= 0 || hp.n_embd <= 0 ||
        hp.n_expert <= 0 || hp.n_expert_used <= 0) {
        log_warn("engine: incomplete hparams; staying on libllama path");
        return nullptr;
    }
    hp.head_dim = hp.n_embd / hp.n_head;

    // Validate the expert-tensor ne-order against the design doc (6.2) so a
    // layout surprise is caught here, not as a wrong logit later.
    if (ge0->ne[0] != hp.n_embd || ge0->ne[2] != hp.n_expert) {
        log_warn("engine: ffn_gate_exps shape mismatch; staying on libllama path");
        return nullptr;
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

    log_info("engine: loaded llama-arch MoE (" + std::to_string(hp.n_layer) +
             " layers, " + std::to_string(hp.n_expert) + "x" +
             std::to_string(hp.n_expert_used) + " experts, n_embd=" +
             std::to_string(hp.n_embd) + ", n_vocab=" +
             std::to_string(hp.n_vocab) + ", n_ctx=" + std::to_string(n_ctx) +
             ")");
    return eng;
}

void Engine::reset_kv() {
    if (impl_->kv != nullptr) impl_->kv->reset();
}

namespace {

// Build one transformer layer, mirroring llama_model_llama::graph and
// build_moe_ffn (docs/ENGINE_CORE_DESIGN.md 6.3 step 2). `inpL` is
// [n_embd, n_tokens=1]; returns the layer output [n_embd, 1].
//
// EC-2 KV cache (section 3.4): this layer's post-rope K and this layer's V for
// the current token are copied into the persistent per-layer cache tensors at
// column `pos`; attention then views the first `n_kv = pos+1` cached columns
// and applies the causal mask through ggml_soft_max_ext. `k_cache`/`v_cache`
// are the layer's resident cache tensors [head_dim, n_head_kv, n_ctx]; stores
// into them are forced into the graph via `graph_stores`.
ggml_tensor *build_layer(ggml_context *gctx, ggml_tensor *inpL,
                         ggml_tensor *inp_pos, int il, ggml_context *wctx,
                         const EngineHParams &hp, ggml_tensor *k_cache,
                         ggml_tensor *v_cache, ggml_tensor *kq_mask,
                         int64_t pos, int64_t n_kv,
                         std::vector<ggml_tensor *> &graph_stores) {
    const std::string p = "blk." + std::to_string(il) + ".";

    ggml_tensor *attn_norm_w = ggml_get_tensor(wctx, (p + "attn_norm.weight").c_str());
    ggml_tensor *wq          = ggml_get_tensor(wctx, (p + "attn_q.weight").c_str());
    ggml_tensor *wk          = ggml_get_tensor(wctx, (p + "attn_k.weight").c_str());
    ggml_tensor *wv          = ggml_get_tensor(wctx, (p + "attn_v.weight").c_str());
    ggml_tensor *wo          = ggml_get_tensor(wctx, (p + "attn_output.weight").c_str());
    ggml_tensor *ffn_norm_w  = ggml_get_tensor(wctx, (p + "ffn_norm.weight").c_str());
    ggml_tensor *gate_inp    = ggml_get_tensor(wctx, (p + "ffn_gate_inp.weight").c_str());
    ggml_tensor *gate_exps   = ggml_get_tensor(wctx, (p + "ffn_gate_exps.weight").c_str());
    ggml_tensor *down_exps   = ggml_get_tensor(wctx, (p + "ffn_down_exps.weight").c_str());
    ggml_tensor *up_exps     = ggml_get_tensor(wctx, (p + "ffn_up_exps.weight").c_str());

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
    // Kcur/Vcur are [head_dim, n_head_kv, n_tokens=1]; the cache view selects
    // the single column `pos`. ggml_cpy records the store as a graph node; we
    // keep its handle so forward() expands it (otherwise the store could be
    // pruned, since the cache tensor, not the cpy, is what attention reads).
    const size_t kv_col_nb = k_cache->nb[2];  // bytes per token-column
    ggml_tensor *k_dst = ggml_view_3d(gctx, k_cache, hp.head_dim, hp.n_head_kv,
                                      n_tokens, k_cache->nb[1], k_cache->nb[2],
                                      static_cast<size_t>(pos) * kv_col_nb);
    ggml_tensor *v_dst = ggml_view_3d(gctx, v_cache, hp.head_dim, hp.n_head_kv,
                                      n_tokens, v_cache->nb[1], v_cache->nb[2],
                                      static_cast<size_t>(pos) * v_cache->nb[2]);
    graph_stores.push_back(ggml_cpy(gctx, Kcur, k_dst));
    graph_stores.push_back(ggml_cpy(gctx, Vcur, v_dst));

    // Attention, mirroring the explicit (non-flash) else-branch of
    // build_attn_mha, now over KV history. Gather the first n_kv cached columns
    // of K and V as [head_dim, n_head_kv, n_kv] views, then permute to the
    // attention layout. The causal mask is applied via ggml_soft_max_ext.
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
    // Causal mask [n_kv, n_tokens]: 0 for attendable positions, -inf otherwise.
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

    // MoE (build_moe_ffn: SOFTMAX gating, norm_w=true, w_scale=1).
    ggml_tensor *router   = ggml_mul_mat(gctx, gate_inp, cur);   // [n_expert, n_tokens]
    ggml_tensor *probs    = ggml_soft_max(gctx, router);         // [n_expert, n_tokens]
    ggml_tensor *selected = ggml_argsort_top_k(gctx, probs, hp.n_expert_used);

    ggml_tensor *probs3  = ggml_reshape_3d(gctx, probs, 1, hp.n_expert, n_tokens);
    ggml_tensor *weights = ggml_get_rows(gctx, probs3, selected);  // [1, n_used, n_tokens]

    // Normalize weights by their clamped sum (clamp constant from build_moe_ffn).
    weights = ggml_reshape_2d(gctx, weights, hp.n_expert_used, n_tokens);
    ggml_tensor *wsum = ggml_sum_rows(gctx, weights);           // [1, n_tokens]
    wsum = ggml_clamp(gctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(gctx, weights, wsum);
    weights = ggml_reshape_3d(gctx, weights, 1, hp.n_expert_used, n_tokens);

    ggml_tensor *cur3 = ggml_reshape_3d(gctx, cur, hp.n_embd, 1, n_tokens);
    ggml_tensor *up   = ggml_mul_mat_id(gctx, up_exps,   cur3, selected);
    ggml_tensor *gate = ggml_mul_mat_id(gctx, gate_exps, cur3, selected);
    ggml_tensor *act  = ggml_swiglu_split(gctx, gate, up);
    ggml_tensor *experts = ggml_mul_mat_id(gctx, down_exps, act, selected);

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

    return ggml_add(gctx, moe_out, ffn_inp);  // residual
}

} // namespace

bool Engine::forward(int32_t token, int32_t pos, std::vector<float> &out) {
    Impl &im = *impl_;
    const EngineHParams &hp = im.hp;

    ggml_tensor *tok_embd = ggml_get_tensor(im.wctx, "token_embd.weight");
    ggml_tensor *out_norm = ggml_get_tensor(im.wctx, "output_norm.weight");
    ggml_tensor *lm_head  = ggml_get_tensor(im.wctx, "output.weight");
    if (tok_embd == nullptr || out_norm == nullptr || lm_head == nullptr) {
        log_error("engine: missing trunk tensor at forward");
        return false;
    }

    // Position must land inside the trained context length. We trust the caller
    // to drive positions in order, but bound-check against the KV capacity so a
    // runaway sequence fails loudly instead of writing out of bounds.
    if (pos < 0 || im.kv == nullptr ||
        pos >= static_cast<int32_t>(im.kv->capacity())) {
        log_error("engine: position " + std::to_string(pos) +
                  " out of KV bounds (capacity " +
                  std::to_string(im.kv != nullptr ? im.kv->capacity() : 0) + ")");
        return false;
    }
    const int64_t n_kv = static_cast<int64_t>(pos) + 1;  // attend over 0..pos

    // Graph build context (no_alloc=true): nodes get data from gallocr. Sized
    // generously for the fixed per-token topology (a few dozen nodes/layer).
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context *gctx = ggml_init(ip);
    if (gctx == nullptr) {
        log_error("engine: ggml_init failed");
        return false;
    }

    // Inputs: the single token id, its position, and the causal mask. The mask
    // is [n_kv, 1]: a single query token at `pos` attends over all cached
    // positions 0..pos, so every entry is attendable (0.0). The general shape
    // (nonzero -inf entries) is kept so a future multi-token prefill batch can
    // reuse this path unchanged.
    ggml_tensor *inp_tok = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_input(inp_tok);
    ggml_set_name(inp_tok, "inp_tok");
    ggml_tensor *inp_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_input(inp_pos);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_tensor *kq_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_kv, 1);
    ggml_set_input(kq_mask);
    ggml_set_name(kq_mask, "kq_mask");

    std::vector<ggml_tensor *> graph_stores;
    ggml_tensor *inpL = ggml_get_rows(gctx, tok_embd, inp_tok);  // [n_embd, 1]
    for (int il = 0; il < hp.n_layer; ++il) {
        inpL = build_layer(gctx, inpL, inp_pos, il, im.wctx, hp,
                           im.k_cache[static_cast<size_t>(il)],
                           im.v_cache[static_cast<size_t>(il)], kq_mask, pos,
                           n_kv, graph_stores);
    }

    // Final norm + lm_head.
    ggml_tensor *cur = ggml_rms_norm(gctx, inpL, hp.rms_eps);
    cur = ggml_mul(gctx, cur, out_norm);
    cur = ggml_mul_mat(gctx, lm_head, cur);  // [n_vocab, 1]
    ggml_set_output(cur);
    ggml_set_name(cur, "result_logits");

    ggml_cgraph *gf = ggml_new_graph(gctx);
    // Expand the KV stores first so they are not pruned (attention reads the
    // cache tensor, not the cpy node, so the cpy must be rooted explicitly).
    for (ggml_tensor *st : graph_stores) {
        ggml_build_forward_expand(gf, st);
    }
    ggml_build_forward_expand(gf, cur);

    // Allocate + compute on the CPU backend (1 thread for determinism, matching
    // the greedy-determinism contract in CONTRIBUTING.md).
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        log_error("engine: cpu backend init failed");
        ggml_free(gctx);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 1);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    bool ok = galloc != nullptr && ggml_gallocr_alloc_graph(galloc, gf);
    if (!ok) {
        log_error("engine: gallocr alloc failed");
        if (galloc != nullptr) ggml_gallocr_free(galloc);
        ggml_backend_free(backend);
        ggml_free(gctx);
        return false;
    }

    // Set inputs AFTER allocation.
    int32_t tok_i = token;
    int32_t pos_i = pos;
    ggml_backend_tensor_set(inp_tok, &tok_i, 0, sizeof(tok_i));
    ggml_backend_tensor_set(inp_pos, &pos_i, 0, sizeof(pos_i));
    // Causal mask for a single query token at `pos`: all cached positions
    // 0..pos are <= pos, so all are attendable (0.0). (A multi-token batch
    // would set -inf for j > i; left as the general shape for reuse.)
    std::vector<float> mask(static_cast<size_t>(n_kv), 0.0f);
    ggml_backend_tensor_set(kq_mask, mask.data(), 0,
                            mask.size() * sizeof(float));

    const bool compute_ok =
        ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (compute_ok) {
        out.resize(static_cast<size_t>(hp.n_vocab));
        ggml_backend_tensor_get(cur, out.data(), 0,
                                static_cast<size_t>(hp.n_vocab) * sizeof(float));
        // The token at `pos` is now committed to the KV cache; advance the
        // bookkeeping so used()/bytes() reflect it.
        im.kv->advance(1);
    } else {
        log_error("engine: graph compute failed");
    }

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(gctx);
    return compute_ok;
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

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

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>
#include <llama.h>

#include <cmath>
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
// gguf no_alloc=false, matching the spike's reliable path) and the parsed
// hparams. The weight context is kept alive for the engine's lifetime so the
// graph leaves (weight tensors) stay backed across every forward() call.
struct Engine::Impl {
    gguf_context *gc   = nullptr;
    ggml_context *wctx = nullptr;  // weight tensors, resident
    EngineHParams hp;

    ~Impl() {
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

    log_info("engine: loaded llama-arch MoE (" + std::to_string(hp.n_layer) +
             " layers, " + std::to_string(hp.n_expert) + "x" +
             std::to_string(hp.n_expert_used) + " experts, n_embd=" +
             std::to_string(hp.n_embd) + ", n_vocab=" +
             std::to_string(hp.n_vocab) + ")");
    return eng;
}

namespace {

// Build one transformer layer, mirroring llama_model_llama::graph and
// build_moe_ffn (docs/ENGINE_CORE_DESIGN.md 6.3 step 2). `cur_in` is
// [n_embd, n_tokens=1]; returns the layer output [n_embd, 1].
ggml_tensor *build_layer(ggml_context *gctx, ggml_tensor *inpL,
                         ggml_tensor *inp_pos, int il, ggml_context *wctx,
                         const EngineHParams &hp) {
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

    // Attention, mirroring the explicit (non-flash) else-branch of
    // build_attn_mha. No KV history (n_kv == n_tokens == 1), so NULL mask.
    ggml_tensor *q = ggml_permute(gctx, Qcur, 0, 2, 1, 3);  // [head_dim, n_tokens, n_head]
    ggml_tensor *k = ggml_permute(gctx, Kcur, 0, 2, 1, 3);
    ggml_tensor *v = ggml_permute(gctx, Vcur, 0, 2, 1, 3);

    ggml_tensor *kq = ggml_mul_mat(gctx, k, q);
    // The reference forces GGML_PREC_F32 accumulation on kq; match it.
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(hp.head_dim));
    kq = ggml_soft_max_ext(gctx, kq, nullptr, kq_scale, 0.0f);

    ggml_tensor *vt = ggml_cont(gctx, ggml_transpose(gctx, v));
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

    // Inputs: the single token id and its position.
    ggml_tensor *inp_tok = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_input(inp_tok);
    ggml_set_name(inp_tok, "inp_tok");
    ggml_tensor *inp_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_set_input(inp_pos);
    ggml_set_name(inp_pos, "inp_pos");

    ggml_tensor *inpL = ggml_get_rows(gctx, tok_embd, inp_tok);  // [n_embd, 1]
    for (int il = 0; il < hp.n_layer; ++il) {
        inpL = build_layer(gctx, inpL, inp_pos, il, im.wctx, hp);
    }

    // Final norm + lm_head.
    ggml_tensor *cur = ggml_rms_norm(gctx, inpL, hp.rms_eps);
    cur = ggml_mul(gctx, cur, out_norm);
    cur = ggml_mul_mat(gctx, lm_head, cur);  // [n_vocab, 1]
    ggml_set_output(cur);
    ggml_set_name(cur, "result_logits");

    ggml_cgraph *gf = ggml_new_graph(gctx);
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

    const bool compute_ok =
        ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (compute_ok) {
        out.resize(static_cast<size_t>(hp.n_vocab));
        ggml_backend_tensor_get(cur, out.data(), 0,
                                static_cast<size_t>(hp.n_vocab) * sizeof(float));
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

} // namespace engine
} // namespace sf

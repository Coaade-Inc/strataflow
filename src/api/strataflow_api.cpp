// C API implementation: wires the modules together behind the stable ABI
// declared in include/strataflow/strataflow.h.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "strataflow/strataflow.h"

#include "common/log.h"
#include "hw/profiler.h"
#include "kv/kv_store.h"
#include "model/model.h"
#include "plan/llama_placement.h"
#include "plan/planner.h"
#include "predict/predictor.h"
#include "sched/scheduler.h"

#include <cstring>
#include <memory>
#include <string>

// ---- opaque types ---------------------------------------------------------
struct sf_context {
    sf::HardwareProfile          hw;
    sf::PlacementPlan            plan;
    std::unique_ptr<sf::Model>   model;
    sf_context_params            params{};
};

struct sf_session {
    sf_context                     *ctx = nullptr;
    std::unique_ptr<sf::KvStore>    kv;
    std::unique_ptr<sf::ExpertPredictor> predictor;
};

extern "C" {

sf_context_params sf_context_default_params(void) {
    sf_context_params p{};
    p.model_path        = nullptr;
    p.vram_budget       = 0;
    p.ram_budget        = 0;
    p.n_threads         = 0;
    p.n_ctx             = 0;
    p.auto_plan         = 1;
    p.preferred_backend = SF_BACKEND_CPU;
    return p;
}

sf_sampling_params sf_sampling_default_params(void) {
    sf_sampling_params s{};
    s.temperature = 0.0f;   // greedy by default (reproducible)
    s.top_p       = 1.0f;
    s.top_k       = 0;
    s.seed        = 0;
    s.max_tokens  = 256;
    return s;
}

sf_status sf_context_create(const sf_context_params *params, sf_context **out_ctx) {
    if (params == nullptr || out_ctx == nullptr) return SF_ERR_INVALID_ARGUMENT;
    if (params->model_path == nullptr)           return SF_ERR_INVALID_ARGUMENT;
    *out_ctx = nullptr;

    auto ctx = std::make_unique<sf_context>();
    ctx->params = *params;

    ctx->hw = sf::profile_hardware(params->model_path, /*quick=*/false);

    // The planner runs INSIDE load_model now, so the chosen placement actually
    // drives how the GGUF backend loads the model (VRAM layers + expert
    // offload). An explicit budget of 0 means "auto from the profile".
    sf_status st = sf::load_model(params->model_path, ctx->hw,
                                  params->vram_budget, params->ram_budget,
                                  ctx->model, &ctx->plan);
    if (st != SF_OK) return st;

    *out_ctx = ctx.release();
    return SF_OK;
}

void sf_context_free(sf_context *ctx) { delete ctx; }

sf_status sf_session_create(sf_context *ctx, sf_session **out_session) {
    if (ctx == nullptr || out_session == nullptr) return SF_ERR_INVALID_ARGUMENT;
    auto s = std::make_unique<sf_session>();
    s->ctx = ctx;

    const auto &shape = ctx->model->shape();
    uint32_t n_ctx = ctx->params.n_ctx ? ctx->params.n_ctx : 4096;
    // Placeholder KV footprint; replaced by the real per-token size in Phase 6.
    s->kv = std::make_unique<sf::KvStore>(n_ctx, /*bytes_per_token=*/256 * 1024);
    s->predictor = std::make_unique<sf::ExpertPredictor>(
        shape.n_layers ? shape.n_layers : 1,
        shape.n_experts ? shape.n_experts : 1);

    *out_session = s.release();
    return SF_OK;
}

void sf_session_free(sf_session *session) { delete session; }

sf_status sf_generate(sf_session *session, const char *prompt,
                      const sf_sampling_params *sampling,
                      sf_token_callback cb, void *user_data) {
    if (session == nullptr || prompt == nullptr || cb == nullptr) {
        return SF_ERR_INVALID_ARGUMENT;
    }
    sf_sampling_params s = sampling ? *sampling : sf_sampling_default_params();

    sf::Model &model = *session->ctx->model;
    sf::Scheduler sched(model, *session->kv, *session->predictor);

    std::vector<int32_t> ids = model.tokenize(prompt);
    if (!session->kv->advance(static_cast<uint32_t>(ids.size()))) {
        return SF_ERR_RUNTIME;  // prompt longer than context
    }

    int32_t token = ids.empty() ? 1 : ids.back();
    int32_t limit = s.max_tokens > 0 ? s.max_tokens : 256;

    for (int32_t i = 0; i < limit; ++i) {
        token = sched.step(token);
        if (token == model.eos_token()) break;
        if (!session->kv->advance(1)) break;

        std::string piece = model.detokenize(token);
        if (cb(piece.c_str(), user_data) != 0) break;  // caller asked to stop
    }
    return SF_OK;
}

sf_status sf_describe_plan(sf_context *ctx, char *buf, size_t buf_size) {
    if (ctx == nullptr || buf == nullptr || buf_size == 0) {
        return SF_ERR_INVALID_ARGUMENT;
    }
    // Surface the concrete auto-decision (what llama.cpp was actually told):
    // how many trunk layers went to VRAM, and whether MoE experts were forced
    // onto the CPU. derive_llama_placement is the same pure translation the
    // GGUF backend applied, so this reflects the real load params.
    sf::LlamaPlacement place = sf::derive_llama_placement(
        ctx->plan, ctx->model->shape(), ctx->hw.gpus);
    std::string decision =
        "auto[n_gpu_layers=" + std::to_string(place.n_gpu_layers) +
        ", experts=" + (place.offload_experts_to_cpu ? "CPU-offload" : "default") +
        "]";

    std::string line = ctx->hw.to_summary() + " | " + ctx->plan.to_summary() +
                       " | " + decision;
    std::strncpy(buf, line.c_str(), buf_size - 1);
    buf[buf_size - 1] = '\0';
    return SF_OK;
}

} // extern "C"

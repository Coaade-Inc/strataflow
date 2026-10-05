// C API implementation: wires the modules together behind the stable ABI
// declared in include/strataflow/strataflow.h.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "strataflow/strataflow.h"

#include "common/log.h"
#include "hw/profiler.h"
#include "kv/kv_store.h"
#include "model/model.h"
#include "plan/llama_placement.h"
#include "plan/planner.h"
#include "predict/predictor.h"
#include "sched/scheduler.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#elif defined(__APPLE__)
#  include <mach/mach.h>
#else
#  include <sys/resource.h>
#endif

namespace {
// Best-effort process peak resident set size (peak RAM) in bytes; 0 if the
// platform cannot report it.
uint64_t query_peak_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<uint64_t>(pmc.PeakWorkingSetSize);
    }
    return 0;
#elif defined(__APPLE__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return static_cast<uint64_t>(ru.ru_maxrss);  // bytes on macOS
    }
    return 0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return static_cast<uint64_t>(ru.ru_maxrss) * 1024ull;  // KiB on Linux
    }
    return 0;
#endif
}
} // namespace

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
    p.expert_slots      = 0;   // auto (hold the whole expert working set)
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

    // Honor an explicit expert-slot budget by routing it to the engine via the
    // same env the engine already reads (STRATAFLOW_ENGINE_EXPERT_SLOTS). This
    // keeps load_model's signature stable while making the CLI flag effective.
    // A caller-set env var still wins if the param is left at auto (0).
    if (params->expert_slots != 0) {
        const std::string v = std::to_string(params->expert_slots);
#if defined(_WIN32)
        _putenv_s("STRATAFLOW_ENGINE_EXPERT_SLOTS", v.c_str());
#else
        setenv("STRATAFLOW_ENGINE_EXPERT_SLOTS", v.c_str(), /*overwrite=*/1);
#endif
    }

    // profile_hardware benchmarks the DISK the model sits on, so it needs the
    // containing DIRECTORY, not the model FILE path. Deriving it here keeps the
    // SSD probe from trying to write '<file>/.strataflow_ssd_probe.tmp' (which
    // fails and warns). Strip the trailing component after the last separator
    // ('/' on POSIX, '/' or '\\' on Windows); if there is no separator, use ""
    // (the current working directory).
    const std::string model_dir = sf::containing_dir(params->model_path);
    ctx->hw = sf::profile_hardware(model_dir, /*quick=*/false);

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

sf_status sf_session_stats(sf_session *session, sf_runtime_stats *out) {
    if (session == nullptr || out == nullptr) return SF_ERR_INVALID_ARGUMENT;
    const bool have_model =
        session->ctx != nullptr && session->ctx->model != nullptr;
    out->resident_weight_bytes =
        have_model ? session->ctx->model->resident_weight_bytes() : 0;
    out->peak_rss_bytes = query_peak_rss_bytes();
    // Expert bytes streamed from disk (ground truth from the engine); 0 on the
    // plain-GGUF/dry-run paths. TTFT is measured by the CLI around sf_generate
    // (the engine/ABI do not time it), so zero-init it here; keeping the field
    // in the struct documents it for any future programmatic caller.
    out->streamed_bytes = have_model ? session->ctx->model->streamed_bytes() : 0;
    out->ttft_us = 0;
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

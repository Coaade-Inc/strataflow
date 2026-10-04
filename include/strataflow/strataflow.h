/*
 * StrataFlow public C API.
 *
 * Stable C ABI so the CLI, the HTTP server, and language bindings (Python via
 * nanobind) all share one entry point. Everything below the API -- the tiered
 * weight store, the scheduler, the predictor -- is C++ and private.
 *
 * Phase 1 status: the surface is defined and the engine is wired end to end,
 * but model execution is a stub until llama.cpp/ggml is vendored (Phase 1b).
 * Calls behave sanely (return errors / placeholder output) so tools and tests
 * can be built against the final API shape today.
 *
 * Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
 */
#ifndef STRATAFLOW_H
#define STRATAFLOW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(STRATAFLOW_SHARED)
#  ifdef STRATAFLOW_BUILD
#    define SF_API __declspec(dllexport)
#  else
#    define SF_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(STRATAFLOW_SHARED)
#  define SF_API __attribute__((visibility("default")))
#else
#  define SF_API
#endif

/* ---- version ---------------------------------------------------------- */
#define STRATAFLOW_VERSION_MAJOR 0
#define STRATAFLOW_VERSION_MINOR 0
#define STRATAFLOW_VERSION_PATCH 1

/* Returns a human-readable version string, e.g. "0.0.1". */
SF_API const char *sf_version(void);

/* ---- status codes ----------------------------------------------------- */
typedef enum sf_status {
    SF_OK = 0,
    SF_ERR_INVALID_ARGUMENT = 1,
    SF_ERR_FILE_NOT_FOUND   = 2,
    SF_ERR_UNSUPPORTED      = 3,  /* feature not implemented yet */
    SF_ERR_OUT_OF_MEMORY    = 4,
    SF_ERR_IO               = 5,
    SF_ERR_MODEL_LOAD       = 6,
    SF_ERR_RUNTIME          = 7,
} sf_status;

/* Human-readable message for a status code. Never NULL. */
SF_API const char *sf_status_str(sf_status status);

/* ---- devices / tiers -------------------------------------------------- */
/* The memory strata a weight unit can live in, fastest to slowest. */
typedef enum sf_tier {
    SF_TIER_VRAM = 0,
    SF_TIER_RAM  = 1,
    SF_TIER_SSD  = 2,
} sf_tier;

typedef enum sf_backend {
    SF_BACKEND_CPU    = 0,
    SF_BACKEND_CUDA   = 1,
    SF_BACKEND_VULKAN = 2,
    SF_BACKEND_METAL  = 3,
    SF_BACKEND_HIP    = 4,
} sf_backend;

/* ---- opaque handles --------------------------------------------------- */
typedef struct sf_context sf_context;   /* engine instance + loaded model */
typedef struct sf_session sf_session;   /* one conversation / KV state    */

/* ---- model / context params ------------------------------------------- */
typedef struct sf_context_params {
    /* Path to a .gguf or .strata model (or a .strata directory). */
    const char *model_path;

    /* Memory budgets in bytes. 0 means "let the planner decide from the
       hardware profile". These mirror the kimi-k3-in-c trunk/cache dials. */
    uint64_t vram_budget;   /* VRAM for trunk + hot-expert cache */
    uint64_t ram_budget;    /* RAM for resident trunk + warm-expert cache */

    uint32_t n_threads;     /* 0 = auto (profiler picks P-cores) */
    uint32_t n_ctx;         /* max context length; 0 = model default */

    /* If non-zero, run the hardware profiler on load and auto-place weights.
       If zero, use the explicit budgets above with a simple default plan. */
    int32_t  auto_plan;

    sf_backend preferred_backend; /* ignored if the backend is unavailable */

    /* Bound the streamed-expert residency pool to this many expert bundles
       (one bundle = gate+up+down for one (layer,expert)). 0 = auto (hold the
       whole expert working set, no eviction). A value below the model's expert
       count forces bounded streaming with LRU eviction - the "run it in a small
       budget" dial. Appended for ABI compatibility; default 0. */
    uint32_t expert_slots;
} sf_context_params;

/* Returns params filled with safe defaults (auto_plan on, CPU backend). */
SF_API sf_context_params sf_context_default_params(void);

/* ---- sampling params -------------------------------------------------- */
typedef struct sf_sampling_params {
    float    temperature;   /* <= 0 means greedy (argmax)         */
    float    top_p;         /* 1.0 disables nucleus sampling      */
    int32_t  top_k;         /* <= 0 disables top-k                */
    uint64_t seed;          /* for reproducible sampling          */
    int32_t  max_tokens;    /* generation cap for one request     */
} sf_sampling_params;

SF_API sf_sampling_params sf_sampling_default_params(void);

/* ---- lifecycle -------------------------------------------------------- */
/* Create an engine context and load a model.
 * On success *out_ctx is set and SF_OK is returned. On failure *out_ctx is
 * left NULL. During Phase 1 (no ggml yet) this may return SF_ERR_UNSUPPORTED
 * for real weights but still succeed for the built-in dry-run model. */
SF_API sf_status sf_context_create(const sf_context_params *params,
                                   sf_context **out_ctx);

SF_API void sf_context_free(sf_context *ctx);

/* Open a session (owns a KV cache). */
SF_API sf_status sf_session_create(sf_context *ctx, sf_session **out_session);
SF_API void      sf_session_free(sf_session *session);

/* ---- generation ------------------------------------------------------- */
/* Streaming token callback. Return non-zero to stop generation early.
 * `text` is a UTF-8 piece for the newly produced token (not NUL-delimited by
 * token boundaries; concatenate across calls). */
typedef int (*sf_token_callback)(const char *text, void *user_data);

/* Generate a completion for `prompt`, streaming pieces to `cb`.
 * Blocks until generation stops. */
SF_API sf_status sf_generate(sf_session *session,
                             const char *prompt,
                             const sf_sampling_params *sampling,
                             sf_token_callback cb,
                             void *user_data);

/* ---- runtime stats ---------------------------------------------------- */
typedef struct sf_runtime_stats {
    /* Bytes of model WEIGHTS held resident in RAM. On the .strata streaming
     * path this EXCLUDES the streamed expert footprint - the proof that a large
     * model runs in a small resident budget. 0 if not applicable. Known only
     * after generation has started (the engine loads lazily). */
    uint64_t resident_weight_bytes;
    /* Process peak resident set size (peak RAM) in bytes, best-effort per OS;
     * 0 if the platform could not report it. */
    uint64_t peak_rss_bytes;
    /* NOTE: fields below are APPENDED for ABI backward-compatibility. Append
     * new fields at the END of this struct only; never reorder or remove the
     * fields above (existing callers depend on the layout). */
    /* Time to first token in microseconds: wall-clock from the start of
     * generation to the FIRST produced token. The engine/ABI do not time this;
     * the CLI measures it around sf_generate and the harness reads it from the
     * CLI output. 0 if not measured (e.g. no token was produced, or the caller
     * did not time it). */
    uint64_t ttft_us;
    /* Total EXPERT bytes read from the .strata/disk during the run (per-token
     * expert streaming). On the plain-GGUF path, where experts are resident,
     * this stays 0. Excludes the one-time trunk load. 0 if not applicable. */
    uint64_t streamed_bytes;
} sf_runtime_stats;

/* Fill `out` with current runtime stats. Call after sf_generate to see the
 * resident weight budget and peak RSS. Returns SF_ERR_INVALID_ARGUMENT on a
 * null pointer. */
SF_API sf_status sf_session_stats(sf_session *session, sf_runtime_stats *out);

/* ---- introspection ---------------------------------------------------- */
/* Fills `buf` (size `buf_size`) with a one-line human summary of the detected
 * hardware and the chosen placement plan. Always NUL-terminates. */
SF_API sf_status sf_describe_plan(sf_context *ctx, char *buf, size_t buf_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STRATAFLOW_H */

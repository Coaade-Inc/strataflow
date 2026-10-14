// Model: owns weights, tokenizer, and the forward pass. This is the primary
// seam where llama.cpp/ggml plugs in during Phase 1b. See docs/PLAN.md 3.x.
//
// Phase 1 provides a deterministic "dry-run" model: it reports a realistic
// ModelShape and produces reproducible token ids, so the engine, CLI, server,
// and tests all run end to end before any real weights exist. When ggml is
// vendored, GgmlModel implements this same interface and the dry-run model
// stays as a test fixture.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include "hw/profiler.h"
#include "plan/planner.h"
#include "strataflow/strataflow.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sf {

class Model {
public:
    virtual ~Model() = default;

    virtual const ModelShape &shape() const = 0;

    // Tokenize a prompt into ids (naive whitespace scheme in the dry-run model).
    virtual std::vector<int32_t> tokenize(const std::string &text) const = 0;
    virtual std::string detokenize(int32_t token) const = 0;

    // Produce the next token id given the previous one. Deterministic in the
    // dry-run model so greedy output is reproducible (the determinism contract
    // from docs/PLAN.md section 7).
    virtual int32_t forward(int32_t last_token) = 0;

    // Token id that signals end of generation.
    virtual int32_t eos_token() const = 0;

    // Bytes of model WEIGHTS held resident in RAM. On the .strata streaming
    // path this excludes the streamed expert footprint, so it is the proof that
    // a large model runs in a small resident budget. 0 if not applicable (e.g.
    // the plain-GGUF path where ggml owns the mapping, or the dry-run model).
    virtual uint64_t resident_weight_bytes() const { return 0; }

    // Total EXPERT bytes streamed from disk during the run (per-token expert
    // streaming on the .strata path). 0 when not applicable: the plain-GGUF
    // path where experts are resident, or the dry-run model. Excludes the
    // one-time trunk load.
    virtual uint64_t streamed_bytes() const { return 0; }

    // FEAT-003 async-prefetch overlap signals, surfaced from the engine's
    // EngineCacheStats. prefetch_warmed = experts speculatively warmed ahead of
    // a layer; prefetch_used = of those, how many that layer then routed;
    // prefetch_completed_before_use = of the used ones, how many the background
    // I/O worker had fully read AND installed before the authoritative acquire
    // (i.e. the overlap actually hid the disk read). All 0 when not applicable
    // (plain-GGUF/dry-run, or a fully-resident pool with no streaming).
    virtual uint64_t prefetch_warmed() const { return 0; }
    virtual uint64_t prefetch_used() const { return 0; }
    virtual uint64_t prefetch_completed_before_use() const { return 0; }

    // FEAT-002 (technique 6.5, lossless KV paging): the RESERVED resident KV
    // cache footprint in bytes (n_layer * 2 * head_dim * n_head_kv *
    // sizeof(float) * W, W = the resident window). When the window is smaller
    // than the trained n_ctx this is strictly less than the full-n_ctx
    // reservation; older positions are tiered losslessly to a RAM cold store so
    // decode stays byte-identical. 0 when not applicable (dry-run, or no engine
    // loaded yet).
    virtual uint64_t resident_kv_bytes() const { return 0; }
};

// Factory: inspects `path` and returns the right implementation.
//   * An existing *.gguf file is loaded by the vendored llama.cpp backend
//     (GgmlModel). On any load failure it falls back to the dry-run model.
//   * Anything else (missing file, *.strata, unknown extension) falls back to
//     the deterministic dry-run model with a logged warning.
// Never crashes on a bad path: always yields a usable model and SF_OK.
//
// `hw` and the budgets (0 = auto) feed the placement planner so the GGUF
// backend can decide VRAM layer offload / expert placement BEFORE the heavy
// tensor load. `out_plan`, when non-null, receives the plan the loader applied
// (for the GGUF path) or an auto plan over the chosen model's shape (dry-run),
// so the C API can describe exactly what was decided.
//
// `mode` and `target_ram_budget` (task-target-ram-modes) select the usable-RAM
// ceiling the planner sizes against; they are appended as DEFAULTED trailing
// params (RamMode::Auto, target 0) so existing callers compile unchanged, and
// forwarded into every plan_placement call below.
//
// FEAT-003: `preferred_backend` mirrors the public sf_backend enum (0 ==
// SF_BACKEND_CPU, the default). It is forwarded to the engine so a GPU build
// can select a GPU ggml backend; on a CPU-only build (the default) it is
// ignored and the engine always selects CPU. Appended as a DEFAULTED trailing
// param so existing callers compile unchanged.
sf_status load_model(const std::string &path, const HardwareProfile &hw,
                     uint64_t vram_budget, uint64_t ram_budget,
                     std::unique_ptr<Model> &out, PlacementPlan *out_plan,
                     uint64_t cache_budget = 0,
                     RamMode mode = RamMode::Auto,
                     uint64_t target_ram_budget = 0,
                     int preferred_backend = 0);

// Phase 1b/2 llama.cpp backend. Loads a GGUF file via the llama.cpp C API and
// implements the Model interface with real tokenize/detokenize and a real
// greedy single-token forward pass. Defined in model/ggml_model.cpp.
//
// Phase 2: reads the model's shape cheaply from GGUF metadata, builds a
// PlacementPlan from `hw` + budgets, and applies it to the llama load params
// (n_gpu_layers + an expert->CPU tensor override) BEFORE loading the tensors.
// The applied plan is written to `out_plan` when non-null.
//
// Returns SF_OK and sets `out` on success. On any failure (file missing,
// unreadable GGUF, context allocation failure) it returns an error status and
// leaves `out` untouched -- load_model() then substitutes the dry-run model.
sf_status load_ggml_model(const std::string &path, const HardwareProfile &hw,
                          uint64_t vram_budget, uint64_t ram_budget,
                          std::unique_ptr<Model> &out, PlacementPlan *out_plan,
                          uint64_t cache_budget = 0,
                          RamMode mode = RamMode::Auto,
                          uint64_t target_ram_budget = 0,
                          int preferred_backend = 0);

} // namespace sf

// Placement planner: decides which weight units live in which tier.
// See docs/PLAN.md section 3.2.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include "hw/profiler.h"

#include <cstdint>
#include <string>

namespace sf {

// Minimal model shape the planner needs. Populated by the model loader; for
// Phase 1 the dry-run model fills in representative values.
struct ModelShape {
    std::string name;
    uint32_t    n_layers      = 0;
    bool        is_moe        = false;
    uint32_t    n_experts     = 0;   // total experts per MoE layer
    uint32_t    n_experts_used = 0;  // experts activated per token (top-k)
    uint64_t    trunk_bytes   = 0;   // always-active weights
    uint64_t    expert_bytes  = 0;   // one expert's weights (per layer)
    uint64_t    total_bytes   = 0;

    // KV-cache sizing inputs (FEAT-002). These let the planner estimate the
    // engine's REAL resident KV footprint so the expert cache is sized against
    // a budget that already subtracts the KV cost. They mirror the engine's
    // EngineHParams (n_ctx_orig / head_dim / n_head_kv). 0 means "unknown": the
    // planner then falls back to a conservative KV estimate that never
    // UNDER-counts (see plan_placement). Populated by the model loader on both
    // the plain-GGUF and .strata paths.
    uint32_t    n_ctx         = 0;   // trained context length (llama.context_length)
    uint32_t    head_dim      = 0;   // n_embd / n_head
    uint32_t    n_head_kv     = 0;   // KV heads (llama.attention.head_count_kv)
};

struct PlacementPlan {
    // How much of the always-active trunk is pinned, and where.
    uint32_t trunk_layers_in_vram = 0;
    uint32_t trunk_layers_in_ram  = 0;

    // Expert cache sizes (slots) per tier. SSD is the backing store (unbounded).
    uint32_t expert_slots_vram = 0;
    uint32_t expert_slots_ram  = 0;

    // The TOTAL expert BUNDLE slot count the engine will ACTUALLY make resident
    // across ALL layers (one bundle = gate+up+down for one (layer,expert)). This
    // is the number the engine's SlotPool is sized to and the number the plan
    // summary reports; it is derived model-adaptively from the shape + budget
    // (see plan_placement). 0 only for a non-MoE/dense model (no experts).
    uint32_t expert_slots_resident = 0;

    bool     stream_experts    = false;  // true when experts don't all fit
    uint64_t planned_peak_bytes = 0;

    // FEAT-002 conservative-budget fields.
    //
    // kv_window is the CHOSEN resident KV window (ring column count W) the plan
    // recommends. 0 means "full n_ctx" (the engine default, byte-identical, no
    // paging). A non-zero value < n_ctx is a LOSSLESS auto window picked when
    // the full-n_ctx KV would blow the budget: cold positions page to the RAM
    // cold store and are re-assembled byte-identically. sf_context_create
    // plumbs this to the engine via STRATAFLOW_ENGINE_KV_WINDOW (an explicit
    // user value still wins).
    uint32_t    kv_window = 0;

    // fits is false when even the MINIMAL configuration (one layer's top-k
    // resident + the smallest safe KV window + the trunk) still projects a peak
    // over the usable-RAM budget. When false, refusal_reason carries a clear
    // ASCII message naming the shortfall and the plan is NOT safe to run; the C
    // ABI turns this into an error (SF_ERR_RUNTIME) and the CLI prints the
    // reason and exits non-zero, instead of handing back an over-committing
    // plan that the OS then SIGKILLs.
    bool        fits = true;
    std::string refusal_reason;

    // Honest budget breakdown behind planned_peak_bytes, so to_summary() and
    // the tests can read exactly what the conservative estimate charged. All in
    // bytes. usable_ram_bytes is the conservative RAM ceiling the fit invariant
    // holds against (planned_peak_bytes <= usable_ram_bytes for any non-refused
    // plan).
    uint64_t    usable_ram_bytes = 0;
    uint64_t    kv_resident_bytes = 0;
    uint64_t    staging_bytes    = 0;   // one layer at FULL n_expert
    uint64_t    compute_bytes    = 0;   // conservative ggml compute estimate
    uint64_t    process_floor_bytes = 0;
    uint64_t    safety_margin_bytes = 0;

    std::string to_summary() const;
};

// Build a plan from the hardware profile, model shape, and budgets (0 = auto).
//
// `cache_budget` is an optional user-facing cap (in bytes) on the RAM used for
// resident expert bundles (the --cache-gb dial). 0 = unused. When > 0 it caps
// the RAM-for-experts before the slot count is derived, so a user can force a
// bounded, smaller resident expert footprint than free RAM would otherwise
// allow. It never raises the choice above what free RAM / the working set
// support; it only lowers it. See plan_placement for the exact precedence.
PlacementPlan plan_placement(const HardwareProfile &hw,
                             const ModelShape &model,
                             uint64_t vram_budget,
                             uint64_t ram_budget,
                             uint64_t cache_budget = 0);

} // namespace sf

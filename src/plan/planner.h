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
// Target-RAM mode selector (task-target-ram-modes). This chooses HOW the
// planner derives its usable-RAM ceiling when the caller does not pass an
// explicit numeric budget (ram_budget / target_ram_budget both 0):
//
//   Auto (the DEFAULT) - good-neighbor conservative share of RAM. The planner
//     leaves headroom for other apps on a shared/laptop machine: it caps at a
//     fraction of physical RAM and shaves a free-RAM headroom (see
//     usable_ram), then pulls resident footprint down (lossless KV window
//     step-down + adaptive per-layer trunk streaming) when the model would
//     otherwise dominate RAM. This is the sensible default for a machine that
//     also runs other work.
//
//   Max - dedicate the machine to the model: use ALL available RAM (plus the
//     full VRAM budget) WITHOUT the conservative free-headroom / fraction
//     shave, so the plan holds the full KV window (kv_window stays 0 = full
//     n_ctx when it fits), the maximal resident expert cache, and the trunk
//     fully resident (trunk_pool_layers 0) whenever all-RAM affords it. This
//     minimizes streaming and maximizes tok/s at the cost of leaving no
//     headroom for other apps.
//
// An explicit target_ram_budget (bytes, see plan_placement) OVERRIDES the mode:
// it is used directly as the exact usable ceiling regardless of Auto/Max.
enum class RamMode { Auto, Max };

struct ModelShape {
    std::string name;
    uint32_t    n_layers      = 0;
    bool        is_moe        = false;
    uint32_t    n_experts     = 0;   // total experts per MoE layer
    uint32_t    n_experts_used = 0;  // experts activated per token (top-k)
    uint64_t    trunk_bytes   = 0;   // always-active weights
    uint64_t    expert_bytes  = 0;   // one expert's weights (per layer)
    uint64_t    total_bytes   = 0;

    // FEAT-003: total bytes of the STREAMABLE per-layer attention projections
    // (blk.N.attn_q/k/v/output.weight summed across all layers). This is the
    // part of trunk_bytes the engine can stream with a bounded hot pool
    // (technique 6.4); the remaining trunk (token_embd/output edge tensors,
    // per-layer norms, router) stays resident or is the separately-streamed
    // edge pair. 0 means "unknown / nothing streamable" and the planner keeps
    // the whole trunk resident. Populated by the model loader on the .strata
    // path (summed from the v2 trunk index) and left 0 on paths where the
    // breakdown is not available (the planner then never streams the trunk).
    uint64_t    attn_trunk_bytes = 0;

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

    // FEAT-003 (technique 6.4, adaptive per-layer trunk streaming). After the
    // KV window the dominant resident cost on a tight box is the per-layer
    // attention trunk (q/k/v/output projections across every layer). When the
    // conservative budget cannot hold the full trunk resident alongside the KV
    // window + experts + staging + compute + floor + margin, the plan streams
    // the per-layer attention projections and keeps only a BOUNDED HOT set
    // resident. trunk_pool_layers is the number of layers' worth of
    // projections the engine keeps resident at once (the hot-trunk ring depth):
    //   0  = do NOT stream the per-layer trunk (whole trunk resident; the
    //        big-RAM / small-model default, byte-identical, no per-token trunk
    //        SSD traffic).
    //   >0 = stream the per-layer attention projections with this many layers
    //        hot (as low as 1 on a very tight box). HONEST TRADEOFF: these
    //        weights are touched EVERY token, so streaming them adds per-token
    //        SSD traffic and is slower on a slow box - accepted because the
    //        FIRST goal is to FIT small. The engine applies it via
    //        Engine::load's trunk_pool_layers parameter (plumbed through
    //        GgmlModel), NOT an env side-channel.
    // trunk_resident_bytes is the bounded hot-trunk working set the peak
    // accounts for (per_layer_attn_trunk * trunk_pool_layers) when streaming,
    // else the full attention-trunk bytes.
    uint32_t    trunk_pool_layers = 0;
    uint64_t    trunk_resident_bytes = 0;

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
// support; it only lowers it.
//
// `mode` and `target_ram_budget` select the usable-RAM ceiling the plan is
// sized against. Both are backward-compatible defaulted trailing parameters so
// existing callers compile unchanged (RamMode::Auto, target 0). The usable-RAM
// ceiling is resolved by this PRECEDENCE (highest wins):
//   1. Explicit low-level knobs (STRATAFLOW_ENGINE_KV_WINDOW / --expert-slots /
//      --cache-gb) - handled ABOVE the planner and via cache_budget; not a
//      budget-selection concern here.
//   2. target_ram_budget (> 0) - an EXPLICIT numeric target in bytes. Used
//      directly as the exact usable ceiling (like an explicit ram_budget), and
//      it OVERRIDES the mode. The lossless KV-window step-down and the adaptive
//      per-layer trunk streaming engage automatically when a tight target
//      lowers usable below the full-resident footprint; a target below the
//      achievable floor keeps fits=false with the floor named in
//      refusal_reason.
//   3. ram_budget (> 0) - the existing explicit numeric RAM budget (--ram-gb).
//      Unchanged behavior; used directly as the usable ceiling.
//   4. mode == RamMode::Max - use ALL available RAM (ram_free_bytes when known,
//      else ram_total_bytes, else the unknown-profile fallback) WITHOUT the
//      conservative shave, so the plan holds the full KV window + maximal
//      experts + trunk fully resident when all-RAM affords it.
//   5. mode == RamMode::Auto (the DEFAULT) - the conservative good-neighbor
//      usable_ram() share that leaves headroom for other apps.
// The zero-RAM / unknown-profile path always yields a non-zero usable budget
// (never collapses to 0), in every mode.
PlacementPlan plan_placement(const HardwareProfile &hw,
                             const ModelShape &model,
                             uint64_t vram_budget,
                             uint64_t ram_budget,
                             uint64_t cache_budget = 0,
                             RamMode mode = RamMode::Auto,
                             uint64_t target_ram_budget = 0);

} // namespace sf

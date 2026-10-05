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

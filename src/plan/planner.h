// Placement planner: decides which weight units live in which tier.
// See docs/PLAN.md section 3.2.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
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

    bool     stream_experts    = false;  // true when experts don't all fit
    uint64_t planned_peak_bytes = 0;

    std::string to_summary() const;
};

// Build a plan from the hardware profile, model shape, and budgets (0 = auto).
PlacementPlan plan_placement(const HardwareProfile &hw,
                             const ModelShape &model,
                             uint64_t vram_budget,
                             uint64_t ram_budget);

} // namespace sf

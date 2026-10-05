// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "plan/planner.h"

#include <algorithm>
#include <sstream>

namespace sf {
namespace {

constexpr uint64_t kGiB = 1024ull * 1024 * 1024;

// Leave headroom for the OS and the KV cache when auto-sizing RAM.
uint64_t auto_ram_budget(const HardwareProfile &hw) {
    uint64_t avail = hw.ram_free_bytes ? hw.ram_free_bytes : hw.ram_total_bytes;
    uint64_t reserve = std::max<uint64_t>(2 * kGiB, avail / 8);  // >=2 GiB or 12.5%
    return avail > reserve ? avail - reserve : avail / 2;
}

} // namespace

PlacementPlan plan_placement(const HardwareProfile &hw,
                             const ModelShape &model,
                             uint64_t vram_budget,
                             uint64_t ram_budget,
                             uint64_t cache_budget) {
    PlacementPlan plan;

    uint64_t vram = vram_budget;
    if (vram == 0 && !hw.gpus.empty()) {
        vram = hw.gpus.front().vram_bytes;
    }
    uint64_t ram = ram_budget ? ram_budget : auto_ram_budget(hw);

    const uint64_t per_layer_trunk =
        model.n_layers ? model.trunk_bytes / model.n_layers : 0;

    // Rule 1: always-active trunk goes on the fastest device first (VRAM, then
    // RAM). This is the KTransformers / llama.cpp-offload insight.
    uint64_t vram_left = vram;
    if (per_layer_trunk > 0) {
        while (plan.trunk_layers_in_vram < model.n_layers &&
               vram_left >= per_layer_trunk) {
            vram_left -= per_layer_trunk;
            ++plan.trunk_layers_in_vram;
        }
        uint64_t ram_left = ram;
        while (plan.trunk_layers_in_vram + plan.trunk_layers_in_ram < model.n_layers &&
               ram_left >= per_layer_trunk) {
            ram_left -= per_layer_trunk;
            ++plan.trunk_layers_in_ram;
        }
        ram = ram_left;
    }

    const uint64_t trunk_bytes =
        uint64_t(plan.trunk_layers_in_vram + plan.trunk_layers_in_ram) * per_layer_trunk;

    // Rule 2: routed experts fill the remaining VRAM, then RAM, as a cache.
    //
    // RAM-AWARE, MODEL-ADAPTIVE EXPERT-RESIDENCY POLICY
    // -------------------------------------------------
    // The engine makes resident a POOL of expert BUNDLES keyed by (layer,expert)
    // across ALL layers (one bundle = gate+up+down for one expert in one layer).
    // We choose expert_slots_resident = how many bundles the engine will hold.
    // Everything derives from the model SHAPE (n_layer, n_expert, n_expert_used,
    // expert_bytes) and the memory BUDGET - no model-specific constants, so it
    // adapts to any layout (8x2, 256x8, ...).
    //
    //   full_slots      = n_layer * n_expert           (the whole working set)
    //   per_bundle      = model.expert_bytes           (one expert, one layer)
    //   ram_for_experts = RAM left after the trunk, optionally capped by the
    //                     user's --cache-gb (cache_budget) when provided.
    //   affordable      = ram_for_experts / per_bundle
    //
    //   expert_slots_resident = clamp(affordable,
    //                                 lower = max(n_expert_used, n_expert when a
    //                                             full single layer fits),
    //                                 upper = full_slots)
    //
    //   When affordable >= full_slots the WHOLE working set is held: NO per-layer
    //   eviction / re-streaming (the Mixtral-on-Colab win). Only when RAM
    //   genuinely cannot hold it do we bound below full_slots, and NEVER below
    //   n_expert_used (a single layer's top-k must always fit).
    //
    // stream_experts is TRUE iff we bounded below full_slots (eviction happens),
    // FALSE when the whole working set is resident. This is what the engine
    // actually does and what the summary reports (no divergence).
    if (model.is_moe && model.expert_bytes > 0) {
        plan.expert_slots_vram =
            static_cast<uint32_t>(vram_left / model.expert_bytes);
        plan.expert_slots_ram =
            static_cast<uint32_t>(ram / model.expert_bytes);

        const uint64_t full_slots =
            uint64_t(model.n_layers) * uint64_t(model.n_experts);
        const uint64_t per_bundle = model.expert_bytes;

        // RAM available to hold resident expert bundles: VRAM cache + the RAM
        // left after the trunk. An explicit --cache-gb cap only ever LOWERS it.
        uint64_t ram_for_experts = vram_left + ram;
        if (cache_budget > 0 && cache_budget < ram_for_experts) {
            ram_for_experts = cache_budget;
        }

        uint64_t affordable = per_bundle > 0 ? ram_for_experts / per_bundle : 0;

        // Lower bound: a single layer's top-k must always fit (n_expert_used).
        // When RAM comfortably holds a full single layer's experts, keep at
        // least one whole layer resident (n_expert) so no within-layer reload is
        // ever forced; this stays model-adaptive (derived from the shape).
        uint64_t lower = model.n_experts_used;
        if (lower == 0) lower = 1;
        const uint64_t one_layer_slots = model.n_experts;
        if (affordable >= one_layer_slots && one_layer_slots > lower) {
            lower = one_layer_slots;
        }

        uint64_t slots = affordable;
        if (slots > full_slots) slots = full_slots;   // never exceed working set
        if (slots < lower)      slots = lower;         // top-k must fit

        plan.expert_slots_resident = static_cast<uint32_t>(slots);
        // Stream (evict/reload) only when we hold LESS than the whole working
        // set. When slots == full_slots the engine holds everything resident.
        plan.stream_experts = slots < full_slots;

        // Peak estimate consistent with what the engine actually holds: trunk +
        // the resident expert bundles + a one-layer staging footprint (the
        // engine stages one layer's top-k into reusable staging tensors). This
        // reflects reality, NOT the whole on-disk expert set.
        const uint64_t resident_experts = slots * per_bundle;
        const uint64_t staging = uint64_t(model.n_experts_used) * per_bundle;
        plan.planned_peak_bytes = trunk_bytes + resident_experts + staging;
    } else {
        // Dense model: no experts to place.
        plan.expert_slots_resident = 0;
        plan.stream_experts = false;
        plan.planned_peak_bytes = trunk_bytes;
    }

    // NOTE: we intentionally do NOT log the plan summary here. plan_placement
    // is a pure function called several times on a single load (a cheap GGUF
    // metadata pre-pass plus the authoritative post-load plan), and on an
    // explicit --expert-slots run the reconciliation to the forced resident
    // count happens later in sf_context_create. Logging the raw auto value here
    // would both duplicate the line and print a stale "fully resident N" for a
    // run that actually holds fewer and streams -- the misleading-diagnostic
    // divergence this feature eliminates. The authoritative summary is logged
    // once by sf_context_create AFTER reconciliation (and the CLI prints it via
    // sf_describe_plan), so the logged line always matches what the engine runs.
    return plan;
}

std::string PlacementPlan::to_summary() const {
    std::ostringstream os;
    os << "trunk[vram=" << trunk_layers_in_vram
       << " ram=" << trunk_layers_in_ram << " layers]";
    if (expert_slots_resident > 0 || stream_experts) {
        os << ", experts[resident=" << expert_slots_resident << " bundles], "
           << (stream_experts ? "experts STREAMED (bounded)"
                              : "experts fully resident");
    } else {
        os << ", dense (no experts)";
    }
    // Report the peak at the largest unit that does not floor to 0, so small
    // fixtures read truthfully instead of showing "~0 GiB".
    const uint64_t kMiB = 1024ull * 1024;
    if (planned_peak_bytes >= kGiB) {
        os << ", est. peak ~" << (planned_peak_bytes / kGiB) << " GiB";
    } else if (planned_peak_bytes >= kMiB) {
        os << ", est. peak ~" << (planned_peak_bytes / kMiB) << " MiB";
    } else {
        os << ", est. peak ~" << (planned_peak_bytes / 1024ull) << " KiB";
    }
    return os.str();
}

} // namespace sf

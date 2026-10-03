// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "plan/planner.h"

#include "common/log.h"

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
                             uint64_t trunk_bytes,
                             uint64_t cache_bytes) {
    PlacementPlan plan;

    uint64_t vram = vram_budget;
    if (vram == 0 && !hw.gpus.empty()) {
        vram = hw.gpus.front().vram_bytes;
    }
    uint64_t ram = ram_budget ? ram_budget : auto_ram_budget(hw);

    const uint64_t per_layer_trunk =
        model.n_layers ? model.trunk_bytes / model.n_layers : 0;

    // Task 5 dial: an explicit trunk_bytes cap limits the RAM available to the
    // resident trunk (the VRAM tier still fills first; this bounds the RAM
    // spill). 0 = auto (no cap). Applied before the trunk-fill loop below.
    if (trunk_bytes != 0 && trunk_bytes < ram) {
        ram = trunk_bytes;
    }

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

    // Rule 2: routed experts fill the remaining VRAM, then RAM, as a cache.
    if (model.is_moe && model.expert_bytes > 0) {
        plan.expert_slots_vram =
            static_cast<uint32_t>(vram_left / model.expert_bytes);
        plan.expert_slots_ram =
            static_cast<uint32_t>(ram / model.expert_bytes);

        // Task 5 dial: an explicit cache_bytes cap OVERRIDES the auto-derived
        // RAM expert-cache size. 0 = auto (keep the budget-derived value).
        if (cache_bytes != 0) {
            plan.expert_slots_ram =
                static_cast<uint32_t>(cache_bytes / model.expert_bytes);
        }

        const uint64_t total_experts =
            uint64_t(model.n_experts) * model.n_layers;
        const uint64_t cacheable =
            uint64_t(plan.expert_slots_vram) + plan.expert_slots_ram;
        plan.stream_experts = cacheable < total_experts;
    }

    // Peak estimate: pinned trunk + both expert caches.
    plan.planned_peak_bytes =
        uint64_t(plan.trunk_layers_in_vram + plan.trunk_layers_in_ram) * per_layer_trunk +
        uint64_t(plan.expert_slots_vram + plan.expert_slots_ram) * model.expert_bytes;

    log_info("placement plan: " + plan.to_summary());
    return plan;
}

std::string PlacementPlan::to_summary() const {
    std::ostringstream os;
    os << "trunk[vram=" << trunk_layers_in_vram
       << " ram=" << trunk_layers_in_ram << " layers], "
       << "experts[vram=" << expert_slots_vram
       << " ram=" << expert_slots_ram << " slots], "
       << (stream_experts ? "streaming from SSD" : "fully resident")
       << ", peak ~" << (planned_peak_bytes / kGiB) << " GiB";
    return os.str();
}

} // namespace sf

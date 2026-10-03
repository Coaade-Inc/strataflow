// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "plan/planner.h"
#include "test_util.h"

using namespace sf;

static constexpr uint64_t GiB = 1024ull * 1024 * 1024;
static constexpr uint64_t MiB = 1024ull * 1024;

static ModelShape kimi_like() {
    ModelShape m;
    m.name = "test-moe";
    m.n_layers = 60;
    m.is_moe = true;
    m.n_experts = 256;
    m.n_experts_used = 8;
    m.trunk_bytes = 2 * GiB;          // ~34 MiB/layer
    m.expert_bytes = 24 * MiB;
    m.total_bytes = m.trunk_bytes +
        uint64_t(m.n_experts) * m.n_layers * m.expert_bytes;
    return m;
}

// A huge MoE on a tiny machine must choose to stream experts, not try to fit.
static void test_low_ram_streams() {
    HardwareProfile hw;
    hw.ram_total_bytes = 16 * GiB;
    hw.ram_free_bytes  = 14 * GiB;

    PlacementPlan plan = plan_placement(hw, kimi_like(), /*vram=*/0, /*ram=*/0);
    CHECK(plan.stream_experts);                 // experts can't all fit
    CHECK(plan.expert_slots_vram == 0);         // no GPU
    CHECK(plan.planned_peak_bytes <= hw.ram_free_bytes);  // bounded RSS
}

// With a GPU, some trunk layers should be pinned in VRAM first.
static void test_gpu_pins_trunk() {
    HardwareProfile hw;
    hw.ram_total_bytes = 64 * GiB;
    hw.ram_free_bytes  = 60 * GiB;
    GpuInfo g; g.name = "test-gpu"; g.vram_bytes = 24 * GiB; hw.gpus.push_back(g);

    PlacementPlan plan = plan_placement(hw, kimi_like(), /*vram=*/0, /*ram=*/0);
    CHECK(plan.trunk_layers_in_vram > 0);
    CHECK(plan.expert_slots_vram > 0);          // leftover VRAM caches experts
}

static void run_all() {
    RUN(test_low_ram_streams);
    RUN(test_gpu_pins_trunk);
}

TEST_MAIN()

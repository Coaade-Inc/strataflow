// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
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

// A tiny MoE whose whole expert working set fits comfortably in RAM must be
// held FULLY resident: expert_slots_resident == n_layer*n_expert (full_slots)
// and stream_experts == false (no per-layer eviction/re-streaming).
static ModelShape tiny_moe() {
    ModelShape m;
    m.name = "tiny-moe";
    m.n_layers = 2;
    m.is_moe = true;
    m.n_experts = 8;
    m.n_experts_used = 2;
    m.trunk_bytes = 4 * MiB;
    m.expert_bytes = 2 * MiB;        // 2*8*2MiB = 32 MiB whole working set
    m.total_bytes = m.trunk_bytes +
        uint64_t(m.n_experts) * m.n_layers * m.expert_bytes;
    return m;
}

static void test_small_moe_fully_resident() {
    HardwareProfile hw;
    hw.ram_total_bytes = 16 * GiB;
    hw.ram_free_bytes  = 14 * GiB;   // vastly more than the 32 MiB working set

    ModelShape m = tiny_moe();
    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(m.n_layers) * m.n_experts);  // 16

    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);
    CHECK_EQ(plan.expert_slots_resident, full_slots);  // whole working set
    CHECK(!plan.stream_experts);                        // no eviction
}

// The SAME shape under a forced tiny RAM budget cannot hold the working set, so
// the policy bounds the choice strictly below full_slots but never below
// n_expert_used, and marks experts as streamed. planned_peak_bytes stays within
// the budget.
static void test_small_moe_bounded_under_tiny_budget() {
    HardwareProfile hw;
    hw.ram_total_bytes = 16 * GiB;
    hw.ram_free_bytes  = 14 * GiB;

    ModelShape m = tiny_moe();
    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(m.n_layers) * m.n_experts);  // 16

    // Force a tiny expert-RAM cap (just a few bundles' worth) via --cache-gb.
    const uint64_t cache_budget = 5 * m.expert_bytes;  // room for ~5 bundles
    PlacementPlan plan =
        plan_placement(hw, m, /*vram=*/0, /*ram=*/0, cache_budget);

    CHECK(plan.expert_slots_resident < full_slots);        // bounded below full
    CHECK(plan.expert_slots_resident >= m.n_experts_used); // top-k must fit
    CHECK(plan.stream_experts);                            // eviction happens
    // The bounded peak must fit the budget: trunk + resident experts + staging.
    CHECK(plan.planned_peak_bytes <=
          m.trunk_bytes + cache_budget +
              uint64_t(m.n_experts_used) * m.expert_bytes);
}

// A DIFFERENT expert layout (kimi_like 256x8) under a realistic budget adapts
// differently from the 8-expert shape: the policy derives the slot count from
// THIS shape's n_experts/n_layer/expert_bytes, not a fixed assumption. The
// choice is bounded (the ~360 GiB working set cannot fit 14 GiB) yet >=
// n_experts_used, and the peak respects the free-RAM budget.
static void test_different_layout_adapts() {
    HardwareProfile hw;
    hw.ram_total_bytes = 16 * GiB;
    hw.ram_free_bytes  = 14 * GiB;

    ModelShape big = kimi_like();   // 256 experts, 60 layers, top-8
    ModelShape small = tiny_moe();  // 8 experts, 2 layers, top-2

    PlacementPlan pbig = plan_placement(hw, big, /*vram=*/0, /*ram=*/0);
    PlacementPlan psmall = plan_placement(hw, small, /*vram=*/0, /*ram=*/0);

    const uint32_t big_full =
        static_cast<uint32_t>(uint64_t(big.n_layers) * big.n_experts);
    const uint32_t small_full =
        static_cast<uint32_t>(uint64_t(small.n_layers) * small.n_experts);

    // The 256x8 model cannot fit -> bounded and streamed; the 8x2 model fits ->
    // fully resident. The two layouts yield DIFFERENT decisions from the SAME
    // budget, proving model-adaptivity (not an 8x2 assumption).
    CHECK(pbig.stream_experts);
    CHECK(pbig.expert_slots_resident < big_full);
    CHECK(pbig.expert_slots_resident >= big.n_experts_used);
    CHECK(!psmall.stream_experts);
    CHECK_EQ(psmall.expert_slots_resident, small_full);
    CHECK(pbig.expert_slots_resident != psmall.expert_slots_resident);
    CHECK(pbig.planned_peak_bytes <= hw.ram_free_bytes);
}

// Precedence: an explicit cache_budget caps the slot count BELOW the free-RAM
// -derived auto choice. For a model that would otherwise be fully resident, a
// tight --cache-gb forces a bounded, smaller resident pool.
static void test_cache_budget_caps_choice() {
    HardwareProfile hw;
    hw.ram_total_bytes = 16 * GiB;
    hw.ram_free_bytes  = 14 * GiB;

    ModelShape m = tiny_moe();
    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(m.n_layers) * m.n_experts);

    // Auto (no cap) holds the whole working set.
    PlacementPlan auto_plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);
    CHECK_EQ(auto_plan.expert_slots_resident, full_slots);

    // A cap of ~4 bundles must lower the choice below the auto (full) count.
    const uint64_t cache_budget = 4 * m.expert_bytes;
    PlacementPlan capped =
        plan_placement(hw, m, /*vram=*/0, /*ram=*/0, cache_budget);
    CHECK(capped.expert_slots_resident < auto_plan.expert_slots_resident);
    CHECK(capped.expert_slots_resident >= m.n_experts_used);
    CHECK(capped.planned_peak_bytes <=
          m.trunk_bytes + cache_budget +
              uint64_t(m.n_experts_used) * m.expert_bytes);
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
    RUN(test_small_moe_fully_resident);
    RUN(test_small_moe_bounded_under_tiny_budget);
    RUN(test_different_layout_adapts);
    RUN(test_cache_budget_caps_choice);
    RUN(test_gpu_pins_trunk);
}

TEST_MAIN()

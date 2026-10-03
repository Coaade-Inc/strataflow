// Tests for the pure plan->llama-params translation (Phase 2).
//
// derive_llama_placement() turns an abstract sf::PlacementPlan into the
// concrete knobs llama.cpp needs: n_gpu_layers and whether the routed MoE
// expert FFN tensors must be forced onto the CPU (the --n-cpu-moe equivalent).
// It is a pure function over plain structs, so these cases run with NO GPU and
// NO real model -- exactly the paths CI cannot exercise on CPU-only runners.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "plan/llama_placement.h"
#include "plan/planner.h"
#include "hw/profiler.h"
#include "test_util.h"

#include <string>
#include <vector>

using namespace sf;

static constexpr uint64_t GiB = 1024ull * 1024 * 1024;
static constexpr uint64_t MiB = 1024ull * 1024;

static ModelShape moe_shape() {
    ModelShape m;
    m.name = "test-moe";
    m.n_layers = 60;
    m.is_moe = true;
    m.n_experts = 256;
    m.n_experts_used = 8;
    m.trunk_bytes = 2 * GiB;
    m.expert_bytes = 24 * MiB;
    m.total_bytes =
        m.trunk_bytes + uint64_t(m.n_experts) * m.n_layers * m.expert_bytes;
    return m;
}

static ModelShape dense_shape() {
    ModelShape m;
    m.name = "test-dense";
    m.n_layers = 32;
    m.is_moe = false;
    m.trunk_bytes = 8 * GiB;
    m.total_bytes = m.trunk_bytes;
    return m;
}

static GpuInfo gpu(uint64_t vram) {
    GpuInfo g;
    g.name = "test-gpu";
    g.vram_bytes = vram;
    g.backend = 1;  // SF_BACKEND_CUDA
    return g;
}

// (a) No GPU => n_gpu_layers 0, no expert override. Identical to a plain CPU
// load, so behaviour matches Phase 1b exactly.
static void test_no_gpu_is_cpu_only() {
    HardwareProfile hw;
    hw.ram_total_bytes = 16 * GiB;
    hw.ram_free_bytes = 14 * GiB;

    ModelShape m = moe_shape();
    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);

    LlamaPlacement p = derive_llama_placement(plan, m, hw.gpus);
    CHECK_EQ(p.n_gpu_layers, 0);
    CHECK(!p.offload_experts_to_cpu);
    CHECK(p.expert_override_pattern.empty());
}

// (b) Small GPU + big MoE => some trunk layers pinned in VRAM, and because the
// experts cannot all stay resident they are overridden to the CPU buffer.
static void test_small_gpu_big_moe_offloads_experts() {
    HardwareProfile hw;
    hw.ram_total_bytes = 64 * GiB;
    hw.ram_free_bytes = 60 * GiB;
    hw.gpus.push_back(gpu(12 * GiB));  // far too small for ~360 GiB of experts

    ModelShape m = moe_shape();
    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);

    LlamaPlacement p = derive_llama_placement(plan, m, hw.gpus);
    CHECK(p.n_gpu_layers > 0);                 // trunk layers go to VRAM
    CHECK(p.offload_experts_to_cpu);           // experts can't all fit
    CHECK(p.expert_override_pattern == std::string(expert_ffn_regex()));
    CHECK(!p.expert_override_pattern.empty());
}

// (c) GPU big enough to hold the whole model => no streaming and every expert
// is resident, so no expert->CPU override is produced.
static void test_huge_gpu_keeps_experts_resident() {
    HardwareProfile hw;
    hw.ram_total_bytes = 128 * GiB;
    hw.ram_free_bytes = 120 * GiB;

    ModelShape m = moe_shape();
    // VRAM comfortably larger than the whole model (trunk + all experts).
    hw.gpus.push_back(gpu(m.total_bytes + 32 * GiB));

    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);
    CHECK(!plan.stream_experts);               // precondition for this case

    LlamaPlacement p = derive_llama_placement(plan, m, hw.gpus);
    CHECK(p.n_gpu_layers > 0);
    CHECK(!p.offload_experts_to_cpu);          // all experts resident in VRAM
    CHECK(p.expert_override_pattern.empty());
}

// A dense model never gets an expert override regardless of VRAM pressure
// (there are no routed experts to move).
static void test_dense_never_offloads_experts() {
    HardwareProfile hw;
    hw.ram_total_bytes = 32 * GiB;
    hw.ram_free_bytes = 30 * GiB;
    hw.gpus.push_back(gpu(4 * GiB));

    ModelShape m = dense_shape();
    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);

    LlamaPlacement p = derive_llama_placement(plan, m, hw.gpus);
    CHECK(!p.offload_experts_to_cpu);
    CHECK(p.expert_override_pattern.empty());
}

// n_gpu_layers never exceeds the model's layer count even if the plan somehow
// claims more (defensive clamp in the translation).
static void test_ngl_clamped_to_layers() {
    ModelShape m = dense_shape();
    PlacementPlan plan;
    plan.trunk_layers_in_vram = m.n_layers + 100;  // absurd on purpose

    std::vector<GpuInfo> gpus{gpu(80 * GiB)};
    LlamaPlacement p = derive_llama_placement(plan, m, gpus);
    CHECK_EQ(p.n_gpu_layers, static_cast<int32_t>(m.n_layers));
}

static void run_all() {
    RUN(test_no_gpu_is_cpu_only);
    RUN(test_small_gpu_big_moe_offloads_experts);
    RUN(test_huge_gpu_keeps_experts_resident);
    RUN(test_dense_never_offloads_experts);
    RUN(test_ngl_clamped_to_layers);
}

TEST_MAIN()

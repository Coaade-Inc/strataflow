// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "plan/planner.h"
#include "test_util.h"

#include <cstdio>

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
    // The RESIDENT expert footprint must respect the --cache-gb cap. (The
    // honest planned_peak_bytes also includes KV + staging + compute + floor +
    // margin, so it is intentionally larger than the resident-expert bytes;
    // the cap bounds the EXPERT pool, which is what this test guards.)
    CHECK(uint64_t(plan.expert_slots_resident) * m.expert_bytes <= cache_budget);
    // The honest peak still fits the conservative usable-RAM budget.
    CHECK(plan.fits);
    CHECK(plan.planned_peak_bytes <= plan.usable_ram_bytes);
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
    // The cap bounds the resident expert pool (the honest peak additionally
    // charges KV + staging + compute + floor + margin, so compare against the
    // expert footprint, not the dishonest trunk+cap+topk sum).
    CHECK(uint64_t(capped.expert_slots_resident) * m.expert_bytes <=
          cache_budget);
    CHECK(capped.fits);
    CHECK(capped.planned_peak_bytes <= capped.usable_ram_bytes);
}

// Mixtral-scale auto-residency on a 12 GB budget. A ~19 GB Mixtral Q3_K_M
// (trunk ~5.5 GiB, each expert bundle ~52 MiB, 32 layers x 8 experts) cannot
// fit in 10 GiB free RAM, so the planner must choose bounded streaming with a
// slot count > 0 but < full_slots (256), and the peak estimate must stay within
// the budget. This guards the auto policy for the real shape+budget that
// triggered the OOM regression (where the pre-fix code misclassified the model
// as dense and planned 0 expert slots).
static void test_mixtral_scale_12gb_auto() {
    HardwareProfile hw;
    hw.ram_total_bytes = 12 * GiB;
    hw.ram_free_bytes = 10 * GiB;

    ModelShape m;
    m.name = "mixtral-q3km-scale";
    m.n_layers = 32;
    m.is_moe = true;
    m.n_experts = 8;
    m.n_experts_used = 2;
    m.trunk_bytes = static_cast<uint64_t>(5.5 * static_cast<double>(GiB));  // ~5.5 GiB
    m.expert_bytes = 52 * MiB;   // each (layer,expert) bundle ~52 MiB
    m.total_bytes = m.trunk_bytes +
        uint64_t(m.n_experts) * m.n_layers * m.expert_bytes;  // ~18.8 GiB

    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(m.n_layers) * m.n_experts);  // 256

    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);

    // Must stream: the ~13.3 GiB expert working set cannot fit the ~4.5 GiB
    // remaining after the 5.5 GiB trunk.
    CHECK(plan.stream_experts);
    CHECK(plan.expert_slots_resident > 0u);
    CHECK(plan.expert_slots_resident >= m.n_experts_used);
    CHECK(plan.expert_slots_resident < full_slots);
    CHECK(plan.planned_peak_bytes <= hw.ram_free_bytes);

    std::printf("[mixtral scale: slots=%u stream=%s peak=%lluMiB] ",
                plan.expert_slots_resident,
                plan.stream_experts ? "yes" : "no",
                static_cast<unsigned long long>(plan.planned_peak_bytes / MiB));
}

// FEAT-002 conservative-budget REGRESSION: the real Mixtral-8x7B Q3_K_M on the
// 12 GiB Colab box. Pre-fix the planner reserved only a 2 GiB/12.5% headroom,
// sized ~122 resident bundles, and printed "est. peak ~9 GiB" - yet the TRUE
// peak (adding ~8 GiB full-n_ctx KV + staging + compute + floor) blows the box
// and the engine was OOM-killed (SIGKILL 137) before any output. This builds
// the SAME synthetic shape and asserts the honest, conservative planner now (a)
// reports a peak that includes KV+staging+compute+floor+margin and fits usable
// RAM with margin, (b) reduces the slot count far below the old ~122, (c)
// lowers the peak via a lossless auto KV window, and (d) refuses loudly when
// the budget is pathologically small.
static ModelShape mixtral_real() {
    ModelShape m;
    m.name = "mixtral-8x7b-q3km";
    m.n_layers = 32;
    m.is_moe = true;
    m.n_experts = 8;
    m.n_experts_used = 2;
    m.trunk_bytes = static_cast<uint64_t>(5.5 * static_cast<double>(GiB));  // ~5.5 GiB
    m.expert_bytes = 52 * MiB;   // each (layer,expert) bundle ~52 MiB
    m.total_bytes = m.trunk_bytes +
        uint64_t(m.n_experts) * m.n_layers * m.expert_bytes;  // ~18.8 GiB
    // REAL KV geometry (this is what the pre-fix estimate omitted):
    // n_ctx=32768, head_dim=128, n_head_kv=8 => floats_per_col=1024 =>
    // full KV = 32*2*1024*4*32768 = 8 GiB.
    m.n_ctx = 32768;
    m.head_dim = 128;
    m.n_head_kv = 8;
    return m;
}

static void test_mixtral_12gb_conservative_budget() {
    HardwareProfile hw;
    hw.ram_total_bytes = 12 * GiB;
    hw.ram_free_bytes = static_cast<uint64_t>(10.6 * static_cast<double>(GiB));  // optimistic MemAvailable
    // no GPU

    ModelShape m = mixtral_real();
    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(m.n_layers) * m.n_experts);  // 256

    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);

    // (a) The plan must FIT and its honest peak must stay under the usable RAM
    // with margin to spare (strictly under physical RAM too).
    CHECK(plan.fits);
    CHECK(plan.planned_peak_bytes <= plan.usable_ram_bytes);
    CHECK(plan.usable_ram_bytes < hw.ram_total_bytes);
    CHECK(plan.planned_peak_bytes < hw.ram_total_bytes);
    // The honest estimate INCLUDES KV + staging + compute + floor + margin, so
    // it must be CLEARLY larger than the old dishonest value (trunk + resident
    // experts alone). This is the assertion that would FAIL against the pre-fix
    // estimate.
    const uint64_t dishonest_old =
        m.trunk_bytes +
        uint64_t(plan.expert_slots_resident) * m.expert_bytes;
    CHECK(plan.planned_peak_bytes > dishonest_old);
    CHECK(plan.kv_resident_bytes > 0u);
    CHECK(plan.staging_bytes == uint64_t(m.n_experts) * m.expert_bytes);  // full layer
    CHECK(plan.compute_bytes > 0u);
    CHECK(plan.safety_margin_bytes > 0u);

    // (b) The slot count is REDUCED far below the pre-fix ~122 (and stays in
    // the valid band: >= top-k, < full working set).
    CHECK(plan.expert_slots_resident < 122u);
    CHECK(plan.expert_slots_resident >= m.n_experts_used);
    CHECK(plan.expert_slots_resident < full_slots);
    CHECK(plan.stream_experts);

    // (c) A lossless auto KV window was chosen (full n_ctx KV = 8 GiB cannot fit
    // the 12 GiB box alongside the 5.5 GiB trunk), and it LOWERS the peak versus
    // sizing the same plan with full-n_ctx KV.
    CHECK(plan.kv_window > 0u);
    CHECK(plan.kv_window < m.n_ctx);
    // Full-n_ctx KV would be ~8 GiB; the chosen window's KV is strictly less.
    const uint64_t full_kv =
        uint64_t(m.n_layers) * 2 * uint64_t(m.head_dim) * m.n_head_kv *
        sizeof(float) * m.n_ctx;
    CHECK(plan.kv_resident_bytes < full_kv);

    std::printf("[mixtral budget: slots=%u kv_window=%u peak=%lluMiB "
                "usable=%lluMiB] ",
                plan.expert_slots_resident, plan.kv_window,
                static_cast<unsigned long long>(plan.planned_peak_bytes / MiB),
                static_cast<unsigned long long>(plan.usable_ram_bytes / MiB));

    // (d) A pathologically small budget cannot fit even the minimal config:
    // the planner must refuse LOUDLY (fits=false + a non-empty reason), never
    // return an over-committing plan.
    HardwareProfile tiny;
    tiny.ram_total_bytes = 2 * GiB;
    tiny.ram_free_bytes = 2 * GiB;
    PlacementPlan refused = plan_placement(tiny, m, /*vram=*/0, /*ram=*/0);
    CHECK(!refused.fits);
    CHECK(!refused.refusal_reason.empty());
}

// An UNKNOWN hardware profile (both RAM fields 0, e.g. the dry-run/ABI path or
// a profiler that could not read the OS RAM) must NOT collapse the usable-RAM
// budget to 0 and refuse every model. It must fall back to a non-zero budget so
// a normal small model still fits.
static void test_unknown_ram_profile_nonzero_budget() {
    HardwareProfile hw;  // ram_total_bytes == 0 && ram_free_bytes == 0

    ModelShape m;
    m.name = "dry-run-moe";
    m.n_layers = 32;
    m.is_moe = true;
    m.n_experts = 8;
    m.n_experts_used = 2;
    m.trunk_bytes = 256 * MiB;
    m.expert_bytes = 24 * MiB;
    m.total_bytes = m.trunk_bytes + m.n_experts * m.n_layers * m.expert_bytes;

    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);
    CHECK(plan.usable_ram_bytes > 0u);  // unknown RAM != 0-byte budget
    CHECK(plan.fits);                   // not spuriously refused
    CHECK(plan.refusal_reason.empty());
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

// ---------------------------------------------------------------------------
// task-target-ram-modes FEAT-001: RamMode {Auto, Max} + explicit target.
// ---------------------------------------------------------------------------

// A Mixtral-geometry MoE whose per-layer attention projections are a large
// fraction of the trunk, so the FEAT-003 adaptive trunk streaming (gated on
// attn_trunk_bytes > 0) can engage once the budget is tight.
static ModelShape mixtral_streamable() {
    ModelShape m = mixtral_real();              // real KV geometry + ~18.8 GiB
    // Most of the 5.5 GiB trunk is streamable per-layer attention projections.
    m.attn_trunk_bytes = static_cast<uint64_t>(4.0 * static_cast<double>(GiB));
    return m;
}

// Auto vs Max on the SAME machine + model: Max dedicates the machine (all RAM,
// no good-neighbor shave) and must therefore hold MORE resident than Auto -
// a strictly larger usable budget, >= KV window, >= resident experts, and <=
// trunk streaming (fewer streamed bytes), with a strictly higher projected
// peak. Auto's peak must stay within its conservative share.
static void test_auto_vs_max_same_machine() {
    HardwareProfile hw;
    hw.ram_total_bytes = 24 * GiB;
    hw.ram_free_bytes  = 22 * GiB;

    ModelShape m = mixtral_streamable();

    PlacementPlan a = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                     /*cache=*/0, RamMode::Auto);
    PlacementPlan x = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                     /*cache=*/0, RamMode::Max);

    // Max's usable ceiling is strictly larger (no 85%/free-headroom shave).
    CHECK(x.usable_ram_bytes > a.usable_ram_bytes);
    // Auto respects its conservative share; Max uses more and peaks higher.
    CHECK(a.planned_peak_bytes <= a.usable_ram_bytes);
    CHECK(x.planned_peak_bytes > a.planned_peak_bytes);
    // FIT INVARIANT on the (possibly CLAMPED) Max plan: after the monotonicity
    // guard re-fits slots against the raised KV/trunk cost, Max's projected peak
    // must still fit its usable budget. Pins the construction-only guarantee so
    // a re-fit regression cannot slip past the ordering checks above.
    CHECK(x.fits);
    CHECK(x.planned_peak_bytes <= x.usable_ram_bytes);
    // Max holds >= experts resident and >= KV window than Auto.
    CHECK(x.expert_slots_resident >= a.expert_slots_resident);
    // KV window: 0 means full n_ctx. Max keeps a window at least as large as
    // Auto (full if Auto stepped down, equal otherwise).
    if (a.kv_window != 0) {
        CHECK(x.kv_window == 0 || x.kv_window >= a.kv_window);
    }
    // Max streams the trunk no harder than Auto (fewer or equal hot layers
    // means fewer streamed bytes; 0 = trunk fully resident).
    if (x.trunk_pool_layers == 0) {
        CHECK(x.trunk_pool_layers <= a.trunk_pool_layers);  // 0 <= anything
    } else {
        CHECK(a.trunk_pool_layers != 0);
        CHECK(x.trunk_pool_layers >= a.trunk_pool_layers);  // more hot layers
    }

    std::printf("[auto vs max: usable %lluMiB<%lluMiB peak %lluMiB<%lluMiB "
                "slots %u<=%u pool %u<=%u] ",
                static_cast<unsigned long long>(a.usable_ram_bytes / MiB),
                static_cast<unsigned long long>(x.usable_ram_bytes / MiB),
                static_cast<unsigned long long>(a.planned_peak_bytes / MiB),
                static_cast<unsigned long long>(x.planned_peak_bytes / MiB),
                a.expert_slots_resident, x.expert_slots_resident,
                a.trunk_pool_layers, x.trunk_pool_layers);
}

// An explicit tight target drives ALL levers so the projected peak fits the
// target, and because the target is BELOW the full trunk-resident floor it
// ENGAGES trunk streaming (trunk_pool_layers > 0) and a stepped-down KV window.
static void test_explicit_target_drives_levers() {
    HardwareProfile hw;
    hw.ram_total_bytes = 24 * GiB;   // plenty of RAM; the TARGET is the ceiling
    hw.ram_free_bytes  = 22 * GiB;

    ModelShape m = mixtral_streamable();
    const uint64_t target = 3 * GiB;  // well below the trunk-resident floor

    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                        /*cache=*/0, RamMode::Auto,
                                        /*target=*/target);

    // The target is used as the EXACT usable ceiling.
    CHECK_EQ(plan.usable_ram_bytes, target);
    CHECK(plan.fits);
    CHECK(plan.planned_peak_bytes <= target);
    // Trunk streaming engaged: a bounded hot pool, not the full trunk.
    CHECK(plan.trunk_pool_layers > 0u);
    CHECK(plan.trunk_pool_layers < m.n_layers);
    // KV window stepped down (lossless) into (0, n_ctx).
    CHECK(plan.kv_window > 0u);
    CHECK(plan.kv_window < m.n_ctx);
    // Experts streamed too (cannot hold the full working set in 3 GiB).
    CHECK(plan.stream_experts);
    CHECK(plan.expert_slots_resident >= m.n_experts_used);

    std::printf("[explicit target 3GiB: peak=%lluMiB pool=%u kv=%u slots=%u] ",
                static_cast<unsigned long long>(plan.planned_peak_bytes / MiB),
                plan.trunk_pool_layers, plan.kv_window,
                plan.expert_slots_resident);
}

// A pathologically tiny target - below the achievable floor (minimal KV window
// + minimal experts + maximal trunk streaming) - must FAIL LOUD: fits==false
// and a non-empty refusal_reason that names the floor/shortfall.
static void test_target_below_floor_refuses_loudly() {
    HardwareProfile hw;
    hw.ram_total_bytes = 24 * GiB;
    hw.ram_free_bytes  = 22 * GiB;

    ModelShape m = mixtral_streamable();
    const uint64_t target = 256 * MiB;  // below even the process floor + margin

    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                        /*cache=*/0, RamMode::Auto,
                                        /*target=*/target);

    CHECK(!plan.fits);
    CHECK(!plan.refusal_reason.empty());
    // The reason must name the achievable floor so the user can raise the
    // target to a workable value.
    CHECK(plan.refusal_reason.find("floor") != std::string::npos);
    CHECK_EQ(plan.usable_ram_bytes, target);  // target still used as the ceiling
}

// An explicit target OVERRIDES the mode: even with RamMode::Max the usable
// ceiling is the target, not all-available RAM. Compare to a plain Max run
// (no target) on the same machine to prove the target won.
static void test_target_overrides_mode() {
    HardwareProfile hw;
    hw.ram_total_bytes = 24 * GiB;
    hw.ram_free_bytes  = 22 * GiB;

    ModelShape m = mixtral_streamable();
    const uint64_t target = 4 * GiB;

    PlacementPlan with_target = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                               /*cache=*/0, RamMode::Max,
                                               /*target=*/target);
    PlacementPlan pure_max = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                            /*cache=*/0, RamMode::Max);

    // The target wins over Max: usable == target, far below all-available RAM.
    CHECK_EQ(with_target.usable_ram_bytes, target);
    CHECK(with_target.usable_ram_bytes < pure_max.usable_ram_bytes);
    CHECK(with_target.planned_peak_bytes <= target);
}

// ---------------------------------------------------------------------------
// task-fix-target-and-max-mode-semantics FEAT-002: the three bugs the real
// Colab run surfaced. These assert REAL mode semantics (not fixture-blind).
// ---------------------------------------------------------------------------

// BUG 2: Max must size its usable ceiling from ALL PHYSICAL RAM (ram_total),
// NOT the instantaneous free RAM. On a box where free << total (the model, or
// anything else, already consumed RAM) the pre-fix free-based Max resolved a
// TINY ceiling and ended up with FEWER residents + MORE streaming than Auto
// (inverted). Assert Max.usable derives from TOTAL (approximately ram_total),
// Max.usable > Auto.usable, Max holds >= experts resident, >= KV window, and
// streams the trunk no harder than Auto (monotonic).
static void test_max_uses_total_not_free_monotonic() {
    HardwareProfile hw;
    hw.ram_total_bytes = 12 * GiB;
    hw.ram_free_bytes = 3 * GiB;   // free << total: the inversion trigger

    ModelShape m = mixtral_streamable();  // ~18.8 GiB, full KV ~2 GiB geometry
    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(m.n_layers) * m.n_experts);

    PlacementPlan a = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                     /*cache=*/0, RamMode::Auto);
    PlacementPlan x = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                     /*cache=*/0, RamMode::Max);

    // Max's ceiling comes from TOTAL (all physical RAM), not the 3 GiB free.
    // Auto's good-neighbor share is derived from the low free RAM, so it is far
    // below Max. (Pre-fix, Max == 3 GiB free, i.e. BELOW or near Auto.)
    CHECK_EQ(x.usable_ram_bytes, hw.ram_total_bytes);
    CHECK(x.usable_ram_bytes > a.usable_ram_bytes);
    CHECK(x.usable_ram_bytes > hw.ram_free_bytes);  // not the free RAM

    // Monotonicity: Max never holds fewer experts than Auto, never a smaller KV
    // window, never streams the trunk harder.
    CHECK(x.expert_slots_resident >= a.expert_slots_resident);
    const uint32_t a_win = a.kv_window ? a.kv_window : m.n_ctx;
    const uint32_t x_win = x.kv_window ? x.kv_window : m.n_ctx;
    CHECK(x_win >= a_win);
    const uint32_t a_hot = a.trunk_pool_layers ? a.trunk_pool_layers : m.n_layers;
    const uint32_t x_hot = x.trunk_pool_layers ? x.trunk_pool_layers : m.n_layers;
    CHECK(x_hot >= a_hot);
    CHECK(x.expert_slots_resident <= full_slots);

    // FIT INVARIANT on the CLAMPED Max plan: the monotonicity guard re-fits the
    // slot count after raising KV/trunk toward Auto, so the final projected peak
    // must still fit Max's usable budget. The guarantee holds by construction,
    // but a re-fit regression that overflowed usable would otherwise go uncaught
    // (the ordering checks above would stay green).
    CHECK(x.fits);
    CHECK(x.planned_peak_bytes <= x.usable_ram_bytes);

    std::printf("[max vs free<<total: auto usable=%lluMiB slots=%u | max "
                "usable=%lluMiB slots=%u] ",
                static_cast<unsigned long long>(a.usable_ram_bytes / MiB),
                a.expert_slots_resident,
                static_cast<unsigned long long>(x.usable_ram_bytes / MiB),
                x.expert_slots_resident);
}

// BUG 1: a target below the TRUE floor refuses with a refusal_reason whose KV
// figure reflects the SMALLEST window (kMinKvWindow), NOT full n_ctx. The
// pre-fix refusal reported FULL KV (the dishonest ~8-10 GiB "minimal config").
static void test_honest_refusal_floor_smallest_kv() {
    HardwareProfile hw;
    hw.ram_total_bytes = 24 * GiB;
    hw.ram_free_bytes = 22 * GiB;

    ModelShape m = mixtral_streamable();
    // Full n_ctx KV for this geometry = 32*2*1024*4*32768 = 8 GiB = 8192 MiB.
    const uint64_t full_kv =
        uint64_t(m.n_layers) * 2 * uint64_t(m.head_dim) * m.n_head_kv *
        sizeof(float) * m.n_ctx;
    CHECK_EQ(full_kv, 8ull * GiB);

    // A target below even the smallest-window minimal config: refuse honestly.
    const uint64_t target = 512 * MiB;
    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                        /*cache=*/0, RamMode::Auto,
                                        /*target=*/target);
    CHECK(!plan.fits);
    CHECK(!plan.refusal_reason.empty());
    CHECK(plan.refusal_reason.find("floor") != std::string::npos);
    // The honest refusal names the SMALLEST KV window (256), NOT the 8192 MiB
    // full-KV figure.
    CHECK(plan.refusal_reason.find("KV window 256") != std::string::npos);
    CHECK(plan.refusal_reason.find(" KV 8192 MiB") == std::string::npos);

    std::printf("[honest refusal: %s] ", plan.refusal_reason.c_str());
}

// BUG 1: an achievable small target - at/just above the TRUE minimal floor -
// FITS by shrinking the KV window into (0, n_ctx) and engaging trunk streaming,
// instead of the dishonest full-KV refusal.
static void test_achievable_small_target_fits() {
    HardwareProfile hw;
    hw.ram_total_bytes = 24 * GiB;
    hw.ram_free_bytes = 22 * GiB;

    ModelShape m = mixtral_streamable();
    // 3 GiB is comfortably above the true minimal-KV floor yet well below the
    // full trunk-resident + full-KV footprint, so the levers must engage.
    const uint64_t target = 3 * GiB;
    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                        /*cache=*/0, RamMode::Auto,
                                        /*target=*/target);
    CHECK(plan.fits);
    CHECK(plan.planned_peak_bytes <= target);
    CHECK(plan.kv_window > 0u);
    CHECK(plan.kv_window < m.n_ctx);
    CHECK(plan.trunk_pool_layers > 0u);     // trunk streaming engaged
    CHECK(plan.stream_experts);
    CHECK(plan.expert_slots_resident >= m.n_experts_used);
}

// BUG 3: estimate calibration. A windowed plan's planned_peak is strictly below
// the SAME shape's full-n_ctx-KV planned peak, AND within a sane factor
// (< 1.5x) of the engine's real resident components (resident KV + resident
// experts + staging + resident trunk + floor) on the real Mixtral geometry
// where the multi-GiB KV dominates. The pre-fix estimate stacked to ~2x.
static void test_estimate_tracks_real_resident() {
    HardwareProfile hw;
    hw.ram_total_bytes = 12 * GiB;
    hw.ram_free_bytes = static_cast<uint64_t>(10.6 * static_cast<double>(GiB));

    ModelShape m = mixtral_real();   // 5.5 GiB trunk, 8 GiB full KV, 52 MiB/exp
    PlacementPlan plan = plan_placement(hw, m, /*vram=*/0, /*ram=*/0);
    CHECK(plan.fits);

    // Windowed KV charge is strictly below the full-n_ctx reservation.
    const uint64_t full_kv =
        uint64_t(m.n_layers) * 2 * uint64_t(m.head_dim) * m.n_head_kv *
        sizeof(float) * m.n_ctx;
    CHECK(plan.kv_window > 0u);
    CHECK(plan.kv_resident_bytes < full_kv);

    // Same plan sized with full-n_ctx KV would project a strictly higher peak
    // (the only changed term is the KV charge).
    const uint64_t full_kv_peak =
        plan.planned_peak_bytes - plan.kv_resident_bytes + full_kv;
    CHECK(plan.planned_peak_bytes < full_kv_peak);

    // Estimate padding (compute scratch + safety margin) is a MINORITY of the
    // real resident components, so planned_peak < 1.5x reality (not the ~2x the
    // pre-fix conservative stacking produced). At Mixtral scale the multi-GiB
    // resident KV dwarfs the fixed floors.
    const uint64_t padding = plan.compute_bytes + plan.safety_margin_bytes;
    const uint64_t real_components = plan.planned_peak_bytes - padding;
    CHECK(padding < real_components / 2);  // planned_peak < 1.5x real

    std::printf("[estimate: peak=%lluMiB kv=%lluMiB padding=%lluMiB "
                "real=%lluMiB] ",
                static_cast<unsigned long long>(plan.planned_peak_bytes / MiB),
                static_cast<unsigned long long>(plan.kv_resident_bytes / MiB),
                static_cast<unsigned long long>(padding / MiB),
                static_cast<unsigned long long>(real_components / MiB));
}

// ---------------------------------------------------------------------------
// task-auto-retune-and-streaming-limiter FEAT-002: AUTO coherence + fill the
// comfortable share with experts, and MAX fits its own budget.
// ---------------------------------------------------------------------------

// FEAT-002 COHERENCE: on a 12 GiB-class box with comfortably large free RAM,
// AUTO must pick a COHERENT plan - it must NEVER keep the full n_ctx KV
// resident while hard-streaming the per-layer attention trunk; the two resident
// levers co-reduce (if the trunk hard-streams, the KV is windowed). Its
// projected peak must fill a SENSIBLE SHARE of the usable budget (not collapse
// to a small fraction), and AUTO.peak <= MAX.peak with MAX.fits == true (MAX
// must fit its OWN budget - the pre-fix MAX projected peak OVER usable here).
static void test_auto_coherent_share_12gb() {
    HardwareProfile hw;
    hw.ram_total_bytes = 12 * GiB;
    hw.ram_free_bytes = static_cast<uint64_t>(11.0 * static_cast<double>(GiB));

    ModelShape m = mixtral_streamable();  // ~18.8 GiB, 8 GiB full KV, 4 GiB attn

    PlacementPlan a = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                     /*cache=*/0, RamMode::Auto);
    PlacementPlan x = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                     /*cache=*/0, RamMode::Max);

    CHECK(a.fits);
    CHECK(a.planned_peak_bytes <= a.usable_ram_bytes);

    // COHERENCE INVARIANT: if AUTO hard-streams the trunk (0 < pool < n_layers)
    // the KV MUST be windowed (0 < kv_window < n_ctx). The pre-fix AUTO kept
    // full KV (kv_window==0) while streaming the trunk hard - forbidden.
    if (a.trunk_pool_layers > 0u && a.trunk_pool_layers < m.n_layers) {
        CHECK(a.kv_window > 0u);
        CHECK(a.kv_window < m.n_ctx);
    }
    // Equivalently: full KV resident (kv_window==0) IMPLIES the trunk is NOT
    // hard-streaming (pool==0 == whole trunk resident).
    if (a.kv_window == 0u) {
        CHECK(a.trunk_pool_layers == 0u);
    }

    // SENSIBLE SHARE: AUTO fills its comfortable share rather than collapsing
    // to a small fraction of it. The real-Mixtral bug left AUTO at ~1/9th of
    // the budget (peak 1353 MiB on a box whose share was ~10 GiB). Assert the
    // peak uses most of the usable share (>= 70%).
    CHECK(a.planned_peak_bytes * 10 >= a.usable_ram_bytes * 7);

    // MAX fits its OWN budget and stays >= AUTO (monotonic).
    CHECK(x.fits);
    CHECK(x.planned_peak_bytes <= x.usable_ram_bytes);
    CHECK(a.planned_peak_bytes <= x.planned_peak_bytes);
    CHECK(x.expert_slots_resident >= a.expert_slots_resident);
    // MAX coherence too.
    if (x.trunk_pool_layers > 0u && x.trunk_pool_layers < m.n_layers) {
        CHECK(x.kv_window > 0u);
        CHECK(x.kv_window < m.n_ctx);
    }

    std::printf("[auto coherent 12GiB: usable=%lluMiB peak=%lluMiB kv=%u pool=%u "
                "slots=%u | max peak=%lluMiB fits=%d slots=%u] ",
                static_cast<unsigned long long>(a.usable_ram_bytes / MiB),
                static_cast<unsigned long long>(a.planned_peak_bytes / MiB),
                a.kv_window, a.trunk_pool_layers, a.expert_slots_resident,
                static_cast<unsigned long long>(x.planned_peak_bytes / MiB),
                x.fits ? 1 : 0, x.expert_slots_resident);
}

// FEAT-002: after the retune AUTO caches materially MORE experts than the old
// 2 on the Mixtral geometry - at least an order of magnitude more when RAM
// affords it (the comfortable share is filled with resident experts instead of
// left mostly unused).
static void test_auto_caches_more_experts() {
    HardwareProfile hw;
    hw.ram_total_bytes = 12 * GiB;
    hw.ram_free_bytes = static_cast<uint64_t>(11.0 * static_cast<double>(GiB));

    ModelShape m = mixtral_streamable();
    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(m.n_layers) * m.n_experts);  // 256

    PlacementPlan a = plan_placement(hw, m, /*vram=*/0, /*ram=*/0,
                                     /*cache=*/0, RamMode::Auto);
    CHECK(a.fits);
    // Materially more than the old 2: strictly greater, and an order of
    // magnitude more (>= 10) on this 12 GiB-class box.
    CHECK(a.expert_slots_resident > 2u);
    CHECK(a.expert_slots_resident >= 10u);
    // Still bounded by the working set (the ~13 GiB expert set cannot all fit).
    CHECK(a.expert_slots_resident < full_slots);
    CHECK(a.stream_experts);

    std::printf("[auto experts 12GiB: slots=%u (old baseline 2)] ",
                a.expert_slots_resident);
}

static void run_all() {
    RUN(test_low_ram_streams);
    RUN(test_auto_coherent_share_12gb);
    RUN(test_auto_caches_more_experts);
    RUN(test_small_moe_fully_resident);
    RUN(test_small_moe_bounded_under_tiny_budget);
    RUN(test_different_layout_adapts);
    RUN(test_cache_budget_caps_choice);
    RUN(test_mixtral_scale_12gb_auto);
    RUN(test_mixtral_12gb_conservative_budget);
    RUN(test_unknown_ram_profile_nonzero_budget);
    RUN(test_gpu_pins_trunk);
    RUN(test_auto_vs_max_same_machine);
    RUN(test_explicit_target_drives_levers);
    RUN(test_target_below_floor_refuses_loudly);
    RUN(test_target_overrides_mode);
    RUN(test_max_uses_total_not_free_monotonic);
    RUN(test_honest_refusal_floor_smallest_kv);
    RUN(test_achievable_small_target_fits);
    RUN(test_estimate_tracks_real_resident);
}

TEST_MAIN()

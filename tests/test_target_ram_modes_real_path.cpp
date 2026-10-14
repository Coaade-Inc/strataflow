// task-target-ram-modes real-load-path gate: prove the three target-RAM modes
// (explicit numeric target, auto good-neighbor default, max dedicate-machine)
// drive the fit levers as designed on the REAL sf::load_model path AND that the
// greedy decode is BYTE-IDENTICAL across all three (changing residency, never
// which bytes compute or the graph order).
//
// This mirrors test_kv_window_real_path.cpp exactly in structure: it packs the
// Mixtral-geometry legacy fixture to a .strata, loads it through the public
// sf::load_model seam under the FEAT-001 planner with each mode/target, and
// asserts the plan geometry; the matched STACKED GGUF (when set) supplies the
// libllama tokenizer + oracle for the byte-identity half. A .strata is not
// libllama-loadable (and vendored llama.cpp cannot load the legacy per-expert
// GGUF), so the oracle runs on the stacked equivalent - the same contract as
// test_kv_window_real_path / test_legacy_per_expert_pack.
//
// Through sf::load_model on the SAME packed .strata this asserts:
//   (a) EXPLICIT TIGHT TARGET (target_ram_budget set below the fixture's
//       trunk-resident+full-KV floor via the simulated-target stand-in, the
//       same technique test_kv_window_real_path uses with ram_budget): the plan
//       FITS, planned_peak_bytes <= target, kv_window in (0, n_ctx), AND trunk
//       streaming ENGAGES (trunk_pool_layers > 0). The target is dimensioned so
//       streaming MUST engage; the assertion HARD-fails (not skips) otherwise.
//       An engine loaded with the plan's levers reports resident_kv_bytes() <
//       full_kv_bytes().
//   (b) AUTO: planned_peak_bytes <= its conservative usable share, and strictly
//       below MAX's peak (the good-neighbor margin).
//   (c) MAX: larger usable than auto, a KV window at least as large as auto's,
//       resident experts at least as many as auto's, and trunk fully resident
//       (trunk_pool_layers 0) or no more streamed than auto (fewer streamed
//       bytes than auto).
//   (d) HARD byte-identity: the greedy decode is BIT-IDENTICAL (max|delta|==0.0)
//       across the explicit-target, auto, and max loads (same output regardless
//       of mode/target), and matches the libllama oracle on the stacked GGUF
//       within the documented soft quant bound (1e-1 for this random-weight
//       q4_k fixture, identical to test_kv_window_real_path).
//
// The plan/geometry assertions (a)-(c) run from the GGUF alone; the decode +
// oracle half (d) is guarded on STRATAFLOW_TEST_MIXTRAL_GEOM_STACKED_GGUF and
// skips cleanly when that fixture is absent (the geometry assertions already
// stand). Generate + run locally:
//   build/release/bin/make-legacy-moe-gguf FIX.gguf FIX.stacked.gguf
//       1024 256 32 8 q4_k 32768 2 8
//   STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF=FIX.gguf
//     STRATAFLOW_TEST_MIXTRAL_GEOM_STACKED_GGUF=FIX.stacked.gguf
//     ctest -R test_target_ram_modes_real_path --output-on-failure
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "hw/profiler.h"
#include "model/engine/engine.h"
#include "model/model.h"
#include "plan/planner.h"
#include "test_util.h"
#include "tws/strata_pack.h"

using namespace sf;

static constexpr uint64_t GiB = 1024ull * 1024 * 1024;

static int32_t argmax(const std::vector<float> &v) {
    int32_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] > best_v) {
            best_v = v[i];
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

// RAII env override + restore (mirrors ScopedEnv in test_kv_window_real_path).
// Here it CLEARS STRATAFLOW_ENGINE_KV_WINDOW for the whole test so the only
// window that can engage is the one the planner chose (per mode/target), not an
// inherited env value - the plan-driven default is what we are proving.
class ScopedEnv {
public:
    ScopedEnv(const char *name, const char *value) : name_(name) {
        const char *prev = std::getenv(name);
        had_prev_ = prev != nullptr;
        if (had_prev_) prev_ = prev;
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, /*overwrite=*/1);
#endif
    }
    ~ScopedEnv() {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), had_prev_ ? prev_.c_str() : "");
#else
        if (had_prev_) setenv(name_.c_str(), prev_.c_str(), 1);
        else unsetenv(name_.c_str());
#endif
    }
    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
    std::string name_;
    bool had_prev_ = false;
    std::string prev_;
};

struct DecodeRun {
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> step_logits;
    uint64_t resident_kv_bytes = 0;
    uint64_t full_kv_bytes = 0;
    uint64_t streamed_bytes = 0;  // per-token expert bytes streamed from disk
    bool ok = false;
};

// Decode `prompt_ids` + `n_generate` greedy tokens through a freshly-loaded
// engine on `engine_path` with the plan's levers (kv_window + trunk_pool_layers
// + expert slots). No env is read for the window (the caller clears it), so a
// non-full window / streamed trunk here comes solely from the plan the mode
// produced, applied via Engine::load - exactly the default load path
// sf::load_model drives.
static DecodeRun decode_with_plan(const char *engine_path,
                                  const std::vector<int32_t> &prompt_ids,
                                  int n_generate, uint32_t expert_slots,
                                  uint32_t kv_window,
                                  uint32_t trunk_pool_layers) {
    DecodeRun r;
    if (prompt_ids.empty()) return r;
    std::unique_ptr<engine::Engine> eng = engine::Engine::load(
        engine_path, expert_slots, kv_window, trunk_pool_layers);
    if (eng == nullptr) return r;
    eng->reset_kv();

    int32_t pos = 0;
    std::vector<float> logits;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        if (!eng->forward(prompt_ids[i], pos, logits)) return r;
        ++pos;
    }
    for (int g = 0; g < n_generate; ++g) {
        if (logits.empty()) return r;
        r.step_logits.push_back(logits);
        const int32_t next = argmax(logits);
        r.tokens.push_back(next);
        if (!eng->forward(next, pos, logits)) return r;
        ++pos;
    }
    r.resident_kv_bytes = eng->resident_kv_bytes();
    r.full_kv_bytes = eng->full_kv_bytes();
    r.streamed_bytes = eng->streamed_bytes();
    r.ok = true;
    return r;
}

static double max_step_delta(const std::vector<std::vector<float>> &a,
                             const std::vector<std::vector<float>> &b) {
    double m = 0.0;
    const size_t steps = a.size() < b.size() ? a.size() : b.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = a[s].size() < b[s].size() ? a[s].size() : b[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(a[s][i]) -
                                       static_cast<double>(b[s][i]));
            if (d > m) m = d;
        }
    }
    return m;
}

// Explicit tight target used directly as the usable ceiling (target_ram_budget
// OVERRIDES the mode and is used as the exact ceiling, like an explicit
// ram_budget). 700 MiB is tight enough that even a windowed KV plus the FULL
// resident trunk cannot fit, so the planner MUST step the KV window down AND
// hard-stream the per-layer attention trunk to fit - the COHERENT combination
// (FEAT-002): when the trunk hard-streams the KV is ALSO windowed, the two
// levers co-reduce. Verified on the fixture (full-n_ctx KV = 2 GiB at n_ctx=
// 32768, head_count_kv=2; trunk ~46 MiB of which ~45 MiB is streamable attn):
// at 700 MiB the plan reports kv_window=256 and trunk stream[hot=1 layers],
// peak ~700 MiB, slots ~13. Because the target is used verbatim as the ceiling,
// this is machine-independent (the profile's real RAM does not change the
// explicit-target plan).
//
// FEAT-002 RETUNE NOTE: before the coherent retune, the previous value here was
// 1.2 GiB, at which the planner USED to engage trunk streaming. After the
// retune the KV window is budgeted against the FULL resident trunk, so at
// 1.2 GiB the planner now (correctly) keeps the small trunk resident and only
// windows the KV (trunk_pool_layers==0) - a COHERENT plan, but not one that
// exercises the trunk-streaming lever. The 700 MiB target below forces trunk
// streaming to engage AND proves the coherence invariant holds when it does.
static constexpr uint64_t kTightTargetBytes = 734003200ull;  // 700 MiB

// A TIGHT box whose INSTANTANEOUS free RAM is far below its total physical RAM
// (free << total), which is exactly the Colab situation that inverted Max: the
// big model (or anything else) had already consumed RAM, so MemAvailable was
// small. This profile BREAKS THE FIXTURE-BLINDNESS: at this budget the
// fixture's full-n_ctx KV (2 GiB at n_ctx=32768, head_count_kv=2) + trunk +
// experts cannot all stay resident under AUTO's good-neighbor share, so AUTO
// steps the KV window down and/or streams, while MAX - sizing to ALL PHYSICAL
// RAM (BUG 2 fix) - holds full KV + full experts + resident trunk. The two
// modes therefore choose DIFFERENT levers, which is what makes the Max
// inversion and the honest-floor fixes OBSERVABLE on the real load path.
//
// total = 3 GiB (all physical RAM Max dedicates); free = 2.25 GiB (the
// instantaneous MemAvailable Auto's good-neighbor share is derived from). With
// free << total, a free-based Max (the BUG) would resolve a SMALLER ceiling
// than Auto; the fixed total-based Max resolves strictly above it. At this
// budget on the Mixtral-geometry fixture (full-n_ctx KV = 2 GiB) the modes
// choose GENUINELY DIFFERENT levers (fixture-blindness broken): AUTO caps at a
// ~1.25 GiB good-neighbor share so it windows the KV down AND streams some
// experts (slots far below full), while MAX dedicates all 3 GiB so it holds
// FULL KV + FULL experts + resident trunk. That slot/window divergence is what
// makes the Max inversion observable - with the pre-fix tiny-expert fixture (or
// a free-based Max) both modes trivially held full residency.
static HardwareProfile colab_profile() {
    HardwareProfile hw;
    hw.ram_total_bytes = 3 * GiB;
    hw.ram_free_bytes = (2 * GiB) + (256ull * 1024 * 1024);  // 2.25 GiB free
    return hw;
}

// task-max-mode-clamp-not-refuse FEAT-002 (TDD regression): a profile that
// breaks the OTHER fixture-blindness - colab_profile() (3 GiB total) leaves
// MAX's full-n_ctx-KV natural plan (peak ~2885 MiB) UNDER usable (~3072 MiB),
// so the real 12-13 GiB-class Mixtral MAX refusal is never exercised on the
// real load path. This profile is dimensioned (verified empirically on THIS
// fixture) so the full natural MAX plan OVERFLOWS the usable ceiling while a
// smaller plan FITS: at ram_total = 2.8 GiB (free 2.1 GiB) the current
// (pre-fix) planner refuses MAX (natural peak ~2885 MiB vs usable ~2867 MiB,
// floor ~695 MiB) while AUTO fits (peak ~1086 MiB, slots 256, kv_window 4096).
// NOTE: the small fixture's absolute sizes differ from real Mixtral (its full
// KV at head_count_kv=2 is ~2 GiB, not 8 GiB), so the OVERFLOW total is a few
// GiB, not 12-13; the pure-function test (test_planner.cpp) carries the exact
// 12-13 GiB reproduction. This profile proves the SAME failure mode - MAX's
// natural full-KV plan overflowing usable and the planner refusing instead of
// clamping - end to end on the real sf::load_model path.
static HardwareProfile colab_max_overflow_profile() {
    HardwareProfile hw;
    hw.ram_total_bytes = (2 * GiB) + (820ull * 1024 * 1024);   // 2.8 GiB total
    hw.ram_free_bytes = (2 * GiB) + (100ull * 1024 * 1024);    // ~2.1 GiB free
    return hw;
}

// FEAT-002 real-load-path reproduction of the MAX-must-clamp regression. On
// colab_max_overflow_profile() the natural full-n_ctx-KV MAX plan overflows
// usable, so RamMode::Max must CLAMP down to the largest plan that FITS (window
// the KV, cap experts) with resident experts >= AUTO's (monotonic), NOT refuse.
// These MAX-fits assertions FAIL against the current (pre-fix) planner, which
// returns fits==false (SF_OK with max_plan.fits==false). The coexisting correct
// behavior (an EXPLICIT target below the floor still refuses) stays green. The
// existing byte-identity half lives in test_target_ram_modes_real_path(); this
// test only pins the plan-geometry regression (no decode), so it runs from the
// GGUF alone and needs no stacked oracle.
static void test_max_clamps_not_refuses_real_path() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF");
    const char *strata_env = std::getenv("STRATAFLOW_TEST_MIXTRAL_GEOM_STRATA");
    if ((gguf == nullptr || gguf[0] == '\0') &&
        (strata_env == nullptr || strata_env[0] == '\0')) {
        std::printf("[skipped: set STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF or "
                    "STRATAFLOW_TEST_MIXTRAL_GEOM_STRATA] ");
        return;
    }

    std::string strata_path;
    bool packed_here = false;
    if (gguf != nullptr && gguf[0] != '\0') {
        strata_path = std::string(gguf) + ".maxclamp.strata";
        std::string err;
        const bool packed = pack_gguf_to_strata(gguf, strata_path, &err);
        CHECK(packed);
        if (!packed) {
            std::printf("[pack failed: %s] ", err.c_str());
            return;
        }
        packed_here = true;
    } else {
        strata_path = strata_env;
    }

    ScopedEnv clear_win("STRATAFLOW_ENGINE_KV_WINDOW", "");
    const HardwareProfile hw = colab_max_overflow_profile();

    // AUTO: windows the KV down and fits its conservative share.
    std::unique_ptr<Model> auto_model;
    PlacementPlan auto_plan{};
    const sf_status st_a =
        load_model(strata_path, hw, /*vram=*/0, /*ram=*/0, auto_model,
                   &auto_plan, /*cache_budget=*/0, RamMode::Auto,
                   /*target_ram_budget=*/0);
    CHECK_EQ(st_a, SF_OK);
    CHECK(auto_model != nullptr);
    if (auto_model == nullptr) {
        if (packed_here) std::remove(strata_path.c_str());
        return;
    }
    const ModelShape &shape = auto_model->shape();
    CHECK(auto_plan.fits);
    CHECK(auto_plan.planned_peak_bytes <= auto_plan.usable_ram_bytes);

    // MAX: must CLAMP to the largest plan that FITS, never refuse. [FAILS
    // pre-fix: the natural full-n_ctx-KV MAX plan overflows usable and the
    // planner returns fits==false.]
    std::unique_ptr<Model> max_model;
    PlacementPlan max_plan{};
    const sf_status st_m =
        load_model(strata_path, hw, /*vram=*/0, /*ram=*/0, max_model, &max_plan,
                   /*cache_budget=*/0, RamMode::Max, /*target_ram_budget=*/0);
    CHECK_EQ(st_m, SF_OK);

    std::printf("[max clamp real-path: AUTO fits=%d peak=%lluMiB usable=%lluMiB "
                "slots=%u kv=%u | MAX fits=%d peak=%lluMiB usable=%lluMiB "
                "slots=%u kv=%u] ",
                auto_plan.fits ? 1 : 0,
                static_cast<unsigned long long>(auto_plan.planned_peak_bytes >> 20),
                static_cast<unsigned long long>(auto_plan.usable_ram_bytes >> 20),
                auto_plan.expert_slots_resident, auto_plan.kv_window,
                max_plan.fits ? 1 : 0,
                static_cast<unsigned long long>(max_plan.planned_peak_bytes >> 20),
                static_cast<unsigned long long>(max_plan.usable_ram_bytes >> 20),
                max_plan.expert_slots_resident, max_plan.kv_window);
    if (!max_plan.fits) {
        std::printf("[MAX refusal: %s] ", max_plan.refusal_reason.c_str());
    }

    CHECK(max_model != nullptr);
    CHECK(max_plan.fits);
    CHECK(max_plan.planned_peak_bytes <= max_plan.usable_ram_bytes);

    // Monotonicity vs AUTO on the real load path (normalize 0 == full).
    CHECK(max_plan.usable_ram_bytes > auto_plan.usable_ram_bytes);
    CHECK(max_plan.expert_slots_resident >= auto_plan.expert_slots_resident);
    const uint32_t auto_win = auto_plan.kv_window ? auto_plan.kv_window : shape.n_ctx;
    const uint32_t max_win = max_plan.kv_window ? max_plan.kv_window : shape.n_ctx;
    CHECK(max_win >= auto_win);
    const uint32_t auto_hot =
        auto_plan.trunk_pool_layers ? auto_plan.trunk_pool_layers : shape.n_layers;
    const uint32_t max_hot =
        max_plan.trunk_pool_layers ? max_plan.trunk_pool_layers : shape.n_layers;
    CHECK(max_hot >= auto_hot);

    // COEXISTING CORRECT BEHAVIOR (stays green pre-fix): an EXPLICIT target
    // below the achievable floor still refuses loudly, naming the floor. The
    // clamp-not-refuse change for MODE paths must not touch this.
    std::unique_ptr<Model> tiny_model;
    PlacementPlan tiny_plan{};
    const uint64_t kTooSmall = 64ull * 1024 * 1024;  // 64 MiB, below the floor
    load_model(strata_path, hw, /*vram=*/0, /*ram=*/0, tiny_model, &tiny_plan,
               /*cache_budget=*/0, RamMode::Auto,
               /*target_ram_budget=*/kTooSmall);
    CHECK(!tiny_plan.fits);
    CHECK(!tiny_plan.refusal_reason.empty());
    CHECK(tiny_plan.refusal_reason.find("floor") != std::string::npos);
    std::printf("[explicit-below-floor refused honestly] ");

    if (packed_here) std::remove(strata_path.c_str());
}

static void test_target_ram_modes_real_path() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF");
    const char *strata_env = std::getenv("STRATAFLOW_TEST_MIXTRAL_GEOM_STRATA");
    const char *stacked = std::getenv("STRATAFLOW_TEST_MIXTRAL_GEOM_STACKED_GGUF");
    if ((gguf == nullptr || gguf[0] == '\0') &&
        (strata_env == nullptr || strata_env[0] == '\0')) {
        std::printf("[skipped: set STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF or "
                    "STRATAFLOW_TEST_MIXTRAL_GEOM_STRATA] ");
        return;
    }

    std::string strata_path;
    bool packed_here = false;
    if (gguf != nullptr && gguf[0] != '\0') {
        strata_path = std::string(gguf) + ".modes.strata";
        std::string err;
        const bool packed = pack_gguf_to_strata(gguf, strata_path, &err);
        CHECK(packed);
        if (!packed) {
            std::printf("[pack failed: %s] ", err.c_str());
            return;
        }
        packed_here = true;
    } else {
        strata_path = strata_env;
    }

    // Clear any inherited window env for the WHOLE test so the only window that
    // can engage is the one each mode's plan chose.
    ScopedEnv clear_win("STRATAFLOW_ENGINE_KV_WINDOW", "");

    const HardwareProfile hw = colab_profile();

    // ---- (a) EXPLICIT TIGHT TARGET -------------------------------------
    std::unique_ptr<Model> target_model;
    PlacementPlan target_plan{};
    const sf_status st_t = load_model(
        strata_path, hw, /*vram=*/0, /*ram=*/0, target_model, &target_plan,
        /*cache_budget=*/0, RamMode::Auto, /*target_ram_budget=*/kTightTargetBytes);
    CHECK_EQ(st_t, SF_OK);
    CHECK(target_model != nullptr);
    if (target_model == nullptr) {
        if (packed_here) std::remove(strata_path.c_str());
        return;
    }
    const ModelShape &shape = target_model->shape();

    // Geometry reached the planner on the real synthesized-metadata path.
    CHECK(shape.n_ctx > 0u);
    CHECK(shape.head_dim > 0u);
    CHECK(shape.n_head_kv > 0u);

    // The explicit target FITS and holds the peak at/under the target ceiling.
    CHECK(target_plan.fits);
    CHECK(target_plan.planned_peak_bytes <= kTightTargetBytes);
    // The KV window stepped down strictly inside (0, n_ctx).
    CHECK(target_plan.kv_window > 0u);
    CHECK(target_plan.kv_window < shape.n_ctx);
    // CRITICAL (prior review issue #2): the tight target MUST force adaptive
    // per-layer trunk streaming to engage - this is a HARD assertion, not a
    // conditional skip. The target above is dimensioned so streaming engages.
    CHECK(target_plan.trunk_pool_layers > 0u);
    CHECK(target_plan.trunk_pool_layers < shape.n_layers);
    // FEAT-002 COHERENCE INVARIANT (hard): because the trunk is HARD-streaming
    // (0 < trunk_pool_layers < n_layers) the KV MUST ALSO be windowed
    // (0 < kv_window < n_ctx) - the two resident levers co-reduce; a plan must
    // NEVER keep the full n_ctx KV resident while hard-streaming the trunk.
    CHECK(target_plan.kv_window > 0u && target_plan.kv_window < shape.n_ctx);

    std::printf("[target: peak=%lluMiB usable=%lluMiB kv_window=%u "
                "trunk_pool=%u slots=%u | n_ctx=%u n_layers=%u] ",
                static_cast<unsigned long long>(target_plan.planned_peak_bytes >> 20),
                static_cast<unsigned long long>(target_plan.usable_ram_bytes >> 20),
                target_plan.kv_window, target_plan.trunk_pool_layers,
                target_plan.expert_slots_resident, shape.n_ctx, shape.n_layers);

    // BUG 1 (hard): a genuinely-too-small target - below the TRUE minimal-KV
    // floor - must REFUSE, and the refusal must name an HONEST floor built from
    // the SMALLEST KV window (kMinKvWindow), NOT the full n_ctx reservation. The
    // dishonest pre-fix behavior reported FULL KV (the ~8-10 GiB "minimal
    // config") even though a small-window config fits in ~2.x GiB. We assert the
    // refusal_reason's KV figure is far below the fixture's full-n_ctx KV. 64
    // MiB is below even the process floor + margin, so no config can fit it.
    std::unique_ptr<Model> tiny_model;
    PlacementPlan tiny_plan{};
    const uint64_t kTooSmall = 64ull * 1024 * 1024;  // 64 MiB, below the floor
    load_model(strata_path, hw, /*vram=*/0, /*ram=*/0, tiny_model, &tiny_plan,
               /*cache_budget=*/0, RamMode::Auto,
               /*target_ram_budget=*/kTooSmall);
    CHECK(!tiny_plan.fits);
    CHECK(!tiny_plan.refusal_reason.empty());
    CHECK(tiny_plan.refusal_reason.find("floor") != std::string::npos);
    // The honest floor's KV figure reflects the SMALLEST window: for this
    // fixture (full KV = 2 GiB = 2048 MiB) the minimal-window KV at W=256 is
    // ~16 MiB, so the refusal must NOT mention the 2048 MiB full-KV figure.
    CHECK(tiny_plan.refusal_reason.find(" KV 2048 MiB") == std::string::npos);
    CHECK(tiny_plan.refusal_reason.find("KV window 256") != std::string::npos);
    std::printf("[too-small-target refused honestly] ");

    // ---- (b) AUTO (good-neighbor default) ------------------------------
    std::unique_ptr<Model> auto_model;
    PlacementPlan auto_plan{};
    const sf_status st_a =
        load_model(strata_path, hw, /*vram=*/0, /*ram=*/0, auto_model,
                   &auto_plan, /*cache_budget=*/0, RamMode::Auto,
                   /*target_ram_budget=*/0);
    CHECK_EQ(st_a, SF_OK);
    CHECK(auto_model != nullptr);
    CHECK(auto_plan.fits);
    // Auto's peak stays within its conservative usable share (the fit invariant).
    CHECK(auto_plan.planned_peak_bytes <= auto_plan.usable_ram_bytes);

    // ---- (c) MAX (dedicate the machine) --------------------------------
    std::unique_ptr<Model> max_model;
    PlacementPlan max_plan{};
    const sf_status st_m =
        load_model(strata_path, hw, /*vram=*/0, /*ram=*/0, max_model, &max_plan,
                   /*cache_budget=*/0, RamMode::Max, /*target_ram_budget=*/0);
    CHECK_EQ(st_m, SF_OK);
    CHECK(max_model != nullptr);
    CHECK(max_plan.fits);
    // FIT INVARIANT on the Max plan after the monotonicity guard's slot re-fit:
    // the final projected peak must still fit Max's usable budget. Guards the
    // same construction-only guarantee the pure-function sweep covers, now on
    // the REAL load path so a re-fit regression would be caught end to end.
    CHECK(max_plan.planned_peak_bytes <= max_plan.usable_ram_bytes);

    // FEAT-002 COHERENCE INVARIANT on BOTH AUTO and MAX (hard): neither mode may
    // keep the full n_ctx KV resident while hard-streaming the trunk. If the
    // trunk hard-streams (0 < trunk_pool_layers < n_layers) the KV window must
    // be windowed (0 < kv_window < n_ctx). This holds trivially when the trunk
    // is fully resident (trunk_pool_layers == 0).
    if (auto_plan.trunk_pool_layers > 0u &&
        auto_plan.trunk_pool_layers < shape.n_layers) {
        CHECK(auto_plan.kv_window > 0u && auto_plan.kv_window < shape.n_ctx);
    }
    if (max_plan.trunk_pool_layers > 0u &&
        max_plan.trunk_pool_layers < shape.n_layers) {
        CHECK(max_plan.kv_window > 0u && max_plan.kv_window < shape.n_ctx);
    }

    std::printf("[auto: peak=%lluMiB usable=%lluMiB kv_window=%u trunk_pool=%u "
                "slots=%u streamed? %d | max: peak=%lluMiB usable=%lluMiB "
                "kv_window=%u trunk_pool=%u slots=%u streamed? %d] ",
                static_cast<unsigned long long>(auto_plan.planned_peak_bytes >> 20),
                static_cast<unsigned long long>(auto_plan.usable_ram_bytes >> 20),
                auto_plan.kv_window, auto_plan.trunk_pool_layers,
                auto_plan.expert_slots_resident, auto_plan.stream_experts ? 1 : 0,
                static_cast<unsigned long long>(max_plan.planned_peak_bytes >> 20),
                static_cast<unsigned long long>(max_plan.usable_ram_bytes >> 20),
                max_plan.kv_window, max_plan.trunk_pool_layers,
                max_plan.expert_slots_resident, max_plan.stream_experts ? 1 : 0);

    // Max dedicates the whole machine: its usable ceiling is strictly larger
    // than auto's conservative good-neighbor share (which caps at 85% and
    // shaves a free-RAM headroom), so max's peak sits strictly above auto's.
    CHECK(max_plan.usable_ram_bytes > auto_plan.usable_ram_bytes);
    CHECK(auto_plan.planned_peak_bytes < max_plan.planned_peak_bytes);

    // Max holds a KV window at least as large as auto (0 = full n_ctx counts as
    // the largest; normalize both to a comparable column count).
    const uint32_t auto_win = auto_plan.kv_window ? auto_plan.kv_window : shape.n_ctx;
    const uint32_t max_win = max_plan.kv_window ? max_plan.kv_window : shape.n_ctx;
    CHECK(max_win >= auto_win);
    // Max keeps at least as many experts resident as auto.
    CHECK(max_plan.expert_slots_resident >= auto_plan.expert_slots_resident);
    // Max streams the trunk no harder than auto: either the trunk is fully
    // resident (trunk_pool_layers == 0) or its hot pool is at least as deep as
    // auto's (fewer streamed layers). Normalize 0 (= whole trunk resident) to
    // the full layer count for the comparison.
    const uint32_t auto_hot =
        auto_plan.trunk_pool_layers ? auto_plan.trunk_pool_layers : shape.n_layers;
    const uint32_t max_hot =
        max_plan.trunk_pool_layers ? max_plan.trunk_pool_layers : shape.n_layers;
    CHECK(max_hot >= auto_hot);

    // BUG 2 FIXTURE-BLINDNESS BREAKER (hard): on THIS budget the two modes must
    // choose GENUINELY DIFFERENT residency, not the trivially-identical full
    // residency the pre-fix tiny-expert fixture produced. Max must hold FULL KV
    // (window 0 == full n_ctx) while Auto's good-neighbor share forced a smaller
    // window; and Max must hold strictly MORE experts resident than Auto (the
    // exact inversion the real Colab run showed, where Max held FEWER). These
    // HARD-fail (not skip) with the fixture env set.
    CHECK(max_plan.usable_ram_bytes > auto_plan.usable_ram_bytes);  // total > share
    CHECK(max_plan.kv_window == 0u);                 // Max holds full n_ctx KV
    CHECK(auto_plan.kv_window > 0u);                 // Auto windowed down
    CHECK(auto_plan.kv_window < shape.n_ctx);
    CHECK(max_plan.expert_slots_resident > auto_plan.expert_slots_resident);
    CHECK(auto_plan.stream_experts);                 // Auto streams some experts
    CHECK(!max_plan.stream_experts);                 // Max holds the full set

    // PEAK-ESTIMATE SANITY (BUG 3, hard): a windowed plan's projected peak is
    // strictly BELOW what the SAME plan would project with the full n_ctx KV
    // reservation. The only difference is the KV term, so recompute Auto's peak
    // with full-n_ctx KV and confirm the windowed estimate is strictly smaller.
    const uint64_t floats_per_col =
        static_cast<uint64_t>(shape.head_dim) * shape.n_head_kv;
    const uint64_t full_kv_bytes_shape = static_cast<uint64_t>(shape.n_layers) *
                                         2 * floats_per_col * sizeof(float) *
                                         shape.n_ctx;
    const uint64_t auto_full_kv_peak =
        auto_plan.planned_peak_bytes - auto_plan.kv_resident_bytes +
        full_kv_bytes_shape;
    CHECK(auto_plan.kv_resident_bytes < full_kv_bytes_shape);
    CHECK(auto_plan.planned_peak_bytes < auto_full_kv_peak);
    // The estimate must track the engine's REAL resident footprint, not stack
    // to ~2x it. The real resident components are everything the engine actually
    // allocates: resident KV + resident experts + one-layer staging + resident
    // trunk + process floor. The only ESTIMATE padding on top is the
    // conservative compute scratch + safety margin. BUG 3 was that padding
    // (plus a full-n_ctx KV charge) roughly DOUBLING reality. Assert the padding
    // stays below the real components here (planned_peak < 2x real) on this
    // small fixture; the strict real-Mixtral-scale < 1.5x bound - where the
    // multi-GiB KV dwarfs the fixed floors - is asserted in test_planner.cpp on
    // the Mixtral geometry (test_estimate_tracks_real_resident).
    const uint64_t auto_padding =
        auto_plan.compute_bytes + auto_plan.safety_margin_bytes;
    const uint64_t auto_real_components =
        auto_plan.planned_peak_bytes - auto_padding;
    CHECK(auto_padding < auto_real_components);  // planned_peak < 2x real

    // ---- (d) HARD byte-identity across modes + oracle ------------------
    // The decode half needs the matched STACKED GGUF for the tokenizer + the
    // libllama oracle. Skip it cleanly (the geometry assertions above already
    // stand) when that fixture is absent.
    if (stacked == nullptr || stacked[0] == '\0') {
        std::printf("[decode half skipped: set "
                    "STRATAFLOW_TEST_MIXTRAL_GEOM_STACKED_GGUF] ");
        if (packed_here) std::remove(strata_path.c_str());
        return;
    }

    const std::string prompt = "The city of";
    const int n_generate = 6;
    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(stacked, prompt, prompt_ids));
    CHECK(!prompt_ids.empty());

    // Decode on the SAME .strata with each mode's levers (window + trunk pool +
    // slots). All three must produce the SAME greedy sequence and bit-identical
    // logits: the modes change RESIDENCY only, never which bytes compute.
    DecodeRun target_run = decode_with_plan(
        strata_path.c_str(), prompt_ids, n_generate,
        target_plan.expert_slots_resident, target_plan.kv_window,
        target_plan.trunk_pool_layers);
    CHECK(target_run.ok);
    DecodeRun auto_run = decode_with_plan(
        strata_path.c_str(), prompt_ids, n_generate,
        auto_plan.expert_slots_resident, auto_plan.kv_window,
        auto_plan.trunk_pool_layers);
    CHECK(auto_run.ok);
    DecodeRun max_run = decode_with_plan(
        strata_path.c_str(), prompt_ids, n_generate,
        max_plan.expert_slots_resident, max_plan.kv_window,
        max_plan.trunk_pool_layers);
    CHECK(max_run.ok);
    if (!target_run.ok || !auto_run.ok || !max_run.ok) {
        if (packed_here) std::remove(strata_path.c_str());
        return;
    }

    // The tight target engaged the KV window, so its resident KV is strictly
    // below the full reservation (the FIT win, proven byte-identical below).
    CHECK(target_run.resident_kv_bytes < target_run.full_kv_bytes);

    // FEAT-002 STREAMED-BYTES/TOKEN DROP (hard). The coherent AUTO retune caches
    // materially MORE experts than the old 2-bundle baseline (slots=104 here vs
    // 2), which directly cuts how many expert bundles are re-streamed per token.
    // Decode the SAME prompt with a forced 2-slot pool (the old baseline that
    // re-streams almost every routed expert every token) and with AUTO's large
    // pool, and assert the per-token streamed-expert bytes DROP. Byte-identity
    // is unaffected (both runs compute the same bytes); only HOW MANY are read
    // from disk changes. A larger resident pool must stream strictly fewer
    // expert bytes per decoded token.
    const int n_decoded = n_generate + static_cast<int>(prompt_ids.size());
    DecodeRun small_pool = decode_with_plan(
        strata_path.c_str(), prompt_ids, n_generate,
        /*expert_slots=*/static_cast<uint32_t>(shape.n_experts_used),
        auto_plan.kv_window, auto_plan.trunk_pool_layers);
    CHECK(small_pool.ok);
    // auto_run above decoded with AUTO's large pool (slots=104); reuse it.
    CHECK(auto_plan.expert_slots_resident > shape.n_experts_used);
    if (small_pool.ok && n_decoded > 0) {
        const double small_per_tok =
            static_cast<double>(small_pool.streamed_bytes) / n_decoded;
        const double auto_per_tok =
            static_cast<double>(auto_run.streamed_bytes) / n_decoded;
        std::printf("[streamed/token: 2-slot=%.0fB auto(%u-slot)=%.0fB] ",
                    small_per_tok, auto_plan.expert_slots_resident,
                    auto_per_tok);
        // The larger pool re-streams strictly fewer expert bytes per token.
        CHECK(auto_run.streamed_bytes < small_pool.streamed_bytes);
    }

    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    const bool oracle_ok = engine::run_oracle_sequence(
        stacked, prompt, n_generate, oracle_tokens, &oracle_step_logits);
    CHECK(oracle_ok);

    bool seq_match = target_run.tokens.size() == auto_run.tokens.size() &&
                     target_run.tokens.size() == max_run.tokens.size() &&
                     target_run.tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < target_run.tokens.size() &&
                       i < auto_run.tokens.size() &&
                       i < max_run.tokens.size() && i < oracle_tokens.size();
         ++i) {
        CHECK_EQ(target_run.tokens[i], auto_run.tokens[i]);
        CHECK_EQ(target_run.tokens[i], max_run.tokens[i]);
        CHECK_EQ(target_run.tokens[i], oracle_tokens[i]);
        if (target_run.tokens[i] != auto_run.tokens[i] ||
            target_run.tokens[i] != max_run.tokens[i] ||
            target_run.tokens[i] != oracle_tokens[i]) {
            seq_match = false;
        }
    }

    const double max_ta = max_step_delta(target_run.step_logits,
                                         auto_run.step_logits);
    const double max_tm = max_step_delta(target_run.step_logits,
                                         max_run.step_logits);
    const double max_to = max_step_delta(target_run.step_logits,
                                         oracle_step_logits);

    std::printf("[seq match=%s target-vs-auto|delta|=%.3e "
                "target-vs-max|delta|=%.3e target-vs-oracle|delta|=%.3e] ",
                seq_match ? "yes" : "no", max_ta, max_tm, max_to);

    CHECK(seq_match);
    // HARD cross-mode bit-identity: the modes change residency only, so the
    // decode is BIT-IDENTICAL regardless of mode/target.
    CHECK(max_ta == 0.0);
    CHECK(max_tm == 0.0);
    // Soft quant-noise bound vs the libllama oracle (same 1e-1 bound as
    // test_kv_window_real_path for this random-weight q4_k fixture at
    // n_ctx=32768; an F32 fixture holds < 1e-3, a real Mixtral q3_k/q4_k well
    // under 2e-2).
    CHECK(max_to < 1e-1);

    if (packed_here) std::remove(strata_path.c_str());
}

static void run_all() {
    RUN(test_target_ram_modes_real_path);
    RUN(test_max_clamps_not_refuses_real_path);
}

TEST_MAIN()

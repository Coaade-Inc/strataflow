// Regression gate for the .strata MoE shape pre-pass (the auto-residency OOM
// regression). BEFORE the fix, load_strata_model built a ModelShape from a
// VOCAB-ONLY llama_model (so expert_bytes/trunk_bytes/total_bytes were 0) and
// the superblock patch only filled is_moe/n_experts/n_layers. plan_placement's
// MoE branch is gated on expert_bytes > 0, so a real legacy-Mixtral .strata
// fell to the DENSE branch: expert_slots_resident == 0, stream_experts ==
// false, and the whole ~19 GB model was planned fully resident, OOM-killing a
// 12 GB box (rc=-9). The fix populates expert_bytes (from the expert index
// blob_length), trunk_bytes/total_bytes (from the superblock + index), and
// n_experts_used (from the embedded GGUF metadata) BEFORE planning, so the plan
// correctly sees MoE + streaming.
//
// This test PACKS the legacy per-expert GGUF fixture to a .strata in-sandbox,
// loads it through the public sf::load_model seam with a modest simulated
// free-RAM budget, and asserts the loaded ModelShape and PlacementPlan are the
// MoE path, NOT the dense branch. The assertions shape.expert_bytes > 0 and
// plan.expert_slots_resident > 0 are exactly what FAILS on the pre-fix code.
//
// GUARDED on STRATAFLOW_TEST_LEGACY_MOE_GGUF (same env-var pattern as
// test_legacy_per_expert_pack), so the suite stays green in CI where the
// fixture is not generated. Generate + run locally:
//   build/debug/bin/make-legacy-moe-gguf FIX.gguf FIX.stacked.gguf
//   STRATAFLOW_TEST_LEGACY_MOE_GGUF=FIX.gguf ctest --preset debug
//       -R test_strata_shape_prepass
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "hw/profiler.h"
#include "model/model.h"
#include "plan/planner.h"
#include "strataflow/strataflow.h"
#include "test_util.h"
#include "tws/strata_file.h"
#include "tws/strata_format.h"
#include "tws/strata_pack.h"

using namespace sf;

static constexpr uint64_t GiB = 1024ull * 1024 * 1024;

// A modest free-RAM budget (12 GiB box with 10 GiB free) matching the machine
// the OOM regression was reported on.
static HardwareProfile modest_profile() {
    HardwareProfile hw;
    hw.ram_total_bytes = 12 * GiB;
    hw.ram_free_bytes = 10 * GiB;
    return hw;
}

// The true in-sandbox regression gate: pack the legacy per-expert fixture to a
// .strata, load it through sf::load_model, and prove the plan is MoE+streaming
// (NOT the pre-fix dense branch with expert_slots_resident == 0).
static void test_strata_moe_not_dense() {
    const char *legacy = std::getenv("STRATAFLOW_TEST_LEGACY_MOE_GGUF");
    if (legacy == nullptr || legacy[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_LEGACY_MOE_GGUF] ");
        return;
    }
    const std::string legacy_path = legacy;
    const std::string strata_path = legacy_path + ".prepass.strata";

    // 1. Pack the legacy per-expert GGUF -> .strata.
    std::string err;
    const bool packed = pack_gguf_to_strata(legacy_path, strata_path, &err);
    CHECK(packed);
    if (!packed) {
        std::printf("[pack failed: %s] ", err.c_str());
        return;
    }

    // 2. Sanity: the superblock itself reports experts (the fixture is 8-expert
    // top-2 over 2 layers, so the index has 2*8 == 16 entries).
    uint64_t expert_bytes_from_index = 0;
    {
        StrataReader reader;
        CHECK(reader.open(strata_path));
        const StrataSuperblock &sb = reader.superblock();
        CHECK(sb.n_experts > 0u);
        CHECK_EQ(sb.n_experts, static_cast<uint32_t>(8));
        CHECK(reader.index_count() > 0u);
        if (reader.index_count() > 0) {
            expert_bytes_from_index = reader.index()[0].blob_length;
        }
        CHECK(expert_bytes_from_index > 0u);
    }

    // 3. Load the model through the public seam with a modest free-RAM budget.
    std::unique_ptr<Model> model;
    PlacementPlan plan{};
    const sf_status st =
        load_model(strata_path, modest_profile(), /*vram=*/0, /*ram=*/0, model,
                   &plan);
    CHECK_EQ(st, SF_OK);
    CHECK(model != nullptr);
    if (model == nullptr) {
        std::remove(strata_path.c_str());
        return;
    }

    // 4. The loaded shape must be MoE with the byte fields and top-k populated.
    // shape.expert_bytes > 0 is the FIELD that was 0 on the pre-fix code and
    // sent the planner to the dense branch; trunk_bytes > 0 and n_experts_used
    // > 0 are the other fields the pre-pass fills.
    const ModelShape &shape = model->shape();
    CHECK(shape.is_moe);
    CHECK_EQ(shape.n_experts, static_cast<uint32_t>(8));
    CHECK(shape.n_experts_used > 0u);
    CHECK(shape.expert_bytes > 0u);
    CHECK(shape.trunk_bytes > 0u);

    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(shape.n_layers) * shape.n_experts);  // 16

    // 5. THE REGRESSION ASSERTION: the plan is NOT the dense branch. The dense
    // branch sets expert_slots_resident == 0; the MoE branch always reserves at
    // least one layer's top-k. For this tiny fixture under a generous budget the
    // whole working set fits, so it is fully resident (slots == full_slots,
    // stream_experts == false); that still proves "not dense".
    CHECK(plan.expert_slots_resident > 0u);
    std::printf(
        "[prepass moe: n_layers=%u n_experts=%u top-k=%u expert_bytes=%lluB "
        "trunk_bytes=%lluB slots=%u stream=%s] ",
        shape.n_layers, shape.n_experts, shape.n_experts_used,
        static_cast<unsigned long long>(shape.expert_bytes),
        static_cast<unsigned long long>(shape.trunk_bytes),
        plan.expert_slots_resident, plan.stream_experts ? "yes" : "no");

    // 6. Force BOUNDED STREAMING with a tight expert-RAM cap (the OOM-avoidance
    // path on a real .strata). A budget of ~3 expert bundles is well below the
    // 16-bundle working set but leaves room for a layer's top-k, so streaming
    // must engage and the resident pool must bound strictly below full_slots
    // while never dropping below n_experts_used.
    const uint64_t cache_budget = 3 * shape.expert_bytes;
    std::unique_ptr<Model> model2;
    PlacementPlan plan2{};
    const sf_status st2 =
        load_model(strata_path, modest_profile(), /*vram=*/0, /*ram=*/0, model2,
                   &plan2, cache_budget);
    CHECK_EQ(st2, SF_OK);
    CHECK(model2 != nullptr);
    if (model2 != nullptr) {
        CHECK(plan2.stream_experts);
        CHECK(plan2.expert_slots_resident >= shape.n_experts_used);
        CHECK(plan2.expert_slots_resident < full_slots);
        std::printf("[bounded: cache_budget=%lluB slots=%u stream=%s] ",
                    static_cast<unsigned long long>(cache_budget),
                    plan2.expert_slots_resident,
                    plan2.stream_experts ? "yes" : "no");
    }

    std::remove(strata_path.c_str());
}

static void run_all() {
    RUN(test_strata_moe_not_dense);
}

TEST_MAIN()

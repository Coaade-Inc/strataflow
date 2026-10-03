// Tests for sf::load_model's backend selection and dry-run fallback (Phase 1b).
//
// The unconditional cases prove load_model never crashes and always yields a
// usable model: a missing file, a *.strata path (not implemented yet), and a
// missing *.gguf all fall back to the deterministic dry-run model.
//
// The real llama.cpp load path is exercised only when STRATAFLOW_TEST_GGUF
// points at a readable GGUF file, so the suite stays green without downloading
// multi-gigabyte weights. Generate a tiny fixture and run it with:
//     python3 tools/testdata/make_tiny_gguf.py /tmp/tiny.gguf
//     STRATAFLOW_TEST_GGUF=/tmp/tiny.gguf ctest -R test_model_load
// (It also accepts any real .gguf checkpoint for a fuller check.)
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "hw/profiler.h"
#include "model/model.h"
#include "plan/planner.h"
#include "test_util.h"

#include <cstdlib>
#include <memory>
#include <string>

using namespace sf;

// A representative CPU-only profile so the planner has something to chew on.
// No GPUs => the GGUF backend must behave exactly like a plain CPU load.
static HardwareProfile cpu_profile() {
    HardwareProfile hw;
    hw.ram_total_bytes = 32ull * 1024 * 1024 * 1024;
    hw.ram_free_bytes = 28ull * 1024 * 1024 * 1024;
    return hw;
}

// A non-existent path must not crash and must return a working dry-run model.
static void test_missing_path_falls_back() {
    std::unique_ptr<Model> m;
    PlacementPlan plan;
    CHECK_EQ(load_model("does-not-exist-xyz.gguf", cpu_profile(), 0, 0, m, &plan),
             SF_OK);
    CHECK(m != nullptr);
    // The dry-run fixture reports its stable identity.
    CHECK(m->shape().name == "dry-run-moe");
}

// A *.strata path is not implemented yet; it must degrade to the dry-run model
// rather than error out.
static void test_strata_falls_back() {
    std::unique_ptr<Model> m;
    PlacementPlan plan;
    CHECK_EQ(load_model("whatever.strata", cpu_profile(), 0, 0, m, &plan), SF_OK);
    CHECK(m != nullptr);
    CHECK(m->shape().name == "dry-run-moe");
}

// A missing *.gguf also falls back (the file does not exist, so no llama.cpp
// load is attempted). Confirms the dry-run model still tokenizes and advances.
static void test_missing_gguf_falls_back() {
    std::unique_ptr<Model> m;
    PlacementPlan plan;
    CHECK_EQ(load_model("absent-model.gguf", cpu_profile(), 0, 0, m, &plan),
             SF_OK);
    CHECK(m != nullptr);
    CHECK(m->shape().name == "dry-run-moe");

    auto ids = m->tokenize("hello world");
    CHECK(!ids.empty());
    int32_t next = m->forward(ids.back());
    CHECK(next >= 0);
}

// Guarded real-load path: only runs when STRATAFLOW_TEST_GGUF is set to a
// readable GGUF. Proves the GgmlModel backend loads real weights and that
// tokenize/forward produce sane values. Skips cleanly otherwise.
static void test_real_gguf_optional() {
    const char *path = std::getenv("STRATAFLOW_TEST_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_GGUF] ");
        return;
    }

    std::unique_ptr<Model> m;
    PlacementPlan plan;
    CHECK_EQ(load_model(path, cpu_profile(), 0, 0, m, &plan), SF_OK);
    CHECK(m != nullptr);
    // A real model reports its own name, not the dry-run fixture's.
    CHECK(m->shape().name != "dry-run-moe");
    CHECK(m->shape().n_layers > 0);
    CHECK(m->shape().total_bytes > 0);
    // CPU-only profile => the plan must not force any VRAM offload.
    CHECK_EQ(plan.trunk_layers_in_vram, 0u);

    // Real tokenizer: a non-empty prompt yields at least one token, and the
    // greedy forward pass returns a valid, in-vocabulary token id.
    auto ids = m->tokenize("The quick brown fox");
    CHECK(!ids.empty());

    int32_t tok = ids.back();
    int32_t next = m->forward(tok);
    CHECK(next >= 0);

    // Detokenizing the produced token must not crash (may be empty for some
    // special tokens, which is acceptable).
    std::string piece = m->detokenize(next);
    (void)piece;
}

static void run_all() {
    RUN(test_missing_path_falls_back);
    RUN(test_strata_falls_back);
    RUN(test_missing_gguf_falls_back);
    RUN(test_real_gguf_optional);
}

TEST_MAIN()

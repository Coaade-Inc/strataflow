// EC-1 oracle correctness gate (docs/ENGINE_CORE_DESIGN.md section 7.3): on the
// SAME tiny MoE gguf, run both the libllama oracle (llama_decode) and OUR OWN
// ggml engine for a single token at position 0, and assert:
//   (a) identical greedy argmax token id  -- the HARD gate, and
//   (b) max absolute logit delta within tolerance -- the soft gate.
//
// The de-risking spike (spike/engine_core/spike_engine.cpp) measured, on this
// fixture: argmax MATCH (both token 71), max|logit delta| = 5.3167e-5, mean
// 1.2078e-5. We use a safe 1e-3 tolerance here (headroom for compiler/platform
// op-ordering differences) while keeping argmax EQUALITY as the real gate.
//
// GUARDED on STRATAFLOW_TEST_MOE_GGUF (same skip-when-absent pattern as
// test_moe_stream / test_streaming_decode), so the suite stays green in CI
// where the fixture is not generated. Generate + run locally:
//   uv run --with gguf --with numpy python3
//       tools/testdata/make_tiny_moe_gguf.py /projects/sandbox/moe.gguf
//   STRATAFLOW_TEST_MOE_GGUF=/projects/sandbox/moe.gguf ctest -R test_engine_oracle
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "model/engine/engine.h"
#include "test_util.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

using namespace sf;

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

// Engine single-token forward must match the libllama oracle on the tiny MoE.
static void test_engine_matches_oracle() {
    const char *path = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }

    // Same seed token both paths run: the model's BOS, at position 0.
    const int32_t bos = engine::vocab_bos_token(path);
    CHECK(bos >= 0);
    if (bos < 0) return;

    std::vector<float> oracle_logits;
    CHECK(engine::run_oracle_single_token(path, bos, oracle_logits));
    CHECK(!oracle_logits.empty());

    std::unique_ptr<engine::Engine> eng = engine::Engine::load(path);
    CHECK(eng != nullptr);
    if (eng == nullptr) return;

    std::vector<float> engine_logits;
    CHECK(eng->forward(bos, /*pos=*/0, engine_logits));
    CHECK(!engine_logits.empty());

    CHECK_EQ(engine_logits.size(), oracle_logits.size());
    if (engine_logits.size() != oracle_logits.size()) return;

    const int32_t oracle_arg = argmax(oracle_logits);
    const int32_t engine_arg = argmax(engine_logits);

    double max_abs = 0.0;
    for (size_t i = 0; i < engine_logits.size(); ++i) {
        const double d = std::fabs(static_cast<double>(engine_logits[i]) -
                                   static_cast<double>(oracle_logits[i]));
        if (d > max_abs) max_abs = d;
    }

    std::printf("[engine argmax=%d oracle argmax=%d max|delta|=%.3e] ",
                engine_arg, oracle_arg, max_abs);

    // (a) HARD gate: identical greedy argmax.
    CHECK_EQ(engine_arg, oracle_arg);
    // (b) soft gate: logits within tolerance. Spike measured ~5.3e-5; 1e-3 is
    // safe headroom across compilers/platforms.
    CHECK(max_abs < 1e-3);
}

static void run_all() {
    RUN(test_engine_matches_oracle);
}

TEST_MAIN()

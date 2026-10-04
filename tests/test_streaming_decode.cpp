// Integration test (CI-proxy, CPU-only): EC-5 streaming determinism on the
// DEFAULT inference path.
//
// Historical note: this test was written in Phase 3 to assert the llama_decode
// -era streaming-buffer (strata_stream_buft) cache stats via an end-to-end
// decode under STRATAFLOW_FORCE_STREAM_EXPERTS. EC-5 made OUR engine the
// default and ONLY inference path (docs/ENGINE_CORE_DESIGN.md section 8 EC-5):
// llama_decode no longer runs the model, so stream_buft is no longer on the
// inference path and its decode-time residency pass no longer runs during
// forward(). This test is therefore RETARGETED (resolution option (a)) to the
// engine's EC-3 top-k expert residency -- the modern equivalent of the Phase 3
// streaming cache -- driven through the public sf::Model seam, which is what
// the CLI/server actually use.
//
// It asserts the hard streaming guarantees on the real inference path:
//   (a) forward() returns valid in-vocabulary tokens (both pools);
//   (b) the greedy token sequence is BYTE-IDENTICAL between an auto (full
//       working set) pool and a deliberately SMALL pool (fewer slots than the
//       fixture's experts/layer) -- the determinism gate: bounded streaming
//       must never change the output;
//   (c) with the small pool the engine's bounded SlotPool shows hits AND misses
//       AND non-zero evictions (experts streamed/evicted/reloaded across
//       layers/tokens), proving the cache was actually exercised; and
//   (d) resident expert bytes stay bounded by n_slots * slot_bytes and strictly
//       below the full-resident footprint across a multi-step decode.
//
// The pool size is driven by STRATAFLOW_ENGINE_EXPERT_SLOTS (the EC-3 knob that
// GgmlModel::forward_engine reads), so (a)/(b) exercise the exact default
// inference path a user hits. For (c)/(d) we read the engine's EngineCacheStats
// via the engine API on an equivalently-configured engine (the stats are not
// surfaced through sf::Model, by design -- the seam is backend-agnostic). The
// byte-identity + eviction guarantee is ALSO covered directly against the
// oracle in test_engine_oracle (EC-3 bounded-pool case); this test adds the
// end-to-end proof through the production sf::Model seam.
//
// Guarded on STRATAFLOW_TEST_MOE_GGUF (skip cleanly if unset), same pattern as
// test_moe_stream / test_engine_oracle. Generate + run:
//     uv run --with gguf --with numpy python3
//         tools/testdata/make_tiny_moe_gguf.py /path/tiny_moe.gguf
//     STRATAFLOW_TEST_MOE_GGUF=/path/tiny_moe.gguf ctest -R test_streaming_decode
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "hw/profiler.h"
#include "model/engine/engine.h"
#include "model/model.h"
#include "plan/planner.h"
#include "test_util.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace sf;

static HardwareProfile cpu_profile() {
    HardwareProfile hw;
    hw.ram_total_bytes = 32ull * 1024 * 1024 * 1024;
    hw.ram_free_bytes = 28ull * 1024 * 1024 * 1024;
    return hw;
}

// Portable set/unset for the EC-3 pool-size knob the engine reads.
static void set_expert_slots(uint32_t n_slots) {
#if defined(_WIN32)
    if (n_slots == 0) {
        _putenv_s("STRATAFLOW_ENGINE_EXPERT_SLOTS", "");
    } else {
        _putenv_s("STRATAFLOW_ENGINE_EXPERT_SLOTS",
                  std::to_string(n_slots).c_str());
    }
#else
    if (n_slots == 0) {
        unsetenv("STRATAFLOW_ENGINE_EXPERT_SLOTS");
    } else {
        setenv("STRATAFLOW_ENGINE_EXPERT_SLOTS",
               std::to_string(n_slots).c_str(), 1);
    }
#endif
}

// Decode `n_steps` greedy tokens through the DEFAULT sf::Model inference path
// (the engine, EC-5) from a fresh model, with the engine's expert-residency
// pool sized by STRATAFLOW_ENGINE_EXPERT_SLOTS. Returns the token sequence.
static std::vector<int32_t> decode_sequence(const char *path, uint32_t n_slots,
                                            int n_steps) {
    set_expert_slots(n_slots);  // 0 == auto (full working set)

    std::unique_ptr<Model> m;
    PlacementPlan plan;
    std::vector<int32_t> out;
    if (load_model(path, cpu_profile(), 0, 0, m, &plan) != SF_OK ||
        m == nullptr) {
        set_expert_slots(0);
        return out;
    }

    auto ids = m->tokenize("hello world");
    int32_t tok = ids.empty() ? 1 : ids.back();
    for (int i = 0; i < n_steps; ++i) {
        tok = m->forward(tok);
        out.push_back(tok);
    }

    set_expert_slots(0);  // reset the knob for subsequent loads/tests
    return out;
}

// Drive the engine directly (same prompt, same slot budget) to read the
// EngineCacheStats the sf::Model seam does not surface. Mirrors the decode the
// sf::Model path performs (prefill the prompt tokens, then greedy decode).
static engine::EngineCacheStats engine_stats_for(const char *path,
                                                 uint32_t slots, int n_steps,
                                                 std::vector<int32_t> *out_seq) {
    engine::EngineCacheStats stats{};
    std::vector<int32_t> prompt_ids;
    if (!engine::vocab_tokenize(path, "hello world", prompt_ids) ||
        prompt_ids.empty()) {
        return stats;
    }

    std::unique_ptr<engine::Engine> eng = engine::Engine::load(path, slots);
    if (eng == nullptr) return stats;
    eng->reset_kv();

    int32_t pos = 0;
    std::vector<float> logits;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        if (!eng->forward(prompt_ids[i], pos, logits)) return stats;
        ++pos;
    }
    // The sf::Model path seeds decode from the LAST prompt token's argmax; here
    // we only need to exercise the cache over n_steps decodes to read stats, so
    // feed back the greedy argmax exactly as GgmlModel::forward does.
    int32_t best = 0;
    {
        float best_v = logits.empty() ? 0.0f : logits[0];
        for (size_t i = 0; i < logits.size(); ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<int32_t>(i); }
        }
    }
    for (int i = 0; i < n_steps; ++i) {
        if (out_seq != nullptr) out_seq->push_back(best);
        if (!eng->forward(best, pos, logits)) break;
        ++pos;
        float best_v = logits.empty() ? 0.0f : logits[0];
        best = 0;
        for (size_t j = 0; j < logits.size(); ++j) {
            if (logits[j] > best_v) { best_v = logits[j]; best = static_cast<int32_t>(j); }
        }
    }
    stats = eng->cache_stats();
    return stats;
}

static void test_streaming_bounded_and_deterministic() {
    const char *path = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }

    const int n_steps = 12;

    // --- Auto pool (holds the whole expert working set, no eviction) through
    //     the DEFAULT sf::Model inference path. This is the reference the
    //     bounded run must match byte-for-byte. --------------------------------
    std::vector<int32_t> full_seq =
        decode_sequence(path, /*n_slots=*/0, n_steps);
    CHECK(!full_seq.empty());
    for (int32_t t : full_seq) CHECK(t >= 0);  // (a) valid tokens

    // --- Small-pool run: fewer slots than the fixture's 8 experts/layer so the
    //     per-layer top-k residency must evict and reload across layers/tokens.
    //     3 slots >= n_expert_used (2) so a single layer's top-k always fits. --
    const uint32_t small_slots = 3;
    std::vector<int32_t> small_seq = decode_sequence(path, small_slots, n_steps);
    CHECK(!small_seq.empty());
    for (int32_t t : small_seq) CHECK(t >= 0);  // (a) valid tokens

    // (b) BYTE-IDENTICAL determinism gate: bounded streaming through the engine
    //     must produce exactly the same tokens as the auto (full) pool, on the
    //     real inference path.
    CHECK(full_seq.size() == small_seq.size());
    CHECK(full_seq == small_seq);

    // (c)/(d) Read the engine's bounded SlotPool stats (not surfaced through
    //     sf::Model) via the engine API with the SAME budget. The auto pool
    //     never evicts; the small pool must show hits, misses, and evictions,
    //     and a resident footprint bounded below the full working set.
    engine::EngineCacheStats full_stats =
        engine_stats_for(path, /*slots=*/0, n_steps, nullptr);
    engine::EngineCacheStats small_stats =
        engine_stats_for(path, small_slots, n_steps, nullptr);

    std::printf("[seq==%s auto(evict=%llu resident=%lluB) "
                "small(hits=%llu miss=%llu evict=%llu resident=%lluB full=%lluB)] ",
                (full_seq == small_seq) ? "yes" : "no",
                static_cast<unsigned long long>(full_stats.evictions),
                static_cast<unsigned long long>(full_stats.resident_bytes()),
                static_cast<unsigned long long>(small_stats.hits),
                static_cast<unsigned long long>(small_stats.misses),
                static_cast<unsigned long long>(small_stats.evictions),
                static_cast<unsigned long long>(small_stats.resident_bytes()),
                static_cast<unsigned long long>(small_stats.full_bytes));

    // (c) the bounded pool was exercised: hits AND misses AND evictions.
    CHECK(small_stats.hits > 0u);
    CHECK(small_stats.misses > 0u);
    CHECK(small_stats.evictions > 0u);
    // The auto pool holds the whole working set and never evicts.
    CHECK_EQ(full_stats.evictions, static_cast<uint64_t>(0));

    // (d) resident expert bytes bounded by n_slots * slot_bytes, and strictly
    //     smaller than the full-resident footprint (bounded-memory proxy).
    CHECK(small_stats.slot_bytes > 0u);
    CHECK_EQ(small_stats.resident_bytes(),
             small_stats.n_slots * small_stats.slot_bytes);
    CHECK(small_stats.resident_bytes() < small_stats.full_bytes);
    CHECK(small_stats.resident_bytes() < full_stats.resident_bytes());
}

static void run_all() {
    RUN(test_streaming_bounded_and_deterministic);
}

TEST_MAIN()

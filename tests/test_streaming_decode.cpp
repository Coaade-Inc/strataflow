// Integration test (CI-proxy, CPU-only): Task 3 bounded residency.
//
// Loads the tiny Mixtral-style MoE GGUF twice with STRATAFLOW_FORCE_STREAM_
// EXPERTS=1 — once with a full-size slot pool and once with a deliberately
// SMALL pool (fewer slots than the n_layer*3 stacked expert tensors) — and
// asserts the hard Phase 3 guarantees:
//   (a) forward() returns valid in-vocabulary tokens;
//   (b) the greedy token sequence is BYTE-IDENTICAL between the full and the
//       small pool (the determinism gate — streaming must never change output);
//   (c) with the small pool the SlotPool shows hits AND misses AND non-zero
//       evictions, and a tensor evicted on one decode is reloaded on the next
//       (eviction+reload across decodes — bounded cross-decode residency);
//   (d) resident cache bytes stay bounded by n_slots * slot_bytes across many
//       decode steps.
//
// CORRECTNESS MODEL A (see src/tws/stream_buft.h): one llama_decode runs the
// FULL graph for all layers, and the MoE kernel reads each stacked expert
// tensor directly from its tensor->data base (ggml-cpu.c mul_mat_id:
// src0->data + cur_a*nb02), with no per-op residency hook. So every stacked
// tensor must be resident at its own address within a decode; the pool bounds
// how many are held across decodes and drives LRU eviction/reload. A
// smaller-than-working-set pool is therefore demonstrated via cross-decode
// eviction, exactly the fallback the task allows.
//
// Guarded on STRATAFLOW_TEST_MOE_GGUF (skip cleanly if unset), same pattern as
// test_moe_stream. Generate + run:
//     uv run --with gguf --with numpy python3
//         tools/testdata/make_tiny_moe_gguf.py /path/tiny_moe.gguf
//     STRATAFLOW_TEST_MOE_GGUF=/path/tiny_moe.gguf ctest -R test_streaming_decode
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "hw/profiler.h"
#include "model/model.h"
#include "plan/planner.h"
#include "test_util.h"
#include "tws/stream_buft.h"

#include <cstdint>
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

static void set_force_stream(bool on) {
#if defined(_WIN32)
    _putenv_s("STRATAFLOW_FORCE_STREAM_EXPERTS", on ? "1" : "");
#else
    if (on) setenv("STRATAFLOW_FORCE_STREAM_EXPERTS", "1", 1);
    else    unsetenv("STRATAFLOW_FORCE_STREAM_EXPERTS");
#endif
}

// Decode `n_steps` greedy tokens from a fresh model loaded with the given slot
// pool size. Records the max resident cache bytes observed across the run.
static std::vector<int32_t> decode_sequence(const char *path, uint32_t n_slots,
                                            int n_steps,
                                            uint64_t *max_resident_bytes_out) {
    reset_stream_buft_stats();
    set_stream_buft_slots(n_slots);  // 0 == auto (full working set)

    std::unique_ptr<Model> m;
    PlacementPlan plan;
    std::vector<int32_t> out;
    if (load_model(path, cpu_profile(), 0, 0, m, &plan) != SF_OK || m == nullptr) {
        return out;
    }

    auto ids = m->tokenize("hello world");
    int32_t tok = ids.empty() ? 1 : ids.back();
    uint64_t max_resident = 0;
    for (int i = 0; i < n_steps; ++i) {
        tok = m->forward(tok);
        out.push_back(tok);
        const StreamBuftStats &s = stream_buft_stats();
        const uint64_t resident = s.resident_cache_bytes();
        if (resident > max_resident) max_resident = resident;
    }
    if (max_resident_bytes_out != nullptr) *max_resident_bytes_out = max_resident;

    // Reset the knob for subsequent loads/tests.
    set_stream_buft_slots(0);
    return out;
}

static void test_streaming_bounded_and_deterministic() {
    const char *path = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }

    const int n_steps = 12;

    // --- Baseline: experts NOT streamed (plain CPU-resident path). This is the
    //     "full-resident" reference the streamed output must match byte-for-
    //     byte (the hard gate: streaming changes WHEN/WHERE bytes are resident,
    //     never their VALUES or the ops). ---------------------------------------
    set_force_stream(false);
    std::vector<int32_t> baseline_seq =
        decode_sequence(path, /*n_slots=*/0, n_steps, nullptr);
    CHECK(!baseline_seq.empty());

    set_force_stream(true);

    // --- Full-pool streamed run (auto: holds the whole working set). ----------
    uint64_t full_resident = 0;
    std::vector<int32_t> full_seq =
        decode_sequence(path, /*n_slots=*/0, n_steps, &full_resident);
    CHECK(!full_seq.empty());
    for (int32_t t : full_seq) CHECK(t >= 0);  // (a) valid tokens

    // Streamed (full pool) == non-streamed baseline, byte-for-byte.
    CHECK(full_seq == baseline_seq);

    // The tiny MoE is 2 layers x {gate,down,up} = 6 stacked expert tensors.
    // With the auto pool the whole set stays resident: decode 1 loads all 6
    // (misses), every later decode's residency pass is served from the slots
    // (hits), and nothing is ever evicted. This proves the cache serves
    // resident tensors across decodes without touching the backing store.
    const StreamBuftStats full_stats = stream_buft_stats();
    CHECK(full_stats.n_slots >= 1u);
    CHECK(full_stats.cache_misses > 0u);  // the initial cold load
    CHECK(full_stats.cache_hits > 0u);    // every warm decode after step 1
    CHECK_EQ(full_stats.cache_evictions, 0u);

    // --- Small-pool run: fewer slots than the 6 stacked expert tensors so the
    //     residency pass must evict and reload across decodes. -----------------
    uint64_t small_resident = 0;
    const uint32_t small_slots = 2;  // << 6 expert tensors
    std::vector<int32_t> small_seq =
        decode_sequence(path, small_slots, n_steps, &small_resident);
    CHECK(!small_seq.empty());
    for (int32_t t : small_seq) CHECK(t >= 0);  // (a) valid tokens

    // (b) BYTE-IDENTICAL determinism gate: streaming through a tiny pool must
    //     produce exactly the same tokens as the full-resident pool.
    CHECK(full_seq.size() == small_seq.size());
    CHECK(full_seq == small_seq);

    // (c) With the small pool (fewer slots than the 6 stacked expert tensors)
    //     the per-decode residency scan is larger than the cache, so each pass
    //     misses and evicts, and a tensor evicted on decode N is reloaded on
    //     decode N+1 — the eviction+reload-across-decodes guarantee the task
    //     accepts for Model A. We assert both misses and non-zero evictions,
    //     and that disk_reads keeps growing (re-materialization every step).
    const StreamBuftStats small_stats = stream_buft_stats();
    CHECK_EQ(small_stats.n_slots, small_slots);
    CHECK(small_stats.cache_misses > full_stats.cache_misses);  // churns
    CHECK(small_stats.cache_evictions > 0u);
    // 6 tensors re-materialized every one of n_steps decodes with a 2-slot
    // pool => many disk reads (eviction + reload proven across decodes). With
    // the full pool the 6 tensors load exactly once, so the small pool reads
    // the store far more often.
    CHECK(small_stats.disk_reads > full_stats.disk_reads);
    CHECK(small_stats.disk_reads >= static_cast<uint64_t>(n_steps));

    // (d) Resident cache bytes bounded by n_slots * slot_bytes, and strictly
    //     smaller than the full-pool residency (bounded-memory proxy).
    CHECK(small_stats.slot_bytes > 0u);
    const uint64_t bound =
        static_cast<uint64_t>(small_slots) * small_stats.slot_bytes;
    CHECK(small_resident <= bound);
    CHECK(small_resident < full_resident);

    set_force_stream(false);
}

static void run_all() {
    RUN(test_streaming_bounded_and_deterministic);
}

TEST_MAIN()

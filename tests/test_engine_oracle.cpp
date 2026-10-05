// Engine oracle correctness gate (docs/ENGINE_CORE_DESIGN.md section 7.3): on
// the SAME tiny MoE gguf, run both the libllama oracle (llama_decode) and OUR
// OWN ggml engine and assert they agree.
//   EC-1 case (single token at position 0):
//     (a) identical greedy argmax token id  -- the HARD gate, and
//     (b) max absolute logit delta within tolerance -- the soft gate.
//   EC-2 case (multi-token prompt + N greedy generated tokens over the
//   engine's own KV cache):
//     (a) identical greedy token-id SEQUENCE -- the HARD gate, and
//     (b) max per-step absolute logit delta within tolerance -- the soft gate.
//   EC-3 case (per-layer SEGMENTED execution with the top-k residency SlotPool
//   bounded BELOW the expert count):
//     (a) the token SEQUENCE is byte-identical to the fully-resident engine AND
//         to the oracle -- the HARD gate,
//     (b) the bounded pool shows hits AND misses AND non-zero evictions
//         (experts streamed/evicted/reloaded), and
//     (c) resident expert bytes stay bounded by the pool size (< full).
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
#include "hw/profiler.h"
#include "model/engine/engine.h"
#include "plan/planner.h"
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

// EC-2 sequence gate (docs/ENGINE_CORE_DESIGN.md 7.3): tokenize a real prompt
// through the model vocab, then greedily generate N tokens through BOTH the
// libllama oracle (llama_decode over the prompt then N steps) and OUR engine
// (prefill the prompt then N decode steps over its own KV cache) on the SAME
// gguf. The produced greedy token-id SEQUENCES must be IDENTICAL (hard gate);
// per-step max |logit delta| stays within a 1e-3 soft tolerance.
static void test_engine_sequence_matches_oracle() {
    const char *path = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }

    const std::string prompt = "hello world";
    const int n_generate = 8;

    // Oracle: prompt + N greedy steps, capturing per-step logits.
    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    CHECK(engine::run_oracle_sequence(path, prompt, n_generate, oracle_tokens,
                                      &oracle_step_logits));
    CHECK_EQ(oracle_tokens.size(), static_cast<size_t>(n_generate));
    if (oracle_tokens.size() != static_cast<size_t>(n_generate)) return;

    // Same prompt tokens the oracle used (add_special=true), via the vocab.
    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(path, prompt, prompt_ids));
    CHECK(!prompt_ids.empty());

    std::unique_ptr<engine::Engine> eng = engine::Engine::load(path);
    CHECK(eng != nullptr);
    if (eng == nullptr) return;
    eng->reset_kv();

    // Engine: prefill the prompt (positions 0..P-1), then greedily decode N
    // tokens, feeding each generated token back at the next position.
    std::vector<int32_t> engine_tokens;
    std::vector<std::vector<float>> engine_step_logits;
    int32_t pos = 0;
    std::vector<float> logits;

    // Prefill: process every prompt token; only the final step's logits drive
    // the first generated token.
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        const bool ok = eng->forward(prompt_ids[i], pos, logits);
        CHECK(ok);
        if (!ok) return;
        ++pos;
    }

    // Decode: argmax -> next token -> feed back, N times.
    for (int g = 0; g < n_generate; ++g) {
        CHECK(!logits.empty());
        if (logits.empty()) return;
        engine_step_logits.push_back(logits);
        const int32_t next = argmax(logits);
        engine_tokens.push_back(next);
        const bool ok = eng->forward(next, pos, logits);
        CHECK(ok);
        if (!ok) return;
        ++pos;
    }

    // (a) HARD gate: identical greedy token-id sequences.
    CHECK_EQ(engine_tokens.size(), oracle_tokens.size());
    bool seq_match = engine_tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < engine_tokens.size() &&
                       i < oracle_tokens.size(); ++i) {
        CHECK_EQ(engine_tokens[i], oracle_tokens[i]);
        if (engine_tokens[i] != oracle_tokens[i]) seq_match = false;
    }

    // (b) soft gate: per-step max |logit delta| within tolerance.
    double max_abs = 0.0;
    const size_t steps =
        engine_step_logits.size() < oracle_step_logits.size()
            ? engine_step_logits.size()
            : oracle_step_logits.size();
    for (size_t s = 0; s < steps; ++s) {
        const auto &e = engine_step_logits[s];
        const auto &o = oracle_step_logits[s];
        const size_t n = e.size() < o.size() ? e.size() : o.size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(e[i]) -
                                       static_cast<double>(o[i]));
            if (d > max_abs) max_abs = d;
        }
    }

    std::printf("[seq match=%s n=%d max per-step |delta|=%.3e tokens:",
                seq_match ? "yes" : "no", n_generate, max_abs);
    for (size_t i = 0; i < engine_tokens.size(); ++i) {
        std::printf(" %d", engine_tokens[i]);
    }
    std::printf("] ");

    CHECK(max_abs < 1e-3);
}

// Decode a prompt + N greedy tokens through an engine loaded with `slots`
// top-k residency slots, returning the token sequence and the final cache
// stats. Shared by the EC-3 bounded-pool test.
static std::vector<int32_t> engine_decode_sequence(
    const char *path, const std::string &prompt, int n_generate,
    uint32_t slots, engine::EngineCacheStats &out_stats,
    std::vector<std::vector<float>> &out_step_logits) {
    std::vector<int32_t> tokens;
    std::vector<int32_t> prompt_ids;
    if (!engine::vocab_tokenize(path, prompt, prompt_ids)) return tokens;

    std::unique_ptr<engine::Engine> eng = engine::Engine::load(path, slots);
    if (eng == nullptr) return tokens;
    eng->reset_kv();

    int32_t pos = 0;
    std::vector<float> logits;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        if (!eng->forward(prompt_ids[i], pos, logits)) return tokens;
        ++pos;
    }
    for (int g = 0; g < n_generate; ++g) {
        if (logits.empty()) return tokens;
        out_step_logits.push_back(logits);
        const int32_t next = argmax(logits);
        tokens.push_back(next);
        if (!eng->forward(next, pos, logits)) return tokens;
        ++pos;
    }
    out_stats = eng->cache_stats();
    return tokens;
}

// EC-3 gate (docs/ENGINE_CORE_DESIGN.md 3.2/3.3, section 8 EC-3): per-layer
// SEGMENTED execution with the top-k residency SlotPool bounded BELOW the
// expert count must produce a token SEQUENCE byte-identical to the fully-
// resident engine AND to the oracle, while the pool shows hits+misses+evictions
// (experts streamed/evicted/reloaded) and resident expert bytes stay bounded by
// the pool size (not the full expert footprint).
static void test_engine_bounded_pool_byte_identical() {
    const char *path = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }

    const std::string prompt = "hello world";
    const int n_generate = 8;

    // Oracle sequence (the hard gate the engine must reproduce).
    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    CHECK(engine::run_oracle_sequence(path, prompt, n_generate, oracle_tokens,
                                      &oracle_step_logits));
    CHECK_EQ(oracle_tokens.size(), static_cast<size_t>(n_generate));
    if (oracle_tokens.size() != static_cast<size_t>(n_generate)) return;

    // Fully-resident engine (auto pool: holds the whole expert working set, no
    // eviction) and a BOUNDED engine (fewer slots than n_expert per layer, so
    // experts must be evicted and reloaded across layers/tokens).
    engine::EngineCacheStats full_stats{};
    engine::EngineCacheStats bnd_stats{};
    std::vector<std::vector<float>> full_logits;
    std::vector<std::vector<float>> bnd_logits;
    std::vector<int32_t> full_tokens = engine_decode_sequence(
        path, prompt, n_generate, /*slots=*/0, full_stats, full_logits);
    // 3 slots: below the fixture's 8 experts/layer (forces eviction) but >=
    // n_expert_used=2 so a single layer's top-k always fits.
    std::vector<int32_t> bnd_tokens = engine_decode_sequence(
        path, prompt, n_generate, /*slots=*/3, bnd_stats, bnd_logits);

    CHECK_EQ(full_tokens.size(), static_cast<size_t>(n_generate));
    CHECK_EQ(bnd_tokens.size(), static_cast<size_t>(n_generate));

    // (a) HARD gate: bounded-pool sequence == fully-resident sequence == oracle.
    bool seq_match = bnd_tokens.size() == oracle_tokens.size() &&
                     full_tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < bnd_tokens.size() && i < oracle_tokens.size(); ++i) {
        CHECK_EQ(bnd_tokens[i], oracle_tokens[i]);
        CHECK_EQ(bnd_tokens[i], full_tokens[i]);
        if (bnd_tokens[i] != oracle_tokens[i] ||
            bnd_tokens[i] != full_tokens[i]) {
            seq_match = false;
        }
    }

    // Logits byte-identical between the bounded and fully-resident engine (same
    // bytes, same ops: streaming only changes WHEN experts become resident).
    double max_fb = 0.0;
    const size_t steps = full_logits.size() < bnd_logits.size()
                             ? full_logits.size()
                             : bnd_logits.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = full_logits[s].size() < bnd_logits[s].size()
                             ? full_logits[s].size()
                             : bnd_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(full_logits[s][i]) -
                                       static_cast<double>(bnd_logits[s][i]));
            if (d > max_fb) max_fb = d;
        }
    }

    std::printf("[seq match=%s full-vs-bounded max|delta|=%.3e "
                "bounded(hits=%llu miss=%llu evict=%llu resident=%lluB full=%lluB) "
                "tokens:",
                seq_match ? "yes" : "no", max_fb,
                static_cast<unsigned long long>(bnd_stats.hits),
                static_cast<unsigned long long>(bnd_stats.misses),
                static_cast<unsigned long long>(bnd_stats.evictions),
                static_cast<unsigned long long>(bnd_stats.resident_bytes()),
                static_cast<unsigned long long>(bnd_stats.full_bytes));
    for (size_t i = 0; i < bnd_tokens.size(); ++i) {
        std::printf(" %d", bnd_tokens[i]);
    }
    std::printf("] ");

    CHECK(seq_match);
    // Bounded and fully-resident engine agree bit-for-bit (same ops).
    CHECK(max_fb == 0.0);

    // (b) cache was exercised: the bounded pool shows hits AND misses AND
    // non-zero evictions, proving experts were streamed/evicted/reloaded.
    CHECK(bnd_stats.hits > 0);
    CHECK(bnd_stats.misses > 0);
    CHECK(bnd_stats.evictions > 0);

    // (c) resident expert bytes are bounded by the pool size and strictly below
    // the fully-resident footprint.
    CHECK(bnd_stats.resident_bytes() < bnd_stats.full_bytes);
    CHECK(bnd_stats.resident_bytes() ==
          bnd_stats.n_slots * bnd_stats.slot_bytes);
    // The fully-resident (auto) engine holds the whole working set and never
    // evicts.
    CHECK_EQ(full_stats.evictions, static_cast<uint64_t>(0));
    CHECK_EQ(full_stats.resident_bytes(), full_stats.full_bytes);
}

// EC-6: predictor-driven prefetch warms the pool ahead of each layer. Assert
// that (a) decode is still byte-identical to the oracle (prefetch must never
// change results), and (b) the prefetch metric is live and effective: after a
// few tokens the predictor warms experts and a meaningful fraction are then
// actually used (the statistical predictor converges on the fixture's routing).
static void test_engine_prefetch_hit_rate() {
    const char *path = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }

    const std::string prompt = "hello world";
    const int n_generate = 12;

    std::vector<int32_t> oracle_tokens;
    CHECK(engine::run_oracle_sequence(path, prompt, n_generate, oracle_tokens,
                                      nullptr));

    // Bounded pool so prefetch has something to warm (an auto pool goes fully
    // resident after token 1 and never needs warming).
    engine::EngineCacheStats st{};
    std::vector<std::vector<float>> logits;
    std::vector<int32_t> toks =
        engine_decode_sequence(path, prompt, n_generate, /*slots=*/3, st, logits);

    // (a) HARD gate: prefetch did not change the output.
    bool seq_match = toks.size() == oracle_tokens.size();
    for (size_t i = 0; i < toks.size() && i < oracle_tokens.size(); ++i) {
        CHECK_EQ(toks[i], oracle_tokens[i]);
        if (toks[i] != oracle_tokens[i]) seq_match = false;
    }

    std::printf("[seq match=%s prefetch(warmed=%llu used=%llu hit_rate=%.2f)] ",
                seq_match ? "yes" : "no",
                static_cast<unsigned long long>(st.prefetch_warmed),
                static_cast<unsigned long long>(st.prefetch_used),
                st.prefetch_hit_rate());

    CHECK(seq_match);
    // (b) the predictor warmed experts and some were used. We assert warming
    // happened and the hit rate is non-trivial; we do NOT assert a high bar
    // (the statistical predictor is intentionally simple and the fixture has
    // random weights), only that the prefetch path is live and useful.
    CHECK(st.prefetch_warmed > 0);
    CHECK(st.prefetch_used > 0);
    CHECK(st.prefetch_hit_rate() > 0.0);
}

// EC-7: a DENSE llama model (no experts; plain gate/up/down FFN) decodes
// through the engine's dense path and matches the oracle token sequence. Proves
// the arch generalization beyond MoE. Guarded on STRATAFLOW_TEST_DENSE_GGUF
// (generate with tools/testdata/make_tiny_dense_gguf.py).
static void test_engine_dense_matches_oracle() {
    const char *path = std::getenv("STRATAFLOW_TEST_DENSE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_DENSE_GGUF] ");
        return;
    }

    const std::string prompt = "hello world";
    const int n_generate = 8;

    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    CHECK(engine::run_oracle_sequence(path, prompt, n_generate, oracle_tokens,
                                      &oracle_step_logits));

    engine::EngineCacheStats st{};
    std::vector<std::vector<float>> eng_logits;
    std::vector<int32_t> toks =
        engine_decode_sequence(path, prompt, n_generate, /*slots=*/0, st, eng_logits);

    bool seq_match = toks.size() == oracle_tokens.size();
    double max_delta = 0.0;
    const size_t steps = oracle_step_logits.size() < eng_logits.size()
                             ? oracle_step_logits.size()
                             : eng_logits.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = oracle_step_logits[s].size() < eng_logits[s].size()
                             ? oracle_step_logits[s].size()
                             : eng_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(oracle_step_logits[s][i]) -
                                       static_cast<double>(eng_logits[s][i]));
            if (d > max_delta) max_delta = d;
        }
    }
    for (size_t i = 0; i < toks.size() && i < oracle_tokens.size(); ++i) {
        CHECK_EQ(toks[i], oracle_tokens[i]);
        if (toks[i] != oracle_tokens[i]) seq_match = false;
    }

    std::printf("[dense seq match=%s n=%d max per-step |delta|=%.3e tokens:",
                seq_match ? "yes" : "no", n_generate, max_delta);
    for (size_t i = 0; i < toks.size(); ++i) std::printf(" %d", toks[i]);
    std::printf("] ");

    CHECK(seq_match);
    CHECK(max_delta < 1e-3);
}

// Quantized-weight gate: a Q8_0 llama-arch MoE must decode through the engine's
// quantized path and match the libllama oracle sequence. Proves the engine runs
// quantized models, not just F32. Guarded on STRATAFLOW_TEST_QUANT_MOE_GGUF
// (generate with tools/testdata/make_tiny_quant_moe_gguf.py).
static void test_engine_quant_matches_oracle() {
    const char *path = std::getenv("STRATAFLOW_TEST_QUANT_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_QUANT_MOE_GGUF] ");
        return;
    }

    const std::string prompt = "hello world";
    const int n_generate = 8;

    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    CHECK(engine::run_oracle_sequence(path, prompt, n_generate, oracle_tokens,
                                      &oracle_step_logits));

    engine::EngineCacheStats st{};
    std::vector<std::vector<float>> eng_logits;
    std::vector<int32_t> toks =
        engine_decode_sequence(path, prompt, n_generate, /*slots=*/0, st, eng_logits);

    bool seq_match = toks.size() == oracle_tokens.size();
    double max_delta = 0.0;
    const size_t steps = oracle_step_logits.size() < eng_logits.size()
                             ? oracle_step_logits.size()
                             : eng_logits.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = oracle_step_logits[s].size() < eng_logits[s].size()
                             ? oracle_step_logits[s].size()
                             : eng_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(oracle_step_logits[s][i]) -
                                       static_cast<double>(eng_logits[s][i]));
            if (d > max_delta) max_delta = d;
        }
    }
    for (size_t i = 0; i < toks.size() && i < oracle_tokens.size(); ++i) {
        CHECK_EQ(toks[i], oracle_tokens[i]);
        if (toks[i] != oracle_tokens[i]) seq_match = false;
    }

    std::printf("[quant(Q8_0) seq match=%s n=%d max per-step |delta|=%.3e tokens:",
                seq_match ? "yes" : "no", n_generate, max_delta);
    for (size_t i = 0; i < toks.size(); ++i) std::printf(" %d", toks[i]);
    std::printf("] ");

    // The token SEQUENCE equality is the hard gate. The logit delta tolerance
    // is looser than the F32 tests on purpose: Q8_0 weights go through block
    // dequant, so small per-logit differences vs the oracle's own dequant path
    // are expected (observed ~2.5e-3); what must match is the greedy argmax.
    CHECK(seq_match);
    CHECK(max_delta < 2e-2);
}

// K-quant gate: a Q4_K / Q6_K llama-arch MoE must decode through the engine's
// quantized path and match the libllama oracle sequence. Real downloaded models
// are almost always K-quants, so this proves the engine runs the quant family a
// real model uses, including quantized TRUNK tensors (token_embd/output/attn at
// a K-quant type). The engine stages experts at the SOURCE tensor's real ggml
// type, and K-quant inner dims are multiples of the 256 superblock, so per-
// expert nb02 slices stay whole-block and the type-agnostic staging path works
// unchanged. `label` names the type for the log line. Mirrors the Q8_0 test.
// `soft_bound` is a per-type calibrated ceiling on the accumulated per-step
// max |logit delta| vs the oracle (derived from the observed deltas; see the
// comment at the CHECK below), tight enough to trip a real numerical drift.
static void run_kquant_oracle(const char *path, const char *label,
                              double soft_bound) {
    const std::string prompt = "hello world";
    const int n_generate = 8;

    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    CHECK(engine::run_oracle_sequence(path, prompt, n_generate, oracle_tokens,
                                      &oracle_step_logits));

    engine::EngineCacheStats st{};
    std::vector<std::vector<float>> eng_logits;
    std::vector<int32_t> toks =
        engine_decode_sequence(path, prompt, n_generate, /*slots=*/0, st, eng_logits);

    bool seq_match = toks.size() == oracle_tokens.size();
    double max_delta = 0.0;
    const size_t steps = oracle_step_logits.size() < eng_logits.size()
                             ? oracle_step_logits.size()
                             : eng_logits.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = oracle_step_logits[s].size() < eng_logits[s].size()
                             ? oracle_step_logits[s].size()
                             : eng_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(oracle_step_logits[s][i]) -
                                       static_cast<double>(eng_logits[s][i]));
            if (d > max_delta) max_delta = d;
        }
    }
    for (size_t i = 0; i < toks.size() && i < oracle_tokens.size(); ++i) {
        CHECK_EQ(toks[i], oracle_tokens[i]);
        if (toks[i] != oracle_tokens[i]) seq_match = false;
    }

    std::printf("[quant(%s) seq match=%s n=%d max per-step |delta|=%.3e tokens:",
                label, seq_match ? "yes" : "no", n_generate, max_delta);
    for (size_t i = 0; i < toks.size(); ++i) std::printf(" %d", toks[i]);
    std::printf("] ");

    // Hard gate: identical greedy token SEQUENCE vs the oracle. This is the
    // real correctness proof and it holds exactly for Q4_K and Q6_K.
    //
    // Soft gate: the accumulated per-step max |logit delta| vs the oracle, with
    // a PER-TYPE calibrated bound (passed in via `soft_bound`) rather than one
    // loose shared ceiling. The engine and the oracle both run the SAME quant
    // dequant path, so this delta measures only the small residual between our
    // ggml forward pass and libllama's on identical weights; it is NOT the
    // engine-vs-F32 dequant error. Measured accumulated max |delta| over the 8
    // autoregressive steps is Q4_K ~2.0e-2 and Q6_K ~1.85e-1 (the per-fixture
    // numbers are not comparable across types: each is diffed only against its
    // OWN oracle run on its OWN random-weight fixture, so where the logits land
    // relative to near-tied vocab entries dominates). The caller sets a bound a
    // small multiple above the observed value for that type (Q4_K 5e-2, Q6_K
    // 2.5e-1), tight enough that a real numerical regression that nudges the
    // delta without yet flipping the argmax would trip it, while the SEQUENCE
    // equality above remains the primary correctness gate.
    CHECK(seq_match);
    CHECK(max_delta < soft_bound);
}

// Guarded on STRATAFLOW_TEST_KQUANT_MOE_GGUF (Q4_K) and the optional
// STRATAFLOW_TEST_Q6K_MOE_GGUF (Q6_K). Skips cleanly when neither is set, same
// pattern as the other guarded tests.
static void test_engine_kquant_matches_oracle() {
    const char *q4k = std::getenv("STRATAFLOW_TEST_KQUANT_MOE_GGUF");
    const char *q6k = std::getenv("STRATAFLOW_TEST_Q6K_MOE_GGUF");
    if ((q4k == nullptr || q4k[0] == '\0') &&
        (q6k == nullptr || q6k[0] == '\0')) {
        std::printf("[skipped: set STRATAFLOW_TEST_KQUANT_MOE_GGUF] ");
        return;
    }
    // Per-type soft bounds: a small multiple above the observed accumulated
    // per-step max |delta| for each type (Q4_K ~2.0e-2, Q6_K ~1.85e-1).
    if (q4k != nullptr && q4k[0] != '\0') run_kquant_oracle(q4k, "Q4_K", 5e-2);
    if (q6k != nullptr && q6k[0] != '\0') run_kquant_oracle(q6k, "Q6_K", 2.5e-1);
}

// AUTO-path oracle gate (FEAT-002): the RAM-aware slot count the PLANNER picks
// must sit on the oracle gate exactly like the fully-resident and fixed-slot
// runs. Decode the SAME prompt + N tokens three ways:
//   (a) fully-resident (Engine::load(path, 0)),
//   (b) a fixed small slot count (3, below the fixture's 8 experts/layer),
//   (c) the AUTO slot count plan_placement() chooses for a representative
//       hardware profile on the fixture's own ModelShape.
// The token SEQUENCE and per-step logits must be byte-identical across (a),
// (b), (c) and match the libllama oracle (F32 tol < 1e-3). This proves the auto
// policy changes only HOW MANY/WHEN experts are resident, never which bytes
// compute.
static void test_engine_auto_slots_matches_oracle() {
    const char *path = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }

    const std::string prompt = "hello world";
    const int n_generate = 8;

    // Oracle sequence (the hard gate all three engine runs must reproduce).
    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    CHECK(engine::run_oracle_sequence(path, prompt, n_generate, oracle_tokens,
                                      &oracle_step_logits));
    CHECK_EQ(oracle_tokens.size(), static_cast<size_t>(n_generate));
    if (oracle_tokens.size() != static_cast<size_t>(n_generate)) return;

    // Compute the AUTO slot count the planner picks for this fixture's shape.
    // The tiny MoE is 2 layers x 8 experts, top-2. Under a tiny forced
    // expert-RAM cap the policy must choose a BOUNDED count (< full_slots=16,
    // >= n_experts_used=2) so the auto path actually exercises eviction here.
    sf::ModelShape shape;
    shape.name = "tiny-moe-fixture";
    shape.n_layers = 2;
    shape.is_moe = true;
    shape.n_experts = 8;
    shape.n_experts_used = 2;
    shape.trunk_bytes = 1 * 1024 * 1024;
    shape.expert_bytes = 1 * 1024 * 1024;  // 1 MiB per bundle (shape-only sizing)
    shape.total_bytes = shape.trunk_bytes +
        static_cast<uint64_t>(shape.n_experts) * shape.n_layers * shape.expert_bytes;

    sf::HardwareProfile hw;
    hw.ram_total_bytes = 16ull * 1024 * 1024 * 1024;
    hw.ram_free_bytes = 14ull * 1024 * 1024 * 1024;
    // Cap expert RAM to ~4 bundles so the auto choice is bounded below the
    // 16-bundle working set (forces eviction, like a small machine would).
    const uint64_t cache_budget = 4 * shape.expert_bytes;
    sf::PlacementPlan plan =
        sf::plan_placement(hw, shape, /*vram=*/0, /*ram=*/0, cache_budget);
    const uint32_t full_slots =
        static_cast<uint32_t>(uint64_t(shape.n_layers) * shape.n_experts);
    const uint32_t auto_slots = plan.expert_slots_resident;
    CHECK(auto_slots > 0);
    CHECK(auto_slots < full_slots);                 // bounded for this profile
    CHECK(auto_slots >= shape.n_experts_used);       // top-k always fits
    CHECK(plan.stream_experts);                      // eviction will happen

    // Decode three ways: (a) fully resident, (b) fixed small, (c) auto.
    engine::EngineCacheStats full_stats{}, fixed_stats{}, auto_stats{};
    std::vector<std::vector<float>> full_logits, fixed_logits, auto_logits;
    std::vector<int32_t> full_tokens = engine_decode_sequence(
        path, prompt, n_generate, /*slots=*/0, full_stats, full_logits);
    std::vector<int32_t> fixed_tokens = engine_decode_sequence(
        path, prompt, n_generate, /*slots=*/3, fixed_stats, fixed_logits);
    std::vector<int32_t> auto_tokens = engine_decode_sequence(
        path, prompt, n_generate, auto_slots, auto_stats, auto_logits);

    CHECK_EQ(full_tokens.size(), static_cast<size_t>(n_generate));
    CHECK_EQ(fixed_tokens.size(), static_cast<size_t>(n_generate));
    CHECK_EQ(auto_tokens.size(), static_cast<size_t>(n_generate));

    // (a) HARD gate: auto sequence == fixed == full == oracle.
    bool seq_match = auto_tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < auto_tokens.size() && i < oracle_tokens.size(); ++i) {
        CHECK_EQ(auto_tokens[i], oracle_tokens[i]);
        CHECK_EQ(auto_tokens[i], full_tokens[i]);
        CHECK_EQ(auto_tokens[i], fixed_tokens[i]);
        if (auto_tokens[i] != oracle_tokens[i] ||
            auto_tokens[i] != full_tokens[i] ||
            auto_tokens[i] != fixed_tokens[i]) {
            seq_match = false;
        }
    }

    // (b) auto logits byte-identical to the fully-resident run (same bytes,
    // same ops; the slot count changes only WHEN experts become resident).
    double max_af = 0.0;
    const size_t steps = full_logits.size() < auto_logits.size()
                             ? full_logits.size()
                             : auto_logits.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = full_logits[s].size() < auto_logits[s].size()
                             ? full_logits[s].size()
                             : auto_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(full_logits[s][i]) -
                                       static_cast<double>(auto_logits[s][i]));
            if (d > max_af) max_af = d;
        }
    }

    // (c) auto logits vs the oracle within the F32 tolerance.
    double max_ao = 0.0;
    const size_t osteps = oracle_step_logits.size() < auto_logits.size()
                              ? oracle_step_logits.size()
                              : auto_logits.size();
    for (size_t s = 0; s < osteps; ++s) {
        const size_t n = oracle_step_logits[s].size() < auto_logits[s].size()
                             ? oracle_step_logits[s].size()
                             : auto_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(
                static_cast<double>(oracle_step_logits[s][i]) -
                static_cast<double>(auto_logits[s][i]));
            if (d > max_ao) max_ao = d;
        }
    }

    std::printf("[auto slots=%u (full=%u) seq match=%s auto-vs-full|delta|=%.3e "
                "auto-vs-oracle|delta|=%.3e auto(hits=%llu miss=%llu evict=%llu)] ",
                auto_slots, full_slots, seq_match ? "yes" : "no", max_af, max_ao,
                static_cast<unsigned long long>(auto_stats.hits),
                static_cast<unsigned long long>(auto_stats.misses),
                static_cast<unsigned long long>(auto_stats.evictions));

    CHECK(seq_match);
    CHECK(max_af == 0.0);     // auto vs fully-resident: bit-for-bit identical
    CHECK(max_ao < 1e-3);     // auto vs oracle: within F32 tolerance
    // The bounded auto pool actually evicted (it is below the working set).
    CHECK(auto_stats.evictions > 0);
    CHECK(auto_stats.resident_bytes() < auto_stats.full_bytes);
}

static void run_all() {
    RUN(test_engine_matches_oracle);
    RUN(test_engine_sequence_matches_oracle);
    RUN(test_engine_bounded_pool_byte_identical);
    RUN(test_engine_auto_slots_matches_oracle);
    RUN(test_engine_prefetch_hit_rate);
    RUN(test_engine_dense_matches_oracle);
    RUN(test_engine_quant_matches_oracle);
    RUN(test_engine_kquant_matches_oracle);
}

TEST_MAIN()

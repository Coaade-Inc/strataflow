// FEAT-003 gate (the real-Mixtral legacy per-expert fix): a GGUF that names its
// routed-expert FFN weights with the LEGACY, UN-STACKED, PER-EXPERT convention
// (blk.N.ffn_{gate,down,up}.E.weight - what the real TheBloke Mixtral-8x7B GGUFs
// use) must now PACK to a .strata whose experts were GROUPED into the stacked
// per-(layer,expert) expert region, with the embedded GGUF metadata SYNTHESIZED
// to declare stacked blk.N.ffn_{gate,down,up}_exps.weight 3-D tensors. The
// engine (which reads only that synthesized metadata, never the raw legacy
// names) then streams and decodes the .strata byte-identically to the libllama
// ORACLE run on the matched STACKED-equivalent GGUF (whose per-expert slices are
// byte-identical to the legacy standalone tensors; see FEAT-002).
//
// Confirmed finding (FEAT-002): vendored llama.cpp b11379 does NOT stack legacy
// per-expert tensors at load; it only ever looks for the stacked _exps tensors.
// So run_oracle_sequence LOADS the stacked fixture but REJECTS the legacy one -
// the oracle therefore runs on the STACKED fixture, and the engine runs on the
// .strata PACKED FROM THE LEGACY fixture. Equal sequences + bounded logit delta
// is the proof the packer-only grouping + metadata synthesis is faithful.
//
// GUARDED on STRATAFLOW_TEST_LEGACY_MOE_GGUF (the legacy fixture) and
// STRATAFLOW_TEST_LEGACY_MOE_STACKED_GGUF (the matched stacked oracle fixture),
// so the suite stays green in CI where the fixtures are not generated. Generate
// + run locally (one shell line each):
//   build/debug/bin/make-legacy-moe-gguf FIX.gguf FIX.stacked.gguf
//   STRATAFLOW_TEST_LEGACY_MOE_GGUF=FIX.gguf
//   STRATAFLOW_TEST_LEGACY_MOE_STACKED_GGUF=FIX.stacked.gguf
//   ctest --preset debug -R test_legacy_per_expert_pack
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier:
// LicenseRef-Coaade-Source-Available-1.0
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "model/engine/engine.h"
#include "test_util.h"
#include "tws/async_io.h"
#include "tws/strata_file.h"
#include "tws/strata_format.h"
#include "tws/strata_pack.h"

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

// Greedily decode `prompt_ids` + `n_generate` tokens through an engine loaded
// from `model_path` with `slots` top-k residency slots. Fills `out_logits` with
// the per-step logit vectors and `out_stats` with the final cache stats.
// Mirrors tests/test_engine_oracle.cpp engine_decode_sequence().
static std::vector<int32_t> engine_decode(const std::string &model_path,
                                          const std::vector<int32_t> &prompt_ids,
                                          int n_generate, uint32_t slots,
                                          std::vector<std::vector<float>> &out_logits,
                                          engine::EngineCacheStats &out_stats,
                                          uint64_t &out_streamed) {
    std::vector<int32_t> tokens;
    std::unique_ptr<engine::Engine> eng = engine::Engine::load(model_path, slots);
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
        out_logits.push_back(logits);
        const int32_t next = argmax(logits);
        tokens.push_back(next);
        if (!eng->forward(next, pos, logits)) return tokens;
        ++pos;
    }
    out_stats = eng->cache_stats();
    out_streamed = eng->streamed_bytes();
    return tokens;
}

// Core gate for ONE (legacy, stacked) fixture pair at a given K-quant tolerance.
static void run_legacy_pair(const char *legacy_env, const char *stacked_env, double tol,
                            const char *label) {
    const char *legacy = std::getenv(legacy_env);
    const char *stacked = std::getenv(stacked_env);
    if (legacy == nullptr || legacy[0] == '\0' || stacked == nullptr ||
        stacked[0] == '\0') {
        std::printf("[skipped %s: set %s and %s] ", label, legacy_env, stacked_env);
        return;
    }
    const std::string legacy_path = legacy;
    const std::string stacked_path = stacked;
    const std::string strata_path = legacy_path + ".legacy.strata";

    // 1. Pack the LEGACY per-expert GGUF -> .strata. BEFORE the fix this failed
    // loudly ("zero streamable expert tensors ... Tensor names seen:
    // 'blk.0.ffn_gate.0.weight' ..."); AFTER the fix the packer GROUPS the
    // per-expert tensors into the stacked expert region and synthesizes stacked
    // metadata, so the pack succeeds.
    std::string err;
    const bool packed = pack_gguf_to_strata(legacy_path, strata_path, &err);
    CHECK(packed);
    if (!packed) {
        std::printf("[%s pack failed: %s] ", label, err.c_str());
        return;
    }

    // 2. The superblock must report the experts as GROUPED, not dropped to
    // trunk: the fixture defaults are n_layer=2, n_expert=8, so index_count ==
    // n_layers * n_experts == 16. A 0-expert pack would report 0/0/0 here.
    uint32_t n_experts = 0;
    {
        StrataReader reader;
        CHECK(reader.open(strata_path));
        const StrataSuperblock &sb = reader.superblock();
        n_experts = sb.n_experts;
        std::printf("[%s superblock: %u layers, %u experts, %llu index] ", label,
                    sb.n_layers, sb.n_experts,
                    static_cast<unsigned long long>(reader.index_count()));
        CHECK_EQ(sb.n_layers, static_cast<uint32_t>(2));
        CHECK_EQ(sb.n_experts, static_cast<uint32_t>(8));
        CHECK_EQ(reader.index_count(), static_cast<uint64_t>(16));
    }

    const std::string prompt = "the city of";
    const int n_generate = 8;

    // Tokenize once from the STACKED fixture's vocab (identical tokenizer in
    // both fixtures) so every decode path gets the same prompt ids.
    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(stacked_path, prompt, prompt_ids));
    CHECK(!prompt_ids.empty());

    // 3. Engine decode the legacy-sourced .strata, fully resident (slots=0),
    // AND the matched stacked GGUF through the SAME engine. Because the packer
    // regroups the legacy per-expert bytes VERBATIM into the stacked expert
    // region (the stacked slices are byte-identical to the legacy standalone
    // tensors; FEAT-002), these two engine runs read the SAME bytes with the
    // SAME ops, so their logits must be bit-for-bit identical (max|delta|==0).
    std::vector<std::vector<float>> strata_logits;
    std::vector<std::vector<float>> stacked_logits;
    engine::EngineCacheStats full_stats{};
    engine::EngineCacheStats stacked_stats{};
    uint64_t full_streamed = 0;
    uint64_t stacked_streamed = 0;
    std::vector<int32_t> strata_tokens =
        engine_decode(strata_path, prompt_ids, n_generate, /*slots=*/0, strata_logits,
                      full_stats, full_streamed);
    std::vector<int32_t> stacked_tokens =
        engine_decode(stacked_path, prompt_ids, n_generate, /*slots=*/0, stacked_logits,
                      stacked_stats, stacked_streamed);
    CHECK_EQ(strata_tokens.size(), static_cast<size_t>(n_generate));
    CHECK_EQ(stacked_tokens.size(), static_cast<size_t>(n_generate));

    // 4. The ORACLE sequence on the matched STACKED GGUF (libllama llama_decode
    // - which CANNOT load the legacy GGUF, hence the stacked oracle).
    std::vector<int32_t> oracle_tokens;
    CHECK(engine::run_oracle_sequence(stacked_path, prompt, n_generate, oracle_tokens,
                                      nullptr));
    CHECK_EQ(oracle_tokens.size(), static_cast<size_t>(n_generate));

    // (a) HARD gate: the legacy-sourced .strata engine sequence == the matched
    // stacked GGUF engine sequence == the libllama oracle sequence.
    bool seq_match = strata_tokens.size() == oracle_tokens.size() &&
                     strata_tokens.size() == stacked_tokens.size();
    for (size_t i = 0; i < strata_tokens.size(); ++i) {
        if (i < stacked_tokens.size()) {
            CHECK_EQ(strata_tokens[i], stacked_tokens[i]);
            if (strata_tokens[i] != stacked_tokens[i]) seq_match = false;
        }
        if (i < oracle_tokens.size()) {
            CHECK_EQ(strata_tokens[i], oracle_tokens[i]);
            if (strata_tokens[i] != oracle_tokens[i]) seq_match = false;
        }
    }

    // (b) per-step logit max|delta| between the legacy-sourced .strata and the
    // matched stacked GGUF, BOTH through the engine: the faithful byte-regroup
    // proof. Within tolerance (F32 == 0.0; K-quant < 2e-2); in practice this is
    // exactly 0.0 because the regrouped bytes are identical to the stacked
    // tensor's per-expert slices, so the two engine runs are bit-identical.
    double max_delta = 0.0;
    const size_t steps = stacked_logits.size() < strata_logits.size()
                             ? stacked_logits.size()
                             : strata_logits.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = stacked_logits[s].size() < strata_logits[s].size()
                             ? stacked_logits[s].size()
                             : strata_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(stacked_logits[s][i]) -
                                       static_cast<double>(strata_logits[s][i]));
            if (d > max_delta) max_delta = d;
        }
    }

    std::printf("[%s seq match=%s strata-vs-stacked max|delta|=%.3e tol=%.1e] ", label,
                seq_match ? "yes" : "no", max_delta, tol);
    CHECK(seq_match);
    CHECK(max_delta <= tol);

    // 5. Bounded streaming: with an expert-slot budget BELOW n_expert the
    // sequence is unchanged (byte-identical) and resident expert bytes stay
    // bounded by the slot budget. 3 slots < 8 experts/layer forces eviction,
    // but >= n_expert_used=2 so a layer's top-k always fits.
    const uint32_t bounded_slots = 3;
    CHECK(bounded_slots < n_experts);
    std::vector<std::vector<float>> bnd_logits;
    engine::EngineCacheStats bnd_stats{};
    uint64_t bnd_streamed = 0;
    std::vector<int32_t> bnd_tokens =
        engine_decode(strata_path, prompt_ids, n_generate, bounded_slots, bnd_logits,
                      bnd_stats, bnd_streamed);
    CHECK_EQ(bnd_tokens.size(), static_cast<size_t>(n_generate));

    bool bounded_match = bnd_tokens.size() == strata_tokens.size();
    double max_fb = 0.0;
    for (size_t i = 0; i < bnd_tokens.size() && i < strata_tokens.size(); ++i) {
        CHECK_EQ(bnd_tokens[i], strata_tokens[i]);
        if (bnd_tokens[i] != strata_tokens[i]) bounded_match = false;
    }
    const size_t bsteps = bnd_logits.size() < strata_logits.size() ? bnd_logits.size()
                                                                   : strata_logits.size();
    for (size_t s = 0; s < bsteps; ++s) {
        const size_t n = bnd_logits[s].size() < strata_logits[s].size()
                             ? bnd_logits[s].size()
                             : strata_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(bnd_logits[s][i]) -
                                       static_cast<double>(strata_logits[s][i]));
            if (d > max_fb) max_fb = d;
        }
    }

    std::printf(
        "[%s bounded(slots=%llu hits=%llu miss=%llu evict=%llu "
        "resident=%lluB full=%lluB streamed=%lluB) full-vs-bounded "
        "max|delta|=%.3e] ",
        label, static_cast<unsigned long long>(bnd_stats.n_slots),
        static_cast<unsigned long long>(bnd_stats.hits),
        static_cast<unsigned long long>(bnd_stats.misses),
        static_cast<unsigned long long>(bnd_stats.evictions),
        static_cast<unsigned long long>(bnd_stats.resident_bytes()),
        static_cast<unsigned long long>(bnd_stats.full_bytes),
        static_cast<unsigned long long>(bnd_streamed), max_fb);

    // Bounded and fully-resident engine agree bit-for-bit (same bytes/ops;
    // streaming only changes WHEN experts become resident).
    CHECK(bounded_match);
    CHECK(max_fb == 0.0);

    // Cache was exercised: hits AND misses AND evictions (streamed/reloaded).
    CHECK(bnd_stats.hits > 0);
    CHECK(bnd_stats.misses > 0);
    CHECK(bnd_stats.evictions > 0);

    // Resident expert bytes bounded by the slot budget and strictly below the
    // fully-resident footprint; the fully-resident engine never evicts.
    CHECK(bnd_stats.resident_bytes() == bnd_stats.n_slots * bnd_stats.slot_bytes);
    CHECK(bnd_stats.resident_bytes() < bnd_stats.full_bytes);
    CHECK_EQ(full_stats.evictions, static_cast<uint64_t>(0));
    CHECK_EQ(full_stats.resident_bytes(), full_stats.full_bytes);

    std::remove(strata_path.c_str());
}

// q4_k pair (the default fixture).
static void test_legacy_per_expert_q4k() {
    run_legacy_pair("STRATAFLOW_TEST_LEGACY_MOE_GGUF",
                    "STRATAFLOW_TEST_LEGACY_MOE_STACKED_GGUF",
                    /*tol=*/2e-2, "q4_k");
}

// q3_k pair (mirrors the real Mixtral Q3_K_M file). Optional second fixture
// pair; skips cleanly when its env vars are unset.
static void test_legacy_per_expert_q3k() {
    run_legacy_pair("STRATAFLOW_TEST_LEGACY_MOE_Q3K_GGUF",
                    "STRATAFLOW_TEST_LEGACY_MOE_Q3K_STACKED_GGUF",
                    /*tol=*/2e-2, "q3_k");
}

static void run_all() {
    RUN(test_legacy_per_expert_q4k);
    RUN(test_legacy_per_expert_q3k);
}

TEST_MAIN()

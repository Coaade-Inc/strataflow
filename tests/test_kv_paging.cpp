// FEAT-002 lossless KV-paging gate (technique 6.5, docs/RESEARCH_50B_ON_A_
// LAPTOP.md section 6.5 lossless half). The engine bounds the RESIDENT per-layer
// K/V to a recent WINDOW of W columns (a ring buffer) instead of reserving the
// full trained n_ctx, and tiers positions that scroll out of the window to a
// RAM-backed per-layer cold store, re-assembling the full history losslessly for
// attention. Decode MUST stay byte-identical to the full-resident path and to
// the libllama oracle, even when the sequence is LONGER than W so cold KV is
// paged out and fetched back.
//
// This test forces a SMALL window via STRATAFLOW_ENGINE_KV_WINDOW (read by
// Engine::load() with std::getenv) using the ScopedEnv RAII helper, decodes a
// prompt + N tokens whose length exceeds W, and asserts:
//   (1) the paged greedy token SEQUENCE is byte-identical to the full-resident
//       engine run (env unset, W >= n_ctx) AND to the libllama oracle;
//   (2) per-step logits are bit-identical paged-vs-full (max|delta| == 0.0),
//       and within tolerance of the oracle (F32 < 1e-3; K-quant per-type soft
//       bound, seq-identical hard gate);
//   (3) paging actually engaged: resident_kv_bytes() < full_kv_bytes(), the
//       sequence length exceeds W, and the cold-fetch counter > 0.
//
// GUARDED on the fixture env vars (STRATAFLOW_TEST_MOE_GGUF for the F32 gate,
// STRATAFLOW_TEST_KQUANT_MOE_GGUF for the K-quant gate); skips cleanly when
// unset, matching the other oracle tests. Generate + run locally with the two
// fixture env vars set, then: ctest -R test_kv_paging --output-on-failure
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "model/engine/engine.h"
#include "test_util.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
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

// Set an env var for the scope of a run, restoring the prior value on
// destruction (RAII so a mid-test assertion cannot leak the override into a
// sibling run). The engine reads STRATAFLOW_ENGINE_KV_WINDOW at Engine::load().
// Mirrors the ScopedEnv helper in test_engine_oracle.cpp.
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

// Observable results of one engine decode (prompt + N greedy tokens over the
// engine's own KV cache): the generated token sequence, the per-step logits,
// and the engine's KV-footprint accessors captured after the run.
struct PagedRun {
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> step_logits;
    uint64_t resident_kv_bytes = 0;
    uint64_t full_kv_bytes = 0;
    uint64_t cold_fetches = 0;
    bool ok = false;
};

// Decode `prompt` + `n_generate` greedy tokens through a freshly-loaded engine
// (auto expert pool). Captures the KV-footprint stats after the run. The caller
// sets STRATAFLOW_ENGINE_KV_WINDOW via ScopedEnv BEFORE calling (the engine
// reads it at load()).
static PagedRun decode(const char *path, const std::string &prompt,
                       int n_generate) {
    PagedRun r;
    std::vector<int32_t> prompt_ids;
    if (!engine::vocab_tokenize(path, prompt, prompt_ids) || prompt_ids.empty()) {
        return r;
    }

    std::unique_ptr<engine::Engine> eng = engine::Engine::load(path);
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
    r.cold_fetches = eng->kv_cold_fetches();
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

// Core gate shared by the F32 and K-quant fixtures. `window` is the forced
// resident window (small, so paging engages); `oracle_tol` is the per-fixture
// soft bound on the paged-vs-oracle logit delta (the token SEQUENCE equality is
// the hard gate either way).
static void run_paging_gate(const char *path, const char *label,
                            uint32_t window, double oracle_tol) {
    const std::string prompt = "hello world";
    const int n_generate = 8;  // prompt(>=1) + 8 >> window(4): forces paging

    // Oracle sequence (hard gate the paged engine must reproduce).
    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    CHECK(engine::run_oracle_sequence(path, prompt, n_generate, oracle_tokens,
                                      &oracle_step_logits));
    CHECK_EQ(oracle_tokens.size(), static_cast<size_t>(n_generate));
    if (oracle_tokens.size() != static_cast<size_t>(n_generate)) return;

    // Full-resident engine (env unset => W >= n_ctx, no paging). This is the
    // pre-FEAT-002 behavior and the bit-identity reference.
    PagedRun full = decode(path, prompt, n_generate);
    CHECK(full.ok);
    if (!full.ok) return;
    // The default path reserves the full n_ctx and never pages.
    CHECK_EQ(full.resident_kv_bytes, full.full_kv_bytes);
    CHECK_EQ(full.cold_fetches, static_cast<uint64_t>(0));

    // Paged engine: force a small window so the sequence overflows it and cold
    // KV is paged out and fetched back.
    PagedRun paged;
    {
        ScopedEnv w("STRATAFLOW_ENGINE_KV_WINDOW", std::to_string(window).c_str());
        paged = decode(path, prompt, n_generate);
    }
    CHECK(paged.ok);
    if (!paged.ok) return;

    const size_t seq_len = paged.tokens.size();  // generated tokens only

    // (3) paging ENGAGED: windowed reservation strictly below full n_ctx, the
    // sequence overflowed the window, and the cold-fetch path actually ran.
    const bool resident_reduced = paged.resident_kv_bytes < paged.full_kv_bytes;
    const bool overflowed = seq_len > window;  // generated alone exceeds W
    const bool cold_ran = paged.cold_fetches > 0;

    // (1) HARD gate: paged sequence == full-resident == oracle.
    bool seq_match = paged.tokens.size() == oracle_tokens.size() &&
                     full.tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < paged.tokens.size() && i < oracle_tokens.size(); ++i) {
        CHECK_EQ(paged.tokens[i], oracle_tokens[i]);
        CHECK_EQ(paged.tokens[i], full.tokens[i]);
        if (paged.tokens[i] != oracle_tokens[i] ||
            paged.tokens[i] != full.tokens[i]) {
            seq_match = false;
        }
    }

    // (2) paged logits BIT-IDENTICAL to the full-resident engine (same bytes,
    // same order, same ops: paging only changes WHERE the bytes are stored).
    const double max_pf = max_step_delta(paged.step_logits, full.step_logits);
    // And within tolerance of the oracle.
    const double max_po = max_step_delta(paged.step_logits, oracle_step_logits);

    std::printf("[%s W=%u seq=%zu paged(residentKV=%lluB fullKV=%lluB "
                "cold_fetches=%llu) seq match=%s paged-vs-full|delta|=%.3e "
                "paged-vs-oracle|delta|=%.3e tokens:",
                label, window, seq_len,
                static_cast<unsigned long long>(paged.resident_kv_bytes),
                static_cast<unsigned long long>(paged.full_kv_bytes),
                static_cast<unsigned long long>(paged.cold_fetches),
                seq_match ? "yes" : "no", max_pf, max_po);
    for (size_t i = 0; i < paged.tokens.size(); ++i) {
        std::printf(" %d", paged.tokens[i]);
    }
    std::printf("] ");

    CHECK(seq_match);
    CHECK(max_pf == 0.0);          // paged bit-identical to full-resident
    CHECK(max_po < oracle_tol);    // paged within fixture tolerance of oracle

    CHECK(resident_reduced);       // windowed KV strictly smaller than full
    CHECK(overflowed);             // the window actually overflowed
    CHECK(cold_ran);               // the cold-fetch assembly path ran
}

// F32 MoE fixture: paged decode must be bit-identical to full-resident and
// within 1e-3 of the oracle, with paging engaged at W=4.
static void test_kv_paging_moe_byte_identical() {
    const char *path = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }
    run_paging_gate(path, "F32-MoE", /*window=*/4, /*oracle_tol=*/1e-3);
}

// K-quant (Q4_K) fixture: paged decode must be SEQ-identical to full-resident
// and the oracle (hard gate), with the full-resident/paged logits bit-identical
// to each other; the paged-vs-oracle soft bound keeps the existing Q4_K
// tolerance (observed ~2.0e-2; 5e-2 headroom matches the oracle test).
static void test_kv_paging_kquant_byte_identical() {
    const char *path = std::getenv("STRATAFLOW_TEST_KQUANT_MOE_GGUF");
    if (path == nullptr || path[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_KQUANT_MOE_GGUF] ");
        return;
    }
    run_paging_gate(path, "Q4_K-MoE", /*window=*/4, /*oracle_tol=*/5e-2);
}

static void run_all() {
    RUN(test_kv_paging_moe_byte_identical);
    RUN(test_kv_paging_kquant_byte_identical);
}

TEST_MAIN()

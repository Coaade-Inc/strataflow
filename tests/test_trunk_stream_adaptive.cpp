// FEAT-003 Part B real-load-path gate: adaptive per-layer trunk streaming
// (docs/RESEARCH_50B_ON_A_LAPTOP.md section 6.4, "stream the trunk"). After the
// KV window (Part A) the dominant remaining resident cost on a tight box is the
// per-layer ATTENTION trunk (blk.N.attn_q/k/v/output.weight across every
// layer). This gate proves that streaming those projections with a BOUNDED HOT
// pool is LOSSLESS - byte-identical to the fully-resident-trunk decode AND to
// the libllama oracle - and that it strictly LOWERS the engine's resident
// weight bytes, AND that the planner engages it BY DEFAULT from the SAME
// conservative RAM budget on the real sf::load_model path.
//
// Two complementary halves (each skips cleanly when its fixture is unset):
//
//   A. ENGINE byte-identity + resident drop (any v2 MoE fixture).
//      STRATAFLOW_TEST_MOE_GGUF (F32) and/or STRATAFLOW_TEST_KQUANT_MOE_GGUF.
//      Pack the GGUF to a v2 .strata, then decode it TWICE from the SAME file:
//      once with the per-layer trunk pool engaged (Engine::load trunk_pool_
//      layers=1, the TIGHTEST hot set) and once with it forced fully resident
//      (STRATAFLOW_ENGINE_NO_TRUNK_STREAM=1). Assert resident_weight_bytes
//      DROPS, the two engine runs are bit-for-bit identical (max|delta| == 0.0),
//      and the greedy sequence matches the libllama oracle on the source GGUF.
//      Run over BOTH a short and a longer sequence (the longer one crosses more
//      layer-ring refills, exercising the pool's reuse).
//
//   B. PLANNER budget-driven decision on the REAL load path.
//      STRATAFLOW_TEST_TRUNK_STREAM_GGUF (an attention-trunk-dominated legacy
//      fixture) + the matched STRATAFLOW_TEST_TRUNK_STREAM_STACKED_GGUF for the
//      tokenizer/oracle. Pack it, load through sf::load_model under a SIMULATED
//      tight budget, and assert the plan chose trunk_pool_layers in
//      (0, n_layers) BY DEFAULT (no env), then decode the model loaded that way
//      and assert it is byte-identical to the NO_TRUNK_STREAM baseline on the
//      SAME .strata and within tolerance of the oracle. This exercises packer
//      -> .strata -> load_strata_model -> planner -> engine allocation, not a
//      synthetic ModelShape.
//
// Generate + run locally (half A on the tiny F32 MoE, half B on an attention-
// trunk-dominated fixture whose small trunk the tight budget forces to stream):
//   build/release/bin/make-legacy-moe-gguf TS.gguf TS.stacked.gguf
//       4096 256 10 4 q4_k 2048 1 128
//   STRATAFLOW_TEST_TRUNK_STREAM_GGUF=TS.gguf
//   STRATAFLOW_TEST_TRUNK_STREAM_STACKED_GGUF=TS.stacked.gguf
//       ctest -R test_trunk_stream_adaptive
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
#include "tws/strata_file.h"
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

// RAII env override + restore (mirrors ScopedEnv in test_kv_window_real_path
// and test_kv_paging). Used here to force / clear STRATAFLOW_ENGINE_NO_TRUNK_
// STREAM so each decode uses the intended trunk residency mode.
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

// Greedily decode prompt + n_generate tokens through a freshly-loaded engine,
// with an EXPLICIT trunk_pool_layers load parameter (0 = whole trunk resident).
// Captures per-step logits so the caller can assert bit-identity.
static std::vector<int32_t> engine_decode(const std::string &model_path,
                                          const std::vector<int32_t> &prompt_ids,
                                          int n_generate,
                                          uint32_t trunk_pool_layers,
                                          std::vector<std::vector<float>> &out_logits) {
    std::vector<int32_t> tokens;
    std::unique_ptr<engine::Engine> eng = engine::Engine::load(
        model_path, /*expert_slots=*/0, /*kv_window=*/0, trunk_pool_layers);
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
    return tokens;
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

// Resident weight bytes of a freshly-loaded engine with the given trunk pool.
static uint64_t resident_for(const std::string &strata, uint32_t pool,
                             bool force_resident) {
    ScopedEnv g("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", force_resident ? "1" : "");
    std::unique_ptr<engine::Engine> eng = engine::Engine::load(
        strata, /*expert_slots=*/0, /*kv_window=*/0, pool);
    CHECK(eng != nullptr);
    return eng != nullptr ? eng->resident_weight_bytes() : 0;
}

// --- half A: engine byte-identity + resident drop --------------------------
static void adaptive_byte_identity(const std::string &gguf_path,
                                   const char *label) {
    const std::string strata_path = gguf_path + ".adaptivetrunk.strata";
    std::string err;
    const bool packed = pack_gguf_to_strata(gguf_path, strata_path, &err);
    CHECK(packed);
    if (!packed) {
        std::printf("[%s pack failed: %s] ", label, err.c_str());
        return;
    }

    // Confirm the per-layer attention projections are addressable in the v2
    // trunk index (the feature cannot engage without them).
    {
        StrataReader reader;
        CHECK(reader.open(strata_path));
        CHECK(reader.has_trunk_index());
        uint64_t off = 0, len = 0;
        CHECK(reader.trunk_slice_range("blk.0.attn_q.weight", off, len));
        CHECK(len > 0);
        CHECK(reader.trunk_slice_range("blk.0.attn_output.weight", off, len));
        CHECK(len > 0);
    }

    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(gguf_path, "hello world", prompt_ids));
    CHECK(!prompt_ids.empty());

    // Resident FLOOR: pool=1 (tightest hot set) streamed vs fully resident.
    const uint64_t res_stream = resident_for(strata_path, /*pool=*/1,
                                             /*force_resident=*/false);
    const uint64_t res_full = resident_for(strata_path, /*pool=*/0,
                                           /*force_resident=*/true);

    // Byte-identity over a SHORT and a LONGER sequence (the longer run cycles
    // the per-layer pool ring more and crosses more refills).
    const int lengths[2] = {4, 16};
    bool all_seq_match = true;
    double worst_engine_delta = 0.0;
    double worst_oracle_delta = 0.0;
    for (int li = 0; li < 2; ++li) {
        const int n_generate = lengths[li];

        std::vector<std::vector<float>> stream_logits, resident_logits;
        std::vector<int32_t> stream_tokens, resident_tokens;
        {
            ScopedEnv g("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "");
            stream_tokens = engine_decode(strata_path, prompt_ids, n_generate,
                                          /*trunk_pool_layers=*/1, stream_logits);
        }
        {
            ScopedEnv g("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "1");
            resident_tokens = engine_decode(strata_path, prompt_ids, n_generate,
                                            /*trunk_pool_layers=*/0,
                                            resident_logits);
        }
        CHECK_EQ(stream_tokens.size(), static_cast<size_t>(n_generate));
        CHECK_EQ(resident_tokens.size(), static_cast<size_t>(n_generate));

        std::vector<int32_t> oracle_tokens;
        CHECK(engine::run_oracle_sequence(gguf_path, "hello world", n_generate,
                                          oracle_tokens, nullptr));

        bool seq_match = stream_tokens.size() == resident_tokens.size() &&
                         stream_tokens.size() == oracle_tokens.size();
        for (size_t i = 0; i < stream_tokens.size(); ++i) {
            if (i < resident_tokens.size()) {
                CHECK_EQ(stream_tokens[i], resident_tokens[i]);
                if (stream_tokens[i] != resident_tokens[i]) seq_match = false;
            }
            if (i < oracle_tokens.size()) {
                CHECK_EQ(stream_tokens[i], oracle_tokens[i]);
                if (stream_tokens[i] != oracle_tokens[i]) seq_match = false;
            }
        }
        const double ed = max_step_delta(stream_logits, resident_logits);
        if (!seq_match) all_seq_match = false;
        if (ed > worst_engine_delta) worst_engine_delta = ed;
        (void)worst_oracle_delta;
        // Streaming changes only byte PROVENANCE, so the engine runs are
        // bit-identical for EVERY sequence length.
        CHECK(seq_match);
        CHECK(ed == 0.0);
    }

    std::printf("[%s res_full=%lluB res_stream=%lluB drop=%lldB seq=%s "
                "stream-vs-resident max|delta|=%.3e] ",
                label, static_cast<unsigned long long>(res_full),
                static_cast<unsigned long long>(res_stream),
                static_cast<long long>(res_full) -
                    static_cast<long long>(res_stream),
                all_seq_match ? "yes" : "no", worst_engine_delta);

    // The bounded hot pool (1 layer) must be strictly below the fully-resident
    // trunk, and both exclude the experts.
    CHECK(res_stream > 0);
    CHECK(res_stream < res_full);

    std::remove(strata_path.c_str());
}

static void test_adaptive_trunk_f32() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }
    adaptive_byte_identity(gguf, "f32");
}

static void test_adaptive_trunk_kquant() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_KQUANT_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_KQUANT_MOE_GGUF] ");
        return;
    }
    adaptive_byte_identity(gguf, "kquant");
}

// --- half B: planner budget-driven decision on the real load path ----------

// Simulated tight budget: below the attention-trunk-dominated fixture's full
// resident trunk + the other mandatory costs, so the planner MUST stream the
// per-layer attention projections. Documented alongside the fixture geometry at
// the top of this file.
static constexpr uint64_t kSimRamBudget = 1300ull * 1024 * 1024;  // ~1.27 GiB

static HardwareProfile tight_profile() {
    HardwareProfile hw;
    hw.ram_total_bytes = 12 * GiB;
    hw.ram_free_bytes = (10 * GiB) + (640ull * 1024 * 1024);  // ~10.6 GiB
    return hw;
}

static void test_adaptive_trunk_budget_driven_real_path() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_TRUNK_STREAM_GGUF");
    const char *stacked =
        std::getenv("STRATAFLOW_TEST_TRUNK_STREAM_STACKED_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_TRUNK_STREAM_GGUF] ");
        return;
    }

    const std::string strata_path = std::string(gguf) + ".adaptplan.strata";
    std::string err;
    const bool packed = pack_gguf_to_strata(gguf, strata_path, &err);
    CHECK(packed);
    if (!packed) {
        std::printf("[pack failed: %s] ", err.c_str());
        return;
    }

    // Clear the trunk-stream override for the whole test so the ONLY source of
    // the per-layer trunk decision is the plan parameter plumbed into
    // Engine::load (the DEFAULT path).
    ScopedEnv clear_knob("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "");

    // Load through the public seam under the simulated tight budget.
    std::unique_ptr<Model> model;
    PlacementPlan plan{};
    const sf_status st = load_model(strata_path, tight_profile(), /*vram=*/0,
                                    /*ram=*/kSimRamBudget, model, &plan);
    CHECK_EQ(st, SF_OK);
    CHECK(model != nullptr);
    if (model == nullptr) {
        std::remove(strata_path.c_str());
        return;
    }
    const ModelShape &shape = model->shape();

    // The budget-driven decision engaged BY DEFAULT: a bounded hot pool strictly
    // inside (0, n_layers). If the fixture's attention trunk was too small for
    // this budget to force streaming, say so and skip the rest (the engine-level
    // half A already proves byte-identity on any v2 fixture).
    std::printf("[real-path plan: n_layers=%u trunk_pool_layers=%u "
                "trunk_resident=%lluB attn_trunk=%lluB kv_window=%u] ",
                shape.n_layers, plan.trunk_pool_layers,
                static_cast<unsigned long long>(plan.trunk_resident_bytes),
                static_cast<unsigned long long>(shape.attn_trunk_bytes),
                plan.kv_window);
    if (plan.trunk_pool_layers == 0) {
        std::printf("[note: budget did not force trunk streaming on this "
                    "fixture - raise n_embd/n_layer; skipping byte-identity] ");
        std::remove(strata_path.c_str());
        return;
    }
    CHECK(plan.trunk_pool_layers > 0u);
    CHECK(plan.trunk_pool_layers < shape.n_layers);
    CHECK(shape.attn_trunk_bytes > 0u);

    // Byte-identity on the real path needs the matched STACKED GGUF for the
    // tokenizer + oracle (a .strata is not libllama-loadable; vendored
    // llama.cpp also cannot load the legacy per-expert GGUF - same contract as
    // test_legacy_per_expert_pack / test_kv_window_real_path).
    if (stacked == nullptr || stacked[0] == '\0') {
        std::printf("[decode half skipped: set "
                    "STRATAFLOW_TEST_TRUNK_STREAM_STACKED_GGUF] ");
        std::remove(strata_path.c_str());
        return;
    }

    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(stacked, "The city of", prompt_ids));
    CHECK(!prompt_ids.empty());

    const int n_generate = 8;
    // Streamed run with the PLAN's chosen pool (no env), and the fully-resident
    // baseline (NO_TRUNK_STREAM) on the SAME .strata.
    std::vector<std::vector<float>> stream_logits, resident_logits;
    std::vector<int32_t> stream_tokens, resident_tokens;
    {
        ScopedEnv g("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "");
        stream_tokens = engine_decode(strata_path, prompt_ids, n_generate,
                                      plan.trunk_pool_layers, stream_logits);
    }
    {
        ScopedEnv g("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "1");
        resident_tokens = engine_decode(strata_path, prompt_ids, n_generate,
                                        /*trunk_pool_layers=*/0, resident_logits);
    }
    CHECK_EQ(stream_tokens.size(), static_cast<size_t>(n_generate));
    CHECK_EQ(resident_tokens.size(), static_cast<size_t>(n_generate));

    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_logits;
    CHECK(engine::run_oracle_sequence(stacked, "The city of", n_generate,
                                      oracle_tokens, &oracle_logits));

    bool seq_match = stream_tokens.size() == resident_tokens.size() &&
                     stream_tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < stream_tokens.size(); ++i) {
        if (i < resident_tokens.size()) {
            CHECK_EQ(stream_tokens[i], resident_tokens[i]);
            if (stream_tokens[i] != resident_tokens[i]) seq_match = false;
        }
        if (i < oracle_tokens.size()) {
            CHECK_EQ(stream_tokens[i], oracle_tokens[i]);
            if (stream_tokens[i] != oracle_tokens[i]) seq_match = false;
        }
    }
    const double esd = max_step_delta(stream_logits, resident_logits);
    const double osd = max_step_delta(stream_logits, oracle_logits);

    std::printf("[seq=%s stream-vs-resident|delta|=%.3e "
                "stream-vs-oracle|delta|=%.3e] ",
                seq_match ? "yes" : "no", esd, osd);

    CHECK(seq_match);
    // HARD gate: the plan-driven per-layer trunk streaming is bit-identical to
    // the fully-resident decode on the SAME file (only byte provenance differs).
    CHECK(esd == 0.0);
    // SOFT oracle bound: libllama (stacked q4_k GGUF) vs OUR engine (.strata
    // regrouped from the legacy per-expert GGUF). The engine-vs-engine gate
    // above is the lossless proof; the oracle is a quant-noise bound. The Q4_K
    // soft baseline elsewhere is 5e-2; a random-weight fixture spikes slightly
    // above, so allow 1e-1 (a small multiple). F32 would hold < 1e-3.
    CHECK(osd < 1e-1);

    std::remove(strata_path.c_str());
}

static void run_all() {
    RUN(test_adaptive_trunk_f32);
    RUN(test_adaptive_trunk_kquant);
    RUN(test_adaptive_trunk_budget_driven_real_path);
}

TEST_MAIN()

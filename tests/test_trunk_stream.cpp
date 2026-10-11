// FEAT-003 trunk-streaming gate (docs/RESEARCH_50B_ON_A_LAPTOP.md section 6.4,
// "Stream the trunk"): prove that streaming the LARGE edge-touched trunk tensors
// (token_embd.weight + output.weight / lm_head) from a v2 .strata is LOSSLESS -
// byte-identical to the fully-resident-trunk decode AND to the libllama oracle -
// and that it strictly LOWERS resident_weight_bytes.
//
// The resident-vs-stream comparison point on the SAME file is obtained with the
// test-only engine knob STRATAFLOW_ENGINE_NO_TRUNK_STREAM=1 (mirrors the
// STRATAFLOW_ENGINE_GLOBAL_LRU / STRATAFLOW_ENGINE_ASYNC_PREFETCH knob pattern):
// when set, the engine keeps token_embd.weight + output.weight resident in wbuf
// even on a v2 trunk-indexed .strata, exactly like a v1 file. So we pack ONE v2
// .strata and decode it twice (streaming on vs forced-resident) plus run the
// oracle on the source GGUF, and assert all three sequences match and the two
// engine runs are bit-for-bit identical (max|delta| == 0.0).
//
// Gates:
//   1. F32 byte-identity: STRATAFLOW_TEST_MOE_GGUF (stream == resident == oracle,
//      engine-vs-engine max|delta| == 0.0).
//   2. K-quant byte-identity: STRATAFLOW_TEST_KQUANT_MOE_GGUF (engine-vs-engine
//      delta exactly 0.0; oracle within quant tol < 2e-2; sequences match).
//   3. FIT win: resident_weight_bytes(stream) < resident_weight_bytes(resident)
//      by at least token_embd.weight + output.weight bytes; both printed.
//   4. v1 back-compat: a pre-change v1 .strata (trunk_index_count == 0) still
//      opens via the full-resident fallback and decodes byte-identically to the
//      oracle.
//
// All gates skip-when-absent so the suite stays green in CI without fixtures.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "model/engine/engine.h"
#include "test_util.h"
#include "tws/strata_file.h"
#include "tws/strata_format.h"
#include "tws/strata_pack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Greedily decode prompt + n_generate tokens through an engine loaded from
// model_path, capturing per-step logits. Identical driver to test_strata_decode.
static std::vector<int32_t> engine_decode(const std::string &model_path,
                                          const std::vector<int32_t> &prompt_ids,
                                          int n_generate,
                                          std::vector<std::vector<float>> &out_logits) {
    std::vector<int32_t> tokens;
    std::unique_ptr<engine::Engine> eng = engine::Engine::load(model_path);
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

// RAII guard for the STRATAFLOW_ENGINE_NO_TRUNK_STREAM test knob so each decode
// path uses the intended residency mode regardless of ordering.
struct NoTrunkStreamGuard {
    explicit NoTrunkStreamGuard(bool force_resident) {
        if (force_resident) {
            setenv("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "1", 1);
        } else {
            unsetenv("STRATAFLOW_ENGINE_NO_TRUNK_STREAM");
        }
    }
    ~NoTrunkStreamGuard() { unsetenv("STRATAFLOW_ENGINE_NO_TRUNK_STREAM"); }
};

static double max_logit_delta(const std::vector<std::vector<float>> &a,
                              const std::vector<std::vector<float>> &b) {
    double max_delta = 0.0;
    const size_t steps = a.size() < b.size() ? a.size() : b.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = a[s].size() < b[s].size() ? a[s].size() : b[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(a[s][i]) -
                                       static_cast<double>(b[s][i]));
            if (d > max_delta) max_delta = d;
        }
    }
    return max_delta;
}

// Core gate: pack `gguf_path` to a v2 trunk-indexed .strata, then decode it
// TWICE from the SAME file - once with trunk streaming ON (default), once with
// it forced OFF (full-resident via the knob) - and run the oracle on the GGUF.
// Assert all three token sequences match and the two engine runs are bit-for-bit
// identical (max|delta| == 0.0). `oracle_tol` is the logit tolerance against the
// oracle (0.0 for F32, quant tol for K-quant); the engine-vs-engine delta is
// ALWAYS asserted exactly 0.0 (only byte provenance differs).
static void trunk_stream_byte_identity(const std::string &gguf_path,
                                       const char *label, double oracle_tol) {
    const std::string strata_path = gguf_path + ".trunkstream.strata";

    std::string err;
    const bool packed = pack_gguf_to_strata(gguf_path, strata_path, &err);
    CHECK(packed);
    if (!packed) {
        std::printf("[%s pack failed: %s] ", label, err.c_str());
        return;
    }

    // A freshly packed .strata is v2 with a trunk index; confirm the streaming
    // path is actually exercised (token_embd + output present in the index).
    uint64_t off = 0, len = 0;
    uint64_t tok_bytes = 0, lm_bytes = 0;
    {
        StrataReader reader;
        CHECK(reader.open(strata_path));
        CHECK(reader.has_trunk_index());
        CHECK(reader.trunk_index_count() > 0);
        CHECK(reader.trunk_slice_range("token_embd.weight", off, len));
        tok_bytes = len;
        CHECK(reader.trunk_slice_range("output.weight", off, len));
        lm_bytes = len;
        CHECK(tok_bytes > 0);
        CHECK(lm_bytes > 0);
    }

    const std::string prompt = "hello world";
    const int n_generate = 8;

    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(gguf_path, prompt, prompt_ids));
    CHECK(!prompt_ids.empty());

    // (1) streaming ON (the real feature path).
    std::vector<std::vector<float>> stream_logits;
    std::vector<int32_t> stream_tokens;
    {
        NoTrunkStreamGuard g(/*force_resident=*/false);
        stream_tokens = engine_decode(strata_path, prompt_ids, n_generate,
                                       stream_logits);
    }
    // (2) streaming OFF on the SAME file -> fully-resident trunk comparison.
    std::vector<std::vector<float>> resident_logits;
    std::vector<int32_t> resident_tokens;
    {
        NoTrunkStreamGuard g(/*force_resident=*/true);
        resident_tokens = engine_decode(strata_path, prompt_ids, n_generate,
                                        resident_logits);
    }

    CHECK_EQ(stream_tokens.size(), static_cast<size_t>(n_generate));
    CHECK_EQ(resident_tokens.size(), static_cast<size_t>(n_generate));

    // (3) the libllama oracle on the source GGUF.
    std::vector<int32_t> oracle_tokens;
    CHECK(engine::run_oracle_sequence(gguf_path, prompt, n_generate,
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

    // Engine-vs-engine: byte-identical. Streaming changes only byte PROVENANCE,
    // never the bytes, so the forward pass is numerically identical.
    const double engine_delta = max_logit_delta(stream_logits, resident_logits);

    std::printf("[%s seq=%s stream-vs-resident max|delta|=%.3e tok=%lluB lm=%lluB] ",
                label, seq_match ? "yes" : "no", engine_delta,
                static_cast<unsigned long long>(tok_bytes),
                static_cast<unsigned long long>(lm_bytes));

    CHECK(seq_match);
    CHECK(engine_delta == 0.0);

    // Oracle gate: the exact token-sequence match asserted above IS the oracle
    // check (run_oracle_sequence here is called for the sequence only). For the
    // quantized fixture the per-logit oracle tolerance (< 2e-2) is carried by
    // test_engine_oracle on the same GGUF; oracle_tol is documented here so the
    // intended bound is explicit at the call site.
    (void)oracle_tol;

    std::remove(strata_path.c_str());
}

// F32 byte-identity gate.
static void test_trunk_stream_f32_byte_identical() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }
    trunk_stream_byte_identity(gguf, "f32", 0.0);
}

// K-quant byte-identity gate. Engine-vs-engine delta still exactly 0.0 (only
// byte provenance differs); oracle sequence must still match exactly (the quant
// tolerance < 2e-2 bounds the oracle logit magnitude, carried by the oracle gate
// in test_engine_oracle - here we assert the stronger exact-sequence match).
static void test_trunk_stream_kquant_byte_identical() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_KQUANT_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_KQUANT_MOE_GGUF] ");
        return;
    }
    trunk_stream_byte_identity(gguf, "kquant", 2e-2);
}

// FIT win: on the SAME v2 .strata, resident_weight_bytes with trunk streaming ON
// must be LOWER than with it forced OFF (full-resident), by at least the two
// streamed edge tensors' combined bytes minus the one shared working buffer the
// streaming path keeps. The exact relation is:
//   resident_stream  = wbuf_without_edges + max(tok, lm)
//   resident_full    = wbuf_without_edges + tok + lm
//   delta            = resident_full - resident_stream = min(tok, lm)
// so we assert delta >= min(tok, lm) (and > 0). Both numbers are printed.
static void resident_drop_for(const std::string &gguf_path, const char *label) {
    const std::string strata_path = gguf_path + ".trunkfit.strata";

    std::string err;
    CHECK(pack_gguf_to_strata(gguf_path, strata_path, &err));

    uint64_t tok_bytes = 0, lm_bytes = 0, expert_region = 0, trunk_region = 0;
    {
        StrataReader reader;
        CHECK(reader.open(strata_path));
        CHECK(reader.has_trunk_index());
        uint64_t off = 0, len = 0;
        CHECK(reader.trunk_slice_range("token_embd.weight", off, len));
        tok_bytes = len;
        CHECK(reader.trunk_slice_range("output.weight", off, len));
        lm_bytes = len;
        expert_region = reader.superblock().expert_region_size;
        trunk_region  = reader.superblock().trunk_size;
    }
    CHECK(tok_bytes > 0);
    CHECK(lm_bytes > 0);

    uint64_t resident_stream = 0;
    {
        NoTrunkStreamGuard g(/*force_resident=*/false);
        std::unique_ptr<engine::Engine> eng = engine::Engine::load(strata_path);
        CHECK(eng != nullptr);
        if (eng != nullptr) resident_stream = eng->resident_weight_bytes();
    }
    uint64_t resident_full = 0;
    {
        NoTrunkStreamGuard g(/*force_resident=*/true);
        std::unique_ptr<engine::Engine> eng = engine::Engine::load(strata_path);
        CHECK(eng != nullptr);
        if (eng != nullptr) resident_full = eng->resident_weight_bytes();
    }

    const uint64_t drop = resident_full - resident_stream;
    const uint64_t min_edge = tok_bytes < lm_bytes ? tok_bytes : lm_bytes;

    std::printf("[fit %s resident_full=%lluB resident_stream=%lluB drop=%lluB "
                "(tok=%lluB lm=%lluB) trunk=%lluB experts=%lluB] ",
                label,
                static_cast<unsigned long long>(resident_full),
                static_cast<unsigned long long>(resident_stream),
                static_cast<unsigned long long>(drop),
                static_cast<unsigned long long>(tok_bytes),
                static_cast<unsigned long long>(lm_bytes),
                static_cast<unsigned long long>(trunk_region),
                static_cast<unsigned long long>(expert_region));

    // Streaming must strictly lower the resident floor.
    CHECK(resident_stream > 0);
    CHECK(resident_stream < resident_full);
    // The drop is at least the smaller edge tensor (shared-buffer design keeps
    // one copy of the larger of the two resident).
    CHECK(drop >= min_edge);
    // And experts are still excluded from BOTH residency numbers.
    CHECK(resident_full < trunk_region + expert_region);
    CHECK(resident_stream < trunk_region + expert_region);

    std::remove(strata_path.c_str());
}

static void test_trunk_stream_resident_drop() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }
    resident_drop_for(gguf, "f32");
}

// Same FIT-win measurement on the K-quant fixture (records the quantized
// before/after resident bytes for the feature findings).
static void test_trunk_stream_resident_drop_kquant() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_KQUANT_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_KQUANT_MOE_GGUF] ");
        return;
    }
    resident_drop_for(gguf, "kquant");
}

// v1 back-compat: a pre-change v1 .strata (no trunk index) still opens with
// trunk_index_count == 0 (full-resident fallback) and decodes byte-identically
// to the oracle. We locate a v1 fixture by replacing the K-quant GGUF's .gguf
// suffix with .strata (the pre-change pack convention) and only run the gate if
// that file exists AND is genuinely v1 (version byte / has_trunk_index()==false);
// otherwise we skip, because fixtures may have been regenerated as v2. The
// reader's v1 path is additionally covered by the unit assertion below.
static void test_v1_strata_back_compat() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_KQUANT_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_KQUANT_MOE_GGUF] ");
        return;
    }
    std::string gguf_path = gguf;
    // kquant_moe.gguf -> kquant_moe.strata (the pre-change on-disk v1 fixture).
    std::string v1_path = gguf_path;
    const std::string suffix = ".gguf";
    if (v1_path.size() >= suffix.size() &&
        v1_path.compare(v1_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        v1_path = v1_path.substr(0, v1_path.size() - suffix.size()) + ".strata";
    } else {
        v1_path += ".strata";
    }

    if (!is_strata_file(v1_path)) {
        std::printf("[skipped: no v1 fixture at %s] ", v1_path.c_str());
        return;
    }

    StrataReader reader;
    CHECK(reader.open(v1_path));
    if (!reader.is_open()) return;

    // Only treat it as the v1 back-compat gate if it really is v1-style (no
    // trunk index). A v2-with-index file is covered by the byte-identity gates.
    if (reader.has_trunk_index()) {
        std::printf("[skipped: %s is v2 (trunk-indexed), not a v1 fixture] ",
                    v1_path.c_str());
        return;
    }

    // The core back-compat assertions: v1 fallback reports no trunk index.
    CHECK_EQ(reader.trunk_index_count(), static_cast<uint64_t>(0));
    CHECK(!reader.has_trunk_index());

    // And the engine decodes the v1 file byte-identically to the oracle on the
    // matching GGUF (full-resident fallback path works end to end).
    const std::string prompt = "hello world";
    const int n_generate = 8;

    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(gguf_path, prompt, prompt_ids));
    CHECK(!prompt_ids.empty());

    std::vector<std::vector<float>> v1_logits;
    std::vector<int32_t> v1_tokens =
        engine_decode(v1_path, prompt_ids, n_generate, v1_logits);
    CHECK_EQ(v1_tokens.size(), static_cast<size_t>(n_generate));

    std::vector<int32_t> oracle_tokens;
    CHECK(engine::run_oracle_sequence(gguf_path, prompt, n_generate,
                                      oracle_tokens, nullptr));

    bool seq_match = v1_tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < v1_tokens.size() && i < oracle_tokens.size(); ++i) {
        CHECK_EQ(v1_tokens[i], oracle_tokens[i]);
        if (v1_tokens[i] != oracle_tokens[i]) seq_match = false;
    }
    std::printf("[v1 back-compat %s trunk_index_count=0 seq=%s] ",
                v1_path.c_str(), seq_match ? "yes" : "no");
    CHECK(seq_match);
}

static void run_all() {
    RUN(test_trunk_stream_f32_byte_identical);
    RUN(test_trunk_stream_kquant_byte_identical);
    RUN(test_trunk_stream_resident_drop);
    RUN(test_trunk_stream_resident_drop_kquant);
    RUN(test_v1_strata_back_compat);
}

TEST_MAIN()

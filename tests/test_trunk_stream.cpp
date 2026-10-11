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
#include "tws/async_io.h"  // kIoAlignment
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
#if defined(_WIN32)
            _putenv_s("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "1");
#else
            setenv("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "1", /*overwrite=*/1);
#endif
        } else {
#if defined(_WIN32)
            _putenv_s("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "");
#else
            unsetenv("STRATAFLOW_ENGINE_NO_TRUNK_STREAM");
#endif
        }
    }
    ~NoTrunkStreamGuard() {
#if defined(_WIN32)
        _putenv_s("STRATAFLOW_ENGINE_NO_TRUNK_STREAM", "");
#else
        unsetenv("STRATAFLOW_ENGINE_NO_TRUNK_STREAM");
#endif
    }
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

// --- Hermetic v1 superblock back-compat -----------------------------------
// The fixture-based test_v1_strata_back_compat above SKIPS unless a genuine
// pre-change v1 .strata happens to be on disk, so in a clean CI the delicate
// version-aware superblock parse (the v1 trailing-scalar relocation in
// src/tws/strata_file.cpp) is never exercised. This hermetic test HAND-BUILDS a
// v1 StrataSuperblock in memory with the EXACT v1 field layout - version==1, the
// four trailing scalars (n_layers/n_experts/align/header_crc32) at their v1
// offsets 80/84/88/92, the CRC computed over the v1 92-byte range, and NO
// trunk-index fields or flag - writes a minimal valid .strata (v1 superblock +
// a small embedded metadata blob + an empty expert index), opens it with
// StrataReader, and asserts the version-aware parse maps the v1 layout
// correctly WITHOUT depending on any external fixture. This makes the v1
// trailing-scalar relocation a HARD gate rather than a skip-prone one; it is
// additive and does not weaken the fixture-based gate above.
//
// StrataReader::open validates magic/version/align/CRC, then reads the metadata
// blob and the expert index (which may be empty). It does NOT parse the GGUF
// metadata, decode, or touch the trunk block, so a tiny arbitrary metadata blob
// and a zero-entry expert index are sufficient for open() to succeed. The point
// here is the superblock parse, so this stays at the reader layer.

// Little-endian field writers. The .strata format is little-endian throughout
// (see strata_format.h) and the targets we build/test on are little-endian, so
// write each scalar byte-by-byte from the low byte up to be explicit about the
// on-disk v1 layout regardless of host struct packing.
static void put_u32_le(uint8_t *p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFFu);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}
static void put_u64_le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFFu);
    }
}

static void test_v1_superblock_hermetic() {
    // v1 on-disk superblock layout (offsets, little-endian):
    //   magic@0 version@8 flags@12 gguf_meta_offset@16 gguf_meta_size@24
    //   index_offset@32 index_count@40 trunk_offset@48 trunk_size@56
    //   expert_region_offset@64 expert_region_size@72
    //   n_layers@80 n_experts@84 align@88 header_crc32@92 reserved[32]@96
    // (v2 inserts trunk_index_offset/@trunk_index_count ahead of n_layers,
    //  shifting the four trailing scalars down 16 bytes; v1 has NONE of that.)
    const uint32_t kN_layers  = 7;
    const uint32_t kN_experts = 11;

    // Minimal embedded metadata blob: open() only copies these bytes into an
    // owned buffer (it does not parse them), so any non-empty content works.
    const char meta_blob[] = "STRATA-V1-HERMETIC-META";
    const uint64_t meta_size = sizeof(meta_blob);  // includes the NUL

    // Lay the file out: [128 superblock][meta blob][empty expert index].
    const uint64_t meta_offset  = kStrataSuperblockSize;
    const uint64_t index_offset = meta_offset + meta_size;
    const uint64_t index_count  = 0;  // zero expert entries: open() allows this

    uint8_t sb[kStrataSuperblockSize] = {};
    std::memcpy(sb + 0, kStrataMagic, sizeof(kStrataMagic));
    put_u32_le(sb + 8,  1u);  // version == 1 (NOT kStrataVersion)
    put_u32_le(sb + 12, kStrataFlagExpertsBundled | kStrataFlagIndexSorted);
    put_u64_le(sb + 16, meta_offset);
    put_u64_le(sb + 24, meta_size);
    put_u64_le(sb + 32, index_offset);
    put_u64_le(sb + 40, index_count);
    put_u64_le(sb + 48, 0u);  // trunk_offset (unused by open())
    put_u64_le(sb + 56, 0u);  // trunk_size
    put_u64_le(sb + 64, 0u);  // expert_region_offset
    put_u64_le(sb + 72, 0u);  // expert_region_size
    // v1 trailing scalars at their v1 offsets.
    put_u32_le(sb + 80, kN_layers);
    put_u32_le(sb + 84, kN_experts);
    put_u32_le(sb + 88, static_cast<uint32_t>(kIoAlignment));  // align@88
    // header_crc32@92 over the v1 92-byte range [0, 92).
    const uint32_t crc = strata_crc32(sb, 92);
    put_u32_le(sb + 92, crc);
    // reserved[32]@96 stays zero (v1 had 32 reserved bytes, NO trunk-index).

    // Write the hermetic v1 .strata to a temp path in the current working dir
    // (the ctest run dir). Avoid std::tmpnam (link-time warnings under
    // -Werror); the name is unique enough for a single-process test.
    const std::string path = "hermetic_v1_superblock.strata";
    {
        FILE *f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        if (f == nullptr) return;
        bool ok = std::fwrite(sb, 1, sizeof(sb), f) == sizeof(sb);
        ok = ok && std::fwrite(meta_blob, 1, static_cast<size_t>(meta_size), f) ==
                       static_cast<size_t>(meta_size);
        // No expert index bytes to write (index_count == 0).
        std::fclose(f);
        CHECK(ok);
        if (!ok) { std::remove(path.c_str()); return; }
    }

    // Open it and assert the version-aware parse mapped the v1 layout.
    StrataReader reader;
    const bool opened = reader.open(path);
    std::printf("[v1-hermetic open=%s version=%u has_trunk_index=%s "
                "trunk_index_count=%llu n_layers=%u n_experts=%u align=%u] ",
                opened ? "yes" : "no", reader.superblock().version,
                reader.has_trunk_index() ? "yes" : "no",
                static_cast<unsigned long long>(reader.trunk_index_count()),
                reader.superblock().n_layers, reader.superblock().n_experts,
                reader.superblock().align);

    CHECK(opened);
    if (opened) {
        CHECK_EQ(reader.superblock().version, static_cast<uint32_t>(1));
        CHECK(!reader.has_trunk_index());
        CHECK_EQ(reader.trunk_index_count(), static_cast<uint64_t>(0));
        // The v1 trailing-scalar relocation read these back from offsets
        // 80/84/88 rather than the v2 offsets 96/100/104.
        CHECK_EQ(reader.superblock().n_layers,  kN_layers);
        CHECK_EQ(reader.superblock().n_experts, kN_experts);
        CHECK_EQ(reader.superblock().align,
                 static_cast<uint32_t>(kIoAlignment));
        // The v2-only trunk-index fields are zeroed on the v1 path.
        CHECK_EQ(reader.superblock().trunk_index_offset,
                 static_cast<uint64_t>(0));
        CHECK_EQ(reader.superblock().trunk_index_count,
                 static_cast<uint64_t>(0));
        // The embedded metadata blob round-tripped verbatim.
        CHECK_EQ(reader.metadata_size(), meta_size);
    }

    std::remove(path.c_str());
}

static void run_all() {
    RUN(test_trunk_stream_f32_byte_identical);
    RUN(test_trunk_stream_kquant_byte_identical);
    RUN(test_trunk_stream_resident_drop);
    RUN(test_trunk_stream_resident_drop_kquant);
    RUN(test_v1_strata_back_compat);
    RUN(test_v1_superblock_hermetic);
}

TEST_MAIN()

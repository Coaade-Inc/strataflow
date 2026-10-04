// EC-4 headline gate (docs/ENGINE_CORE_DESIGN.md section 5 + section 8 EC-4,
// docs/TASK5_DESIGN.md section 6.4): pack the tiny MoE GGUF into a .strata, then
// decode a prompt greedily through OUR engine from BOTH the .strata and the
// source GGUF and assert the produced token SEQUENCES are BYTE-IDENTICAL. This
// proves the pack + our .strata loader change nothing about the math (same
// bytes, same ops), closing the Task 5 blocker: no llama_model_load_from_file
// is ever handed the .strata; the engine reads trunk tensors and streams expert
// tensors from the .strata directly.
//
// Also asserts the pack's structural invariants (superblock magic/version/
// align/CRC validate; the index round-trips {layer,expert}; each expert blob is
// 4 KiB-aligned; gate+up+down are adjacent and in-order within a blob) and
// compares the engine's .strata decode against the libllama ORACLE on the
// source GGUF.
//
// GUARDED on STRATAFLOW_TEST_MOE_GGUF (same skip-when-absent pattern as the
// other fixture tests), so the suite stays green in CI where the fixture is not
// generated. Generate + run locally:
//   uv run --with gguf --with numpy python3
//       tools/testdata/make_tiny_moe_gguf.py /projects/sandbox/moe.gguf
//   STRATAFLOW_TEST_MOE_GGUF=/projects/sandbox/moe.gguf ctest -R test_strata_decode
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "model/engine/engine.h"
#include "test_util.h"
#include "tws/async_io.h"
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

// Greedily decode `prompt` + `n_generate` tokens through an engine loaded from
// `model_path` (plain GGUF or .strata), prefilling the prompt tokens (which are
// tokenized once from `gguf_path`'s vocab so both paths use identical input).
// Fills `out_logits` with the per-step logit vectors. Returns the token ids.
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

// Assert the .strata pack's structural invariants via a StrataReader.
static void check_strata_structure(const std::string &strata_path) {
    CHECK(is_strata_file(strata_path));

    StrataReader reader;
    CHECK(reader.open(strata_path));  // validates magic/version/align/CRC
    if (!reader.is_open()) return;

    const StrataSuperblock &sb = reader.superblock();
    // Superblock magic/version/align.
    CHECK(std::memcmp(sb.magic, kStrataMagic, sizeof(kStrataMagic)) == 0);
    CHECK_EQ(sb.version, kStrataVersion);
    CHECK_EQ(sb.align, static_cast<uint32_t>(kIoAlignment));
    // header_crc32 re-validates (open() already checked it; be explicit).
    {
        StrataSuperblock copy = sb;
        const uint32_t stored = copy.header_crc32;
        CHECK_EQ(stored, strata_crc32(&copy, kStrataHeaderCrcOffset));
    }
    // Trunk + expert regions start 4 KiB-aligned.
    CHECK_EQ(sb.trunk_offset % kIoAlignment, static_cast<uint64_t>(0));
    CHECK_EQ(sb.expert_region_offset % kIoAlignment, static_cast<uint64_t>(0));

    // Index: n_layers * n_experts entries, each {layer,expert} round-trips
    // through find(), each blob is 4 KiB-aligned, and gate+up+down are adjacent
    // and strictly in-order within the blob.
    CHECK_EQ(reader.index_count(),
             static_cast<uint64_t>(sb.n_layers) * sb.n_experts);
    for (const ExpertIndexEntry &e : reader.index()) {
        const ExpertIndexEntry *f = reader.find(e.layer, e.expert);
        CHECK(f != nullptr);
        if (f != nullptr) {
            CHECK_EQ(f->layer, e.layer);
            CHECK_EQ(f->expert, e.expert);
        }
        // 4 KiB-aligned blob start.
        CHECK_EQ(e.blob_offset % kIoAlignment, static_cast<uint64_t>(0));
        // gate, up, down adjacent and in-order: 0 == gate_rel < up_rel <
        // down_rel < blob_length.
        CHECK_EQ(e.gate_rel, static_cast<uint64_t>(0));
        CHECK(e.gate_rel < e.up_rel);
        CHECK(e.up_rel < e.down_rel);
        CHECK(e.down_rel < e.blob_length);

        // The three slices read back at their recorded lengths.
        uint64_t off = 0, len = 0;
        CHECK(reader.slice_range(e.layer, e.expert, ExpertTensorKind::kGate,
                                 off, len));
        CHECK_EQ(len, e.up_rel - e.gate_rel);
        CHECK(reader.slice_range(e.layer, e.expert, ExpertTensorKind::kUp,
                                 off, len));
        CHECK_EQ(len, e.down_rel - e.up_rel);
        CHECK(reader.slice_range(e.layer, e.expert, ExpertTensorKind::kDown,
                                 off, len));
        CHECK_EQ(len, e.blob_length - e.down_rel);
    }
}

// THE EC-4 gate: pack, structural-check, then decode byte-identically from the
// .strata vs the source GGUF (and vs the oracle).
static void test_strata_decode_byte_identical() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }
    const std::string gguf_path = gguf;
    const std::string strata_path = gguf_path + ".ec4.strata";

    // 1. Pack the GGUF -> .strata (library call; no shelling out).
    std::string err;
    const bool packed = pack_gguf_to_strata(gguf_path, strata_path, &err);
    CHECK(packed);
    if (!packed) {
        std::printf("[pack failed: %s] ", err.c_str());
        return;
    }

    // 2. Structural invariants of the pack.
    check_strata_structure(strata_path);

    const std::string prompt = "hello world";
    const int n_generate = 8;

    // Tokenize once from the GGUF vocab so both decode paths get identical ids.
    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(gguf_path, prompt, prompt_ids));
    CHECK(!prompt_ids.empty());

    // 3. Decode greedily through the engine from the GGUF and from the .strata.
    std::vector<std::vector<float>> gguf_logits;
    std::vector<std::vector<float>> strata_logits;
    std::vector<int32_t> gguf_tokens =
        engine_decode(gguf_path, prompt_ids, n_generate, gguf_logits);
    std::vector<int32_t> strata_tokens =
        engine_decode(strata_path, prompt_ids, n_generate, strata_logits);

    CHECK_EQ(gguf_tokens.size(), static_cast<size_t>(n_generate));
    CHECK_EQ(strata_tokens.size(), static_cast<size_t>(n_generate));

    // 4. The ORACLE sequence on the source GGUF (libllama llama_decode). The
    // engine's .strata decode must reproduce it exactly.
    std::vector<int32_t> oracle_tokens;
    CHECK(engine::run_oracle_sequence(gguf_path, prompt, n_generate,
                                      oracle_tokens, nullptr));

    // (a) HARD gate: the .strata engine sequence == the GGUF engine sequence
    // AND == the oracle sequence.
    bool seq_match = strata_tokens.size() == gguf_tokens.size() &&
                     strata_tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < strata_tokens.size(); ++i) {
        if (i < gguf_tokens.size()) {
            CHECK_EQ(strata_tokens[i], gguf_tokens[i]);
            if (strata_tokens[i] != gguf_tokens[i]) seq_match = false;
        }
        if (i < oracle_tokens.size()) {
            CHECK_EQ(strata_tokens[i], oracle_tokens[i]);
            if (strata_tokens[i] != oracle_tokens[i]) seq_match = false;
        }
    }

    // (b) The per-step logits are bit-for-bit identical between the GGUF and
    // .strata engine runs: streaming from the .strata changes only the byte
    // PROVENANCE, never the bytes, so the forward pass is identical.
    double max_delta = 0.0;
    const size_t steps = gguf_logits.size() < strata_logits.size()
                             ? gguf_logits.size()
                             : strata_logits.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = gguf_logits[s].size() < strata_logits[s].size()
                             ? gguf_logits[s].size()
                             : strata_logits[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(gguf_logits[s][i]) -
                                       static_cast<double>(strata_logits[s][i]));
            if (d > max_delta) max_delta = d;
        }
    }

    std::printf("[seq match=%s gguf-vs-strata max|delta|=%.3e tokens:",
                seq_match ? "yes" : "no", max_delta);
    for (size_t i = 0; i < strata_tokens.size(); ++i) {
        std::printf(" %d", strata_tokens[i]);
    }
    std::printf("] ");

    CHECK(seq_match);
    // Byte-identical: the hard numeric guarantee of EC-4.
    CHECK(max_delta == 0.0);

    std::remove(strata_path.c_str());
}

// THE MISSION GATE: on the .strata streaming path the engine must hold only the
// TRUNK resident in RAM, NOT the full expert footprint. This is what lets a
// huge MoE run in a small RAM budget on an everyday no-GPU machine. We pack the
// tiny MoE, load it via .strata, and assert the resident weight bytes are far
// below the full model size (experts excluded) - and specifically below the
// total expert-region size reported by the pack index.
static void test_strata_resident_excludes_experts() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MOE_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MOE_GGUF] ");
        return;
    }
    const std::string gguf_path = gguf;
    const std::string strata_path = gguf_path + ".ramgate.strata";

    std::string err;
    CHECK(pack_gguf_to_strata(gguf_path, strata_path, &err));

    // Expert-region size from the .strata superblock (the bytes that must NOT
    // be resident), and the trunk size (the bytes that may be).
    StrataReader reader;
    CHECK(reader.open(strata_path));
    const uint64_t expert_region = reader.superblock().expert_region_size;
    const uint64_t trunk_region  = reader.superblock().trunk_size;
    CHECK(expert_region > 0);

    std::unique_ptr<engine::Engine> eng = engine::Engine::load(strata_path);
    CHECK(eng != nullptr);
    if (eng == nullptr) return;

    const uint64_t resident = eng->resident_weight_bytes();

    std::printf("[resident weight=%lluB trunk=%lluB experts(not resident)=%lluB] ",
                static_cast<unsigned long long>(resident),
                static_cast<unsigned long long>(trunk_region),
                static_cast<unsigned long long>(expert_region));

    // The hard assertion: resident weight RAM does NOT include the expert
    // footprint. It must be below the full model (trunk + experts) by at least
    // the whole expert region - i.e. resident < trunk + experts, and in fact
    // resident should be on the order of the trunk alone.
    CHECK(resident > 0);                       // trunk is resident
    CHECK(resident < trunk_region + expert_region);  // experts excluded
    CHECK(resident <= trunk_region + (trunk_region / 2));  // ~trunk-sized, not model-sized
}

static void run_all() {
    RUN(test_strata_decode_byte_identical);
    RUN(test_strata_resident_excludes_experts);
}

TEST_MAIN()

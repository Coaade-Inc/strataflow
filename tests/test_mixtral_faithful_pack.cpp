// Defect-1 regression gate (the real-Mixtral OOM): the packer must NEVER write
// a structurally-valid 0-expert .strata for a model whose GGUF metadata
// declares a MoE. On the real 19 GB Mixtral every expert tensor missed the
// name-based classifier, so the packer silently pushed them to the resident
// trunk and emitted a .strata with n_layers=0/n_experts=0/0 index entries; the
// engine then held the whole model resident -> SIGKILL (OOM). The exact
// name/shape trigger is NOT reproducible offline (every Mixtral-faithful
// fixture at real dims classifies fine), so this test pins the TWO provable,
// testable halves of the fix:
//
//   1. HAPPY PATH (guarded on STRATAFLOW_TEST_MIXTRAL_FAITHFUL_GGUF): a
//      Mixtral-faithful MoE still packs to a .strata whose superblock reports
//      the EXPECTED n_layers==2 / n_experts==8 / index_count==16 (the experts
//      were classified, NOT dropped to trunk), and the engine's .strata decode
//      is byte-identical to the engine GGUF decode (max|delta|==0) and matches
//      the libllama oracle sequence (EC-4). Mirrors tests/test_strata_decode.
//
//   2. LOUD-FAILURE PATH (needs no large fixture): a synthesized GGUF that
//      DECLARES llama.expert_count=8 but whose expert tensors are deliberately
//      renamed so the name-based classifier misses them must make
//      pack_gguf_to_strata return false with a non-empty err that mentions the
//      declared expert count - i.e. the OOM-bomb .strata is refused, loudly.
//      This is the direct negative assertion of the metadata cross-check.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "model/engine/engine.h"
#include "test_util.h"
#include "tws/async_io.h"
#include "tws/strata_file.h"
#include "tws/strata_format.h"
#include "tws/strata_pack.h"

#include <ggml.h>
#include <gguf.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <process.h>
#else
#  include <unistd.h>
#endif

using namespace sf;

// A temp filename in the current working directory (the test's build tree,
// inside the workspace), never an absolute path or /tmp - matches the repo
// convention in tests/test_async_io.cpp so the test is portable across runners
// (CI, the GitHub Actions matrix, dev boxes) where /projects/sandbox/ may be
// absent or read-only. Unique per process so parallel ctest runs don't collide.
static std::string temp_filename(const char *stem) {
    return std::string(stem) + "_" +
           std::to_string(
#if defined(_WIN32)
               static_cast<unsigned long>(_getpid())
#else
               static_cast<unsigned long>(::getpid())
#endif
               );
}

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
// from `model_path` (plain GGUF or .strata). Fills `out_logits` with per-step
// logit vectors. Mirrors tests/test_strata_decode.cpp engine_decode().
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

// HAPPY PATH: the Mixtral-faithful fixture packs with its experts classified
// (NOT dropped to trunk) and decodes byte-identically to the GGUF + oracle.
static void test_mixtral_faithful_pack_classifies_and_decodes() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MIXTRAL_FAITHFUL_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MIXTRAL_FAITHFUL_GGUF] ");
        return;
    }
    const std::string gguf_path = gguf;
    const std::string strata_path = gguf_path + ".faithful.strata";

    // 1. Pack the faithful GGUF -> .strata.
    std::string err;
    const bool packed = pack_gguf_to_strata(gguf_path, strata_path, &err);
    CHECK(packed);
    if (!packed) {
        std::printf("[pack failed: %s] ", err.c_str());
        return;
    }

    // 2. The superblock must report the experts as CLASSIFIED, not dropped to
    // trunk: the make-mixtral-faithful-gguf defaults are n_layer=2, n_expert=8,
    // so index_count == n_layers * n_experts == 16. A 0-expert pack (the OOM
    // bomb) would report n_layers=0/n_experts=0/0 here.
    {
        StrataReader reader;
        CHECK(reader.open(strata_path));
        const StrataSuperblock &sb = reader.superblock();
        std::printf("[faithful superblock: %u layers, %u experts, %llu index] ",
                    sb.n_layers, sb.n_experts,
                    static_cast<unsigned long long>(reader.index_count()));
        CHECK_EQ(sb.n_layers, static_cast<uint32_t>(2));
        CHECK_EQ(sb.n_experts, static_cast<uint32_t>(8));
        CHECK_EQ(reader.index_count(), static_cast<uint64_t>(16));
    }

    const std::string prompt = "hello world";
    const int n_generate = 8;

    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(gguf_path, prompt, prompt_ids));
    CHECK(!prompt_ids.empty());

    // 3. Engine decode from GGUF and from .strata.
    std::vector<std::vector<float>> gguf_logits;
    std::vector<std::vector<float>> strata_logits;
    std::vector<int32_t> gguf_tokens =
        engine_decode(gguf_path, prompt_ids, n_generate, gguf_logits);
    std::vector<int32_t> strata_tokens =
        engine_decode(strata_path, prompt_ids, n_generate, strata_logits);

    CHECK_EQ(gguf_tokens.size(), static_cast<size_t>(n_generate));
    CHECK_EQ(strata_tokens.size(), static_cast<size_t>(n_generate));

    // 4. The ORACLE sequence on the source GGUF (libllama llama_decode).
    std::vector<int32_t> oracle_tokens;
    CHECK(engine::run_oracle_sequence(gguf_path, prompt, n_generate,
                                      oracle_tokens, nullptr));

    // (a) HARD gate: .strata engine seq == GGUF engine seq == oracle seq.
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

    // (b) Byte-identical logits between the GGUF and .strata engine runs.
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

    std::printf("[seq match=%s gguf-vs-strata max|delta|=%.3e] ",
                seq_match ? "yes" : "no", max_delta);

    CHECK(seq_match);
    CHECK(max_delta == 0.0);

    std::remove(strata_path.c_str());
}

// Write a tiny llama-arch GGUF to `path` that DECLARES llama.expert_count=8 but
// whose per-layer expert tensors are deliberately MISNAMED (ffn_gate_experts
// instead of ffn_gate_exps, etc.) so parse_expert_tensor_name misses every one.
// This reproduces the real-Mixtral failure mode (metadata says MoE, zero
// expert tensors recognized) in a form that is deterministic and offline.
// Returns false on any ggml/gguf error.
static bool write_misnamed_moe_gguf(const std::string &path) {
    const int64_t n_embd = 64, n_ff = 128, n_layer = 2, n_expert = 8;
    const int64_t n_head = 4, head_dim = n_embd / n_head;

    ggml_init_params ip{};
    ip.mem_size = 64u * 1024u * 1024u;
    ip.no_alloc = false;
    ggml_context *mc = ggml_init(ip);
    if (mc == nullptr) return false;
    gguf_context *gc = gguf_init_empty();
    if (gc == nullptr) { ggml_free(mc); return false; }

    std::mt19937 rng(1);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    auto add_f32 = [&](const std::string &name, int n_dims, const int64_t *ne) {
        ggml_tensor *t = ggml_new_tensor(mc, GGML_TYPE_F32, n_dims, ne);
        ggml_set_name(t, name.c_str());
        int64_t n = 1;
        for (int d = 0; d < n_dims; ++d) n *= ne[d];
        float *dst = static_cast<float *>(t->data);
        for (int64_t i = 0; i < n; ++i) dst[i] = dist(rng);
        gguf_add_tensor(gc, t);
    };

    gguf_set_val_str(gc, "general.architecture", "llama");
    gguf_set_val_str(gc, "general.name", "misnamed-moe");
    gguf_set_val_u32(gc, "llama.context_length", 128);
    gguf_set_val_u32(gc, "llama.embedding_length", (uint32_t)n_embd);
    gguf_set_val_u32(gc, "llama.block_count", (uint32_t)n_layer);
    gguf_set_val_u32(gc, "llama.feed_forward_length", (uint32_t)n_ff);
    gguf_set_val_u32(gc, "llama.attention.head_count", (uint32_t)n_head);
    gguf_set_val_u32(gc, "llama.attention.head_count_kv", (uint32_t)n_head);
    gguf_set_val_u32(gc, "llama.rope.dimension_count", (uint32_t)head_dim);
    gguf_set_val_f32(gc, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    // DECLARE a MoE: this is the authoritative signal the packer cross-checks.
    gguf_set_val_u32(gc, "llama.expert_count", (uint32_t)n_expert);
    gguf_set_val_u32(gc, "llama.expert_used_count", 2);

    const int64_t ne_embd[1] = {n_embd};
    for (int64_t l = 0; l < n_layer; ++l) {
        char p[32];
        std::snprintf(p, sizeof(p), "blk.%lld.", (long long)l);
        const std::string pre = p;
        add_f32(pre + "attn_norm.weight", 1, ne_embd);
        add_f32(pre + "ffn_norm.weight", 1, ne_embd);
        const int64_t ne_gate_inp[2] = {n_embd, n_expert};
        add_f32(pre + "ffn_gate_inp.weight", 2, ne_gate_inp);
        // MISNAMED expert tensors: '..._experts' instead of the recognized
        // '..._exps', so parse_expert_tensor_name returns valid=false for all
        // of them and the classifier produces ZERO experts.
        const int64_t ne_gate[3] = {n_embd, n_ff, n_expert};
        const int64_t ne_down[3] = {n_ff, n_embd, n_expert};
        const int64_t ne_up[3]   = {n_embd, n_ff, n_expert};
        add_f32(pre + "ffn_gate_experts.weight", 3, ne_gate);
        add_f32(pre + "ffn_down_experts.weight", 3, ne_down);
        add_f32(pre + "ffn_up_experts.weight", 3, ne_up);
    }

    const bool ok = gguf_write_to_file(gc, path.c_str(), false);
    gguf_free(gc);
    ggml_free(mc);
    return ok;
}

// LOUD-FAILURE PATH (the direct negative assertion of the metadata
// cross-check): a GGUF that declares a MoE but whose expert tensors are all
// misnamed must make pack_gguf_to_strata return false with a non-empty err
// that names the declared expert count - NOT silently emit a 0-expert .strata.
static void test_pack_fails_loudly_on_misclassified_moe() {
    // CWD-relative temp files (build tree, inside the workspace), NOT an
    // absolute /projects/sandbox/ path: the fixture is synthesized here and
    // removed below via std::remove, matching tests/test_async_io.cpp.
    const std::string gguf_path = temp_filename("sf_misnamed_moe") + ".gguf";
    const std::string strata_path = temp_filename("sf_misnamed_moe") + ".strata";
    std::remove(strata_path.c_str());

    const bool wrote = write_misnamed_moe_gguf(gguf_path);
    CHECK(wrote);
    if (!wrote) {
        std::printf("[could not synthesize misnamed MoE GGUF] ");
        return;
    }

    std::string err;
    const bool packed = pack_gguf_to_strata(gguf_path, strata_path, &err);
    // The pack MUST be refused.
    CHECK(!packed);
    // The diagnostic must be present and name the declared expert count so an
    // operator can see WHY (metadata says MoE, zero experts recognized).
    CHECK(!err.empty());
    const bool mentions_count = err.find("expert_count=8") != std::string::npos;
    CHECK(mentions_count);
    std::printf("[loud-failure err: %s] ", err.c_str());

    std::remove(gguf_path.c_str());
    std::remove(strata_path.c_str());
}

static void run_all() {
    RUN(test_pack_fails_loudly_on_misclassified_moe);
    RUN(test_mixtral_faithful_pack_classifies_and_decodes);
}

TEST_MAIN()

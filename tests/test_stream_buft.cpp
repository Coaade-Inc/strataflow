// Unit tests for the pure parts of strata_stream_buft WITHOUT llama/ggml:
// the expert tensor-name parser and the registration stats accounting. The
// buffer type's end-to-end behaviour (allocation, init_tensor registration,
// decode) is covered by the integration test in test_moe_stream.cpp, which
// drives it through a real llama load.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "test_util.h"
#include "tws/stream_buft.h"

using namespace sf;

static void test_parse_expert_tensor_names() {
    // The three stacked expert tensors llama.cpp emits for a Mixtral-style MoE
    // layer, with and without the ".weight" suffix.
    ExpertTensorId g = parse_expert_tensor_name("blk.0.ffn_gate_exps.weight");
    CHECK(g.valid);
    CHECK_EQ(g.layer, 0u);
    CHECK(g.kind == ExpertTensorKind::kGate);

    ExpertTensorId d = parse_expert_tensor_name("blk.7.ffn_down_exps");
    CHECK(d.valid);
    CHECK_EQ(d.layer, 7u);
    CHECK(d.kind == ExpertTensorKind::kDown);

    ExpertTensorId u = parse_expert_tensor_name("blk.12.ffn_up_exps.weight");
    CHECK(u.valid);
    CHECK_EQ(u.layer, 12u);
    CHECK(u.kind == ExpertTensorKind::kUp);

    ExpertTensorId gu =
        parse_expert_tensor_name("blk.3.ffn_gate_up_exps.weight");
    CHECK(gu.valid);
    CHECK_EQ(gu.layer, 3u);
    CHECK(gu.kind == ExpertTensorKind::kGateUp);

    // Multi-digit layer indices.
    ExpertTensorId big = parse_expert_tensor_name("blk.123.ffn_gate_exps.weight");
    CHECK(big.valid);
    CHECK_EQ(big.layer, 123u);
}

static void test_parse_rejects_non_experts() {
    // Non-expert tensors (attention, norms, router gate, dense FFN, embeddings)
    // must NOT parse as experts, so they stay on their normal buffer type.
    const char *non_experts[] = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate_inp.weight",   // the router, NOT an expert
        "blk.0.ffn_gate.weight",       // dense FFN, not stacked experts
        "blk.0.ffn_down.weight",
        "blk..ffn_gate_exps.weight",   // missing layer index
        "blk.0.ffn_sideways_exps.weight",
        "ffn_gate_exps.weight",        // missing blk prefix
        "",
    };
    for (const char *n : non_experts) {
        ExpertTensorId id = parse_expert_tensor_name(n);
        CHECK(!id.valid);
    }
}

static void test_identity_equality() {
    ExpertTensorId a = parse_expert_tensor_name("blk.2.ffn_up_exps.weight");
    ExpertTensorId b = parse_expert_tensor_name("blk.2.ffn_up_exps");
    CHECK(a == b);

    ExpertTensorId c = parse_expert_tensor_name("blk.2.ffn_down_exps.weight");
    CHECK(!(a == c));
}

static void test_name_and_singleton() {
    // The buffer type exposes a stable name and a stable (singleton) pointer.
    CHECK(std::string(strata_stream_buft_name()) == "strata-stream");
    auto *b1 = strata_stream_buft();
    auto *b2 = strata_stream_buft();
    CHECK(b1 == b2);
    CHECK(b1 != nullptr);
}

static void test_stats_reset() {
    reset_stream_buft_stats();
    const StreamBuftStats &s = stream_buft_stats();
    CHECK_EQ(s.registrations, 0u);
    CHECK_EQ(s.non_expert, 0u);
    CHECK_EQ(s.stored_bytes, 0u);
    CHECK_EQ(s.buffers, 0u);
}

static void run_all() {
    RUN(test_parse_expert_tensor_names);
    RUN(test_parse_rejects_non_experts);
    RUN(test_identity_equality);
    RUN(test_name_and_singleton);
    RUN(test_stats_reset);
}

TEST_MAIN()

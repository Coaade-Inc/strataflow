#!/usr/bin/env python3
# Generate a tiny Mixtral-style MoE GGUF with a LARGE trained context length so
# the KV-paging regression (tests/test_kv_paging.cpp
# test_kv_paging_bigctx_small_window) can reproduce the real-model regime: a
# large n_ctx with a resident window WELL BELOW it. Every other dimension stays
# tiny (same n_embd=32, n_head=4, n_layer=2, n_expert=8, top-2 as the sibling
# make_tiny_moe_gguf.py) so decode is fast and the fixture is small; only
# add_context_length is bumped to BIG_CTX. For testing only -- the weights are
# random, so the decoded tokens are garbage but deterministic.
#
# This is the large-n_ctx sibling of make_tiny_moe_gguf.py. The engine's KV
# paging reserves a resident ring of W columns (STRATAFLOW_ENGINE_KV_WINDOW)
# instead of the full BIG_CTX, so with W << BIG_CTX the resident_kv_bytes is a
# small fraction (W/n_ctx) of full_kv_bytes and the cold store (grown on demand)
# never pre-allocates the full n_ctx. On the real Mixtral (n_ctx=32768) the
# pre-fix engine committed an 8 GiB cold store up front at load and OOMed; this
# fixture reproduces that regime in miniature.
#
# Usage:
#   pip install gguf numpy        # or: uv run --with gguf --with numpy ...
#   python3 make_tiny_moe_bigctx_gguf.py /path/to/tiny_moe_bigctx.gguf
#   STRATAFLOW_TEST_BIGCTX_MOE_GGUF=/path/to/tiny_moe_bigctx.gguf \
#       ctest -R test_kv_paging
#
# Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
import sys

import numpy as np
import gguf

out = sys.argv[1] if len(sys.argv) > 1 else "tiny_moe_bigctx.gguf"

# BIG_CTX is far above any test window (the gate uses W=64/128): n_ctx >> W makes
# the W/n_ctx resident ratio and the cold-vs-full-n_ctx distinction unambiguous,
# while staying small enough to generate/run quickly in-sandbox.
BIG_CTX = 4096

n_embd, n_head, n_layer, n_ff = 32, 4, 2, 64
n_expert, n_expert_used = 8, 2
head_dim = n_embd // n_head

# Vocab: 3 control tokens + 256 byte-fallback tokens (required so the SPM
# tokenizer can resegment any input) + a handful of normal pieces.
specials = ["<unk>", "<s>", "</s>"]
byte_toks = [f"<0x{i:02X}>" for i in range(256)]
normal = ["\u2581hello", "\u2581world", "\u2581test", "\u2581the", "\u2581a"]
toks = specials + byte_toks + normal
n_vocab = len(toks)
types = (
    [gguf.TokenType.CONTROL] * 3
    + [gguf.TokenType.BYTE] * 256
    + [gguf.TokenType.NORMAL] * len(normal)
)

w = gguf.GGUFWriter(out, "llama")
w.add_name("tiny-moe-bigctx")
w.add_context_length(BIG_CTX)
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head)
w.add_rope_dimension_count(head_dim)
w.add_layer_norm_rms_eps(1e-5)
# The MoE knobs that flip ModelShape.is_moe and drive the expert override.
w.add_expert_count(n_expert)
w.add_expert_used_count(n_expert_used)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
w.add_tokenizer_model("llama")
w.add_token_list(toks)
w.add_token_scores([0.0] * n_vocab)
w.add_token_types(types)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_unk_token_id(0)

rng = np.random.default_rng(0)


def tensor(shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.02


w.add_tensor("token_embd.weight", tensor((n_vocab, n_embd)))
w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
w.add_tensor("output.weight", tensor((n_vocab, n_embd)))
for i in range(n_layer):
    p = f"blk.{i}."
    w.add_tensor(p + "attn_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor(p + "attn_q.weight", tensor((n_embd, n_embd)))
    w.add_tensor(p + "attn_k.weight", tensor((n_embd, n_embd)))
    w.add_tensor(p + "attn_v.weight", tensor((n_embd, n_embd)))
    w.add_tensor(p + "attn_output.weight", tensor((n_embd, n_embd)))
    w.add_tensor(p + "ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
    # Router + 3-D stacked expert tensors (Mixtral/llama-arch layout). The
    # strata_stream_buft matches these ffn_*_exps names via expert_ffn_regex().
    w.add_tensor(p + "ffn_gate_inp.weight", tensor((n_expert, n_embd)))
    w.add_tensor(p + "ffn_gate_exps.weight", tensor((n_expert, n_ff, n_embd)))
    w.add_tensor(p + "ffn_down_exps.weight", tensor((n_expert, n_embd, n_ff)))
    w.add_tensor(p + "ffn_up_exps.weight", tensor((n_expert, n_ff, n_embd)))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(
    f"wrote {out}: vocab {n_vocab}, {n_layer} layers, "
    f"{n_expert} experts (top-{n_expert_used}), n_ctx {BIG_CTX}"
)

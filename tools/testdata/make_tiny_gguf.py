#!/usr/bin/env python3
# Generate a tiny but VALID llama-architecture GGUF with random weights, so the
# GgmlModel real-load path can be exercised without downloading a multi-GB
# checkpoint. For testing only -- the weights are random, so output is garbage.
#
# Usage:
#   pip install gguf numpy        # or: uv run --with gguf --with numpy ...
#   python3 make_tiny_gguf.py /path/to/tiny.gguf
#   STRATAFLOW_TEST_GGUF=/path/to/tiny.gguf ctest -R test_model_load
#
# Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
import sys

import numpy as np
import gguf

out = sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf"

n_embd, n_head, n_layer, n_ff = 32, 4, 2, 64
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
w.add_name("tiny-test")
w.add_context_length(64)
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head)
w.add_rope_dimension_count(head_dim)
w.add_layer_norm_rms_eps(1e-5)
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
    # ne[0] = n_embd for gate/up, n_ff for down (llama.cpp's expected layout).
    w.add_tensor(p + "ffn_gate.weight", tensor((n_ff, n_embd)))
    w.add_tensor(p + "ffn_down.weight", tensor((n_embd, n_ff)))
    w.add_tensor(p + "ffn_up.weight", tensor((n_ff, n_embd)))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out} (vocab {n_vocab}, {n_layer} layers)")

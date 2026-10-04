#!/usr/bin/env python3
# Generate a tiny QUANTIZED llama-arch MoE GGUF (Q8_0 weights, F32 norms), so
# the engine's quantized-weight path can be validated against the oracle without
# a multi-GB download. Mirrors make_tiny_moe_gguf.py but quantizes the matmul
# weight tensors to Q8_0 (2D weights and the 3D stacked expert tensors).
#
# Usage:
#   uv run --with gguf --with numpy python3 make_tiny_quant_moe_gguf.py out.gguf
#   STRATAFLOW_TEST_QUANT_MOE_GGUF=out.gguf ctest -R test_engine_oracle
#
# Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
import sys

import numpy as np
import gguf

out = sys.argv[1] if len(sys.argv) > 1 else "tiny_quant_moe.gguf"

n_embd, n_head, n_layer, n_ff = 32, 4, 2, 64
n_expert, n_expert_used = 8, 2
head_dim = n_embd // n_head
QT = gguf.GGMLQuantizationType.Q8_0  # block size 32; our dims are multiples of 32

specials = ["<unk>", "<s>", "</s>"]
byte_toks = [f"<0x{i:02X}>" for i in range(256)]
normal = ["\u2581hello", "\u2581world", "\u2581the", "\u2581a", "\u2581of"]
toks = specials + byte_toks + normal
n_vocab = len(toks)
types = ([gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.BYTE] * 256
         + [gguf.TokenType.NORMAL] * len(normal))

w = gguf.GGUFWriter(out, "llama")
w.add_name("tiny-quant-moe")
w.add_context_length(256)
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head)
w.add_rope_dimension_count(head_dim)
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(n_expert)
w.add_expert_used_count(n_expert_used)
w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
w.add_tokenizer_model("llama")
w.add_token_list(toks)
w.add_token_scores([0.0] * n_vocab)
w.add_token_types(types)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_unk_token_id(0)

rng = np.random.default_rng(0)


def f32(shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.02


def q8(arr):
    # Quantize a float32 array (last dim multiple of 32) to Q8_0 bytes.
    return gguf.quantize(arr, QT)


# token_embd and output are commonly kept at full/half precision in real models,
# but Q8_0 is fine here and exercises the quantized get_rows / lm_head matmul.
# When raw_dtype is a quant type, add_tensor treats the passed array as the
# already-quantized BYTE array and infers the logical shape, so pass q8(...)
# directly (no raw_shape).
w.add_tensor("token_embd.weight", q8(f32((n_vocab, n_embd))), raw_dtype=QT)
w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))  # norm: F32
w.add_tensor("output.weight", q8(f32((n_vocab, n_embd))), raw_dtype=QT)
for i in range(n_layer):
    p = f"blk.{i}."
    w.add_tensor(p + "attn_norm.weight", np.ones(n_embd, dtype=np.float32))
    for nm, shp in [("attn_q", (n_embd, n_embd)), ("attn_k", (n_embd, n_embd)),
                    ("attn_v", (n_embd, n_embd)), ("attn_output", (n_embd, n_embd))]:
        w.add_tensor(p + nm + ".weight", q8(f32(shp)), raw_dtype=QT)
    w.add_tensor(p + "ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
    # Router stays F32 (small; keeps gating numerics clean).
    w.add_tensor(p + "ffn_gate_inp.weight", f32((n_expert, n_embd)))
    # Stacked expert tensors quantized per 2D expert slice.
    for nm, shp in [("ffn_gate_exps", (n_expert, n_ff, n_embd)),
                    ("ffn_down_exps", (n_expert, n_embd, n_ff)),
                    ("ffn_up_exps",   (n_expert, n_ff, n_embd))]:
        w.add_tensor(p + nm + ".weight", q8(f32(shp)), raw_dtype=QT)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out}: {n_layer} layers, {n_expert} experts (top-{n_expert_used}), "
      f"Q8_0 weights")

# SPIKE -- do not ship. Tiny Mixtral-style (llama arch) MoE GGUF for the
# Phase 3 streaming feasibility spike. Random weights, byte-fallback vocab.
import sys
import numpy as np
import gguf

out = sys.argv[1] if len(sys.argv) > 1 else "tiny_moe.gguf"
n_embd, n_head, n_layer, n_ff = 32, 4, 2, 64
n_expert, n_expert_used = 8, 2
head_dim = n_embd // n_head

specials = ["<unk>", "<s>", "</s>"]
byte_toks = [f"<0x{i:02X}>" for i in range(256)]
normal = ["\u2581hello", "\u2581world", "\u2581test", "\u2581the", "\u2581a"]
toks = specials + byte_toks + normal
n_vocab = len(toks)
types = ([gguf.TokenType.CONTROL]*3 + [gguf.TokenType.BYTE]*256
         + [gguf.TokenType.NORMAL]*len(normal))

w = gguf.GGUFWriter(out, "llama")
w.add_name("tiny-moe")
w.add_context_length(64)
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head)
w.add_rope_dimension_count(head_dim)
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(n_expert)
w.add_expert_used_count(n_expert_used)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
w.add_tokenizer_model("llama")
w.add_token_list(toks)
w.add_token_scores([0.0]*n_vocab)
w.add_token_types(types)
w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0)

rng = np.random.default_rng(0)
tt = lambda s: (rng.standard_normal(s).astype(np.float32) * 0.02)
w.add_tensor("token_embd.weight", tt((n_vocab, n_embd)))
w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
w.add_tensor("output.weight", tt((n_vocab, n_embd)))
for i in range(n_layer):
    p = f"blk.{i}."
    w.add_tensor(p+"attn_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor(p+"attn_q.weight", tt((n_embd, n_embd)))
    w.add_tensor(p+"attn_k.weight", tt((n_embd, n_embd)))
    w.add_tensor(p+"attn_v.weight", tt((n_embd, n_embd)))
    w.add_tensor(p+"attn_output.weight", tt((n_embd, n_embd)))
    w.add_tensor(p+"ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
    # router + 3D expert tensors (Mixtral/llama-arch layout)
    w.add_tensor(p+"ffn_gate_inp.weight", tt((n_expert, n_embd)))
    w.add_tensor(p+"ffn_gate_exps.weight", tt((n_expert, n_ff, n_embd)))
    w.add_tensor(p+"ffn_down_exps.weight", tt((n_expert, n_embd, n_ff)))
    w.add_tensor(p+"ffn_up_exps.weight", tt((n_expert, n_ff, n_embd)))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {out}: {n_layer} layers, {n_expert} experts (top-{n_expert_used})")

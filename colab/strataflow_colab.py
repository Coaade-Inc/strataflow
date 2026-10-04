#!/usr/bin/env python3
# StrataFlow on Google Colab - self-contained demo + real-hardware test.
#
# What this proves on a real (CPU-only) machine:
#   1. StrataFlow builds and runs its OWN ggml forward pass (not llama_decode).
#   2. A larger-than-toy F32 MoE is packed to .strata and decoded with ONLY the
#      trunk resident in RAM while experts stream from disk - the bounded-RAM,
#      no-GPU mission - and the resident weight bytes are reported to prove it.
#
# Honest limits (read before running):
#   - The engine today runs F32 llama-arch MoE and dense llama only. A downloaded
#     QUANTIZED real model (Q4/Q6) will NOT run yet (the engine's staging tensors
#     are F32). So this demo uses a generated F32 MoE that is genuinely larger
#     than the unit-test fixture (hundreds of MB), which is the real exercise of
#     the streaming path on real hardware.
#   - Colab free tier is ~12 GB RAM and ~70-100 GB ephemeral disk. Keep the
#     generated model well under the disk size.
#
# This script is meant to be driven by the cells in colab/README.md, but it also
# runs standalone: `python3 strataflow_colab.py --layers 24 --experts 32`.
# Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
import argparse
import os
import subprocess
import sys


def sh(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


def make_moe_gguf(path, n_layer, n_expert, n_embd=256, n_head=8, n_ff=512,
                  n_expert_used=4):
    """Write an F32 llama-arch MoE GGUF with random weights, scalable in size."""
    import numpy as np
    import gguf

    head_dim = n_embd // n_head
    specials = ["<unk>", "<s>", "</s>"]
    byte_toks = [f"<0x{i:02X}>" for i in range(256)]
    normal = ["\u2581hello", "\u2581world", "\u2581the", "\u2581a", "\u2581of"]
    toks = specials + byte_toks + normal
    n_vocab = len(toks)
    types = ([gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.BYTE] * 256
             + [gguf.TokenType.NORMAL] * len(normal))

    w = gguf.GGUFWriter(path, "llama")
    w.add_name("colab-moe")
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
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("llama")
    w.add_token_list(toks)
    w.add_token_scores([0.0] * n_vocab)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)

    rng = np.random.default_rng(0)
    t = lambda s: (rng.standard_normal(s).astype(np.float32) * 0.02)
    w.add_tensor("token_embd.weight", t((n_vocab, n_embd)))
    w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor("output.weight", t((n_vocab, n_embd)))
    for i in range(n_layer):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", np.ones(n_embd, dtype=np.float32))
        w.add_tensor(p + "attn_q.weight", t((n_embd, n_embd)))
        w.add_tensor(p + "attn_k.weight", t((n_embd, n_embd)))
        w.add_tensor(p + "attn_v.weight", t((n_embd, n_embd)))
        w.add_tensor(p + "attn_output.weight", t((n_embd, n_embd)))
        w.add_tensor(p + "ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
        w.add_tensor(p + "ffn_gate_inp.weight", t((n_expert, n_embd)))
        w.add_tensor(p + "ffn_gate_exps.weight", t((n_expert, n_ff, n_embd)))
        w.add_tensor(p + "ffn_down_exps.weight", t((n_expert, n_embd, n_ff)))
        w.add_tensor(p + "ffn_up_exps.weight", t((n_expert, n_ff, n_embd)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    mb = os.path.getsize(path) / (1024 * 1024)
    print(f"wrote {path}: {n_layer} layers, {n_expert} experts "
          f"(top-{n_expert_used}), n_embd={n_embd}, n_ff={n_ff} -> {mb:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/strataflow")
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-ff", type=int, default=512)
    ap.add_argument("--prompt", default="hello world from colab")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--workdir", default="/content",
                    help="where to write the generated .gguf/.strata")
    args = ap.parse_args()

    repo = args.repo
    build = os.path.join(repo, "build", "release")
    cli = os.path.join(build, "bin", "strataflow")
    pack = os.path.join(build, "bin", "strata-pack")
    os.makedirs(args.workdir, exist_ok=True)
    model = os.path.join(args.workdir, "colab_moe.gguf")
    strata = os.path.join(args.workdir, "colab_moe.strata")

    print("=" * 70)
    print("STEP 1: generate a larger-than-toy F32 MoE GGUF")
    print("=" * 70)
    make_moe_gguf(model, args.layers, args.experts, args.n_embd, n_ff=args.n_ff)

    print("\n" + "=" * 70)
    print("STEP 2: pack GGUF -> .strata (trunk + aligned per-expert blobs)")
    print("=" * 70)
    sh(f"{pack} {model} {strata}")

    print("\n" + "=" * 70)
    print("STEP 3: show the placement plan (hardware auto-detected)")
    print("=" * 70)
    sh(f"{cli} --model {strata} --plan", check=False)

    on_disk_mb = os.path.getsize(strata) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("STEP 4: decode from .strata (engine owns inference; experts stream)")
    print(f"  model on disk: {on_disk_mb:.0f} MiB")
    print("  watch for: 'engine: loaded .strata trunk ... experts stream' and")
    print("  the final 'stats:' line reporting resident weights + peak RSS.")
    print("=" * 70)
    # --expert-slots bounds how many experts are held resident at once. Here we
    # cap it well below the model's expert count so you can SEE bounded RAM: the
    # resident model-weight bytes stay near the trunk size while the on-disk
    # model is much larger.
    sh(f'{cli} --model {strata} --expert-slots 8 '
       f'--max-tokens {args.max_tokens} --prompt "{args.prompt}"', check=False)

    print("\n" + "=" * 70)
    print("WHAT THIS PROVES:")
    print(" - The model DECODES through StrataFlow's own ggml forward pass")
    print("   (no llama_decode, no GPU).")
    print(f" - The model is {on_disk_mb:.0f} MiB on disk, but the 'stats:' line")
    print("   above shows resident model weights far below that - experts stream")
    print("   from disk through a bounded cache. That is the whole mission:")
    print("   big model, small RAM, no GPU.")
    print(" - Output text is gibberish ONLY because the weights are random; what")
    print("   is proven is the mechanism, not model quality. Running real")
    print("   (quantized) pretrained models is the next milestone - the engine")
    print("   is F32-only today (see docs/ROADMAP.md).")


if __name__ == "__main__":
    main()

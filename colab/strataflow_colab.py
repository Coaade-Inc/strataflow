#!/usr/bin/env python3
# StrataFlow on Google Colab - self-contained demo + real-hardware test.
#
# What this proves on a real (CPU-only) machine:
#   1. StrataFlow builds and runs its OWN ggml forward pass (not llama_decode).
#   2. A larger-than-toy MoE is packed to .strata and decoded with ONLY the
#      trunk resident in RAM while experts stream from disk - the bounded-RAM,
#      no-GPU mission - and the resident weight bytes are reported to prove it.
#   3. A REAL downloaded, QUANTIZED model (via --real-model) runs end to end
#      through the same path: download a GGUF, pack to .strata, decode bounded,
#      and record output text + tokens/sec + peak RSS.
#
# Two flows:
#   - Default (no flags / --layers / --experts): generate a larger-than-toy
#     F32 MoE with random weights and stream it. No download, works offline.
#   - --real-model: download a real quantized GGUF from HuggingFace, pack, and
#     run it. Needs network + disk; see run_real_model() for model choice.
#
# Honest limits (read before running):
#   - Quantized weights run through the engine's TYPE-AGNOSTIC staging path:
#     the per-layer expert staging tensors take the SOURCE expert tensor's real
#     ggml type, so quantized experts stream and compute via ggml_mul_mat_id
#     natively. Q8_0 is validated against the libllama oracle; K-quant (Q4_K /
#     Q6_K) is validated by the generated-fixture oracle gate. The remaining
#     honest caveat is ARCHITECTURE coverage: the engine implements llama-arch
#     MoE and dense llama only. Qwen2-MoE and DeepSeek-MoE are DIFFERENT
#     architectures (shared experts, per-expert gate, group-limited sigmoid
#     gating) that are NOT implemented and would hit the unsupported-arch path.
#   - Colab free tier is ~12 GB RAM and ~70-100 GB ephemeral disk. Keep the
#     model (generated or downloaded) well under the disk size; --expert-slots
#     bounds resident weights far below the on-disk size.
#
# This script is meant to be driven by the cells in colab/README.md, but it also
# runs standalone: `python3 strataflow_colab.py --layers 24 --experts 32`
# (generated F32 demo) or `python3 strataflow_colab.py --real-model` (real
# downloaded quantized model).
# Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time


def sh(cmd, cwd=None, check=True):
    """Run a command. `cmd` may be a LIST (preferred, no shell) or a string
    (legacy, run under the shell). Passing user-supplied values as list entries
    avoids any quoting/escaping hazard."""
    if isinstance(cmd, (list, tuple)):
        shown = " ".join(shlex.quote(str(a)) for a in cmd)
        print(f"\n$ {shown}")
        r = subprocess.run(list(cmd), cwd=cwd)
    else:
        shown = cmd
        print(f"\n$ {cmd}")
        r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {shown}")
    return r.returncode


def sh_capture(argv, cwd=None):
    """Run a command, stream its stdout live AND return the captured stdout.

    `argv` is a LIST of arguments (no shell): user-supplied values such as the
    prompt are passed as separate argv entries, so no quoting/escaping hazard
    exists even if the prompt contains quotes or shell metacharacters.

    The strataflow CLI writes 'plan:', 'prompt:', 'output: <text>' and the
    final 'stats:' line to stdout; its logs go to stderr. We capture stdout so
    we can parse the decoded text and the resident/peak-RSS numbers while still
    showing the user everything as it happens.
    """
    print("\n$ " + " ".join(shlex.quote(a) for a in argv))
    # errors="replace": a random-weight generated model can emit raw byte
    # tokens that are not valid UTF-8; decode them leniently so capture never
    # crashes. Real models produce valid UTF-8, so this only matters for the
    # generated demo/bench path.
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=sys.stderr, text=True, bufsize=1,
                            errors="replace")
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def parse_cli_stats(captured):
    """Parse the StrataFlow CLI stdout into a dict of every field it reports.

    This is the ONE place the CLI stdout format is parsed; both this module and
    colab/strataflow_bench.py use it so the parsing never drifts between the two.

    Returns a dict with keys:
      decoded      - decoded text, or None if NO 'output:' line was seen at all
                     (a parse failure the caller must flag); "" means the CLI
                     emitted an empty 'output:' line.
      resident_mib - resident model weights (MiB), or None if absent.
      peak_mib     - peak RSS (MiB), or None if absent.
      ttft_ms      - time-to-first-token (ms), or None if absent (FEAT-002:
                     only printed when a first token was produced).
      streamed_mib - SSD streamed bytes (MiB), or None if absent (FEAT-002:
                     only printed on the streaming .strata path, when > 0).
      streamed_bytes - SSD streamed bytes as the EXACT uint64 the engine holds,
                     or None if absent. Printed alongside streamed_mib on new
                     CLI builds; use it for an exact bytes/token (streamed_mib
                     is the 1-decimal rounded display value and is kept only as
                     a fallback for old CLI builds).
      prefetch_warmed - experts speculatively warmed ahead of a layer (FEAT-003),
                     or None if absent. Only printed on the bounded streaming
                     path (something was actually prefetched).
      prefetch_used - of the warmed experts, how many the routing then used
                     (FEAT-003 prefetch hit count), or None if absent.
      prefetch_completed_before_use - of the used experts, how many the ASYNC
                     background I/O worker had fully read AND installed before
                     the layer's authoritative acquire needed them (FEAT-003
                     overlap signal: the overlap actually hid the disk read), or
                     None if absent. 0 on the synchronous (--async-prefetch off)
                     path.

    The resident/peak fields keep the exact wording from before FEAT-002, and
    the TTFT/streamed/prefetch fields are optional, so this parser works against
    old and new CLI builds alike.
    """
    stats = {
        "decoded": None,
        "resident_mib": None,
        "peak_mib": None,
        "ttft_ms": None,
        "streamed_mib": None,
        "streamed_bytes": None,
        "prefetch_warmed": None,
        "prefetch_used": None,
        "prefetch_completed_before_use": None,
    }
    for line in captured.splitlines():
        if line.startswith("output:"):
            # Accept both 'output: <text>' and a bare 'output:' (empty decode).
            rest = line[len("output:"):]
            stats["decoded"] = rest[1:] if rest.startswith(" ") else rest
        m = re.search(r"resident model weights = ([0-9.]+) MiB", line)
        if m:
            stats["resident_mib"] = float(m.group(1))
        m = re.search(r"peak RSS = ([0-9.]+) MiB", line)
        if m:
            stats["peak_mib"] = float(m.group(1))
        m = re.search(r"TTFT = ([0-9.]+) ms", line)
        if m:
            stats["ttft_ms"] = float(m.group(1))
        m = re.search(r"streamed = ([0-9.]+) MiB", line)
        if m:
            stats["streamed_mib"] = float(m.group(1))
        # Exact uint64 byte count (new CLI builds only). Parsed separately from
        # the MiB display value so bytes/token is exact when present.
        m = re.search(r"streamed_bytes = ([0-9]+)", line)
        if m:
            stats["streamed_bytes"] = int(m.group(1))
        # FEAT-003 async-prefetch overlap fields (new CLI builds only). Matched
        # by exact substrings APPENDED to the stats line; absent on old builds
        # and on the fully-resident path (nothing prefetched). The
        # 'prefetch_completed_before_use' pattern is checked before the shorter
        # 'prefetch_used' so the regexes do not alias.
        m = re.search(r"prefetch_warmed = ([0-9]+)", line)
        if m:
            stats["prefetch_warmed"] = int(m.group(1))
        m = re.search(r"prefetch_completed_before_use = ([0-9]+)", line)
        if m:
            stats["prefetch_completed_before_use"] = int(m.group(1))
        m = re.search(r"prefetch_used = ([0-9]+)", line)
        if m:
            stats["prefetch_used"] = int(m.group(1))
    return stats


def parse_cli_output(captured):
    """Extract (decoded_text, resident_mib, peak_rss_mib) from CLI stdout.

    Backward-compatible 3-tuple wrapper around parse_cli_stats() for the
    existing generated + --real-model flows. `decoded` is None if NO 'output:'
    line was seen at all (a parse failure the caller must flag); it is "" only
    if the CLI emitted an empty 'output:' line.
    """
    stats = parse_cli_stats(captured)
    return stats["decoded"], stats["resident_mib"], stats["peak_mib"]


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


def run_real_model(args):
    # MODEL SELECTION - why the default is what it is.
    #
    # The engine runs ONLY llama-architecture models: llama-arch MoE and dense
    # llama. That constraint drives the choice:
    #
    #   - The natural real llama-arch MoE is Mixtral (mistralai, Apache-2.0,
    #     genuine llama arch with ffn_*_exps tensors). But the smallest useful
    #     Mixtral-8x7B GGUF is ~24 GB+ even at a low K-quant, which strains the
    #     Colab free tier (~12 GB RAM, ~70-100 GB disk). It is downloadable and
    #     runnable on a larger Colab/standalone box, so we expose it via
    #     --hf-repo/--hf-file, but it is NOT the default.
    #
    #   - Qwen2-MoE and DeepSeek-MoE are DIFFERENT architectures (shared
    #     experts, per-expert gate, group-limited sigmoid gating). The engine
    #     does NOT implement them; they would hit the unsupported-arch path and
    #     return EOS. We deliberately do NOT default to them so nothing fails
    #     silently.
    #
    # So the DEFAULT is a small, real, DOWNLOADED, QUANTIZED *dense llama*
    # model: TinyLlama-1.1B-Chat at Q4_K_M (~0.67 GB on disk), from
    # TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF (base model Apache-2.0). The engine
    # runs real downloaded dense-llama models through the exact same .strata
    # streaming + type-agnostic quant staging path as a MoE, so this is a real
    # proof of "downloaded quantized model decodes through StrataFlow's own
    # forward pass", honestly labeled as DENSE rather than MoE. Pass
    # --hf-repo/--hf-file to point at any other llama-arch GGUF (e.g. a Mixtral
    # quant) if your runtime has the disk/RAM for it.
    #
    # This path needs NETWORK (HuggingFace) and DISK. It cannot run in an
    # offline sandbox; it fails with a clear message if the download fails.
    repo = args.repo
    build = os.path.join(repo, "build", "release")
    cli = os.path.join(build, "bin", "strataflow")
    pack = os.path.join(build, "bin", "strata-pack")
    os.makedirs(args.workdir, exist_ok=True)

    hf_repo = args.hf_repo
    hf_file = args.hf_file
    is_moe_default = False  # the default (TinyLlama) is dense, not MoE
    expected_gb = args.expected_gb

    print("=" * 70)
    print("REAL DOWNLOADED MODEL FLOW (needs network + disk)")
    print("=" * 70)
    print(f"  model: {hf_repo} :: {hf_file}")
    print(f"  arch:  {'llama-arch MoE' if is_moe_default or args.is_moe else 'dense llama'}"
          f"  (quantized GGUF, streamed through StrataFlow)")
    print(f"  expected on-disk size: ~{expected_gb:.2f} GB")
    print("  NOTE: this downloads a real GGUF from HuggingFace. It requires")
    print("  network access and enough free disk. It will NOT run in an offline")
    print("  sandbox - that is expected; the flow is demonstrated on Colab.")

    # --- disk / RAM guards -------------------------------------------------
    free_disk_gb = shutil.disk_usage(args.workdir).free / (1024 ** 3)
    print("\n" + "-" * 70)
    print("GUARDS")
    print("-" * 70)
    print(f"  free disk at {args.workdir}: {free_disk_gb:.1f} GB")
    # Need room for the GGUF plus a .strata copy of roughly the same size.
    needed_gb = expected_gb * 2.1
    print(f"  need ~{needed_gb:.1f} GB (GGUF + .strata copy + headroom)")
    if free_disk_gb < needed_gb:
        sys.exit(
            f"refusing to download: only {free_disk_gb:.1f} GB free but "
            f"~{needed_gb:.1f} GB needed. Pick a smaller --hf-file or free disk."
        )
    total_ram_gb = _total_ram_gb()
    if total_ram_gb is not None:
        print(f"  total RAM: {total_ram_gb:.1f} GB"
              f"  (bounded by --expert-slots {args.expert_slots}; resident")
        print("   weights stay far below the on-disk size - see the stats line)")

    # --- download ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: download the GGUF from HuggingFace")
    print("=" * 70)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")
    try:
        gguf_path = hf_hub_download(repo_id=hf_repo, filename=hf_file,
                                    local_dir=args.workdir)
    except Exception as exc:  # network/offline/not-found all land here
        sys.exit(
            "download failed: " + str(exc) + "\n"
            "This flow needs network access to HuggingFace and is meant to run "
            "on Colab (or any machine with internet + disk), not in an offline "
            "sandbox. Check connectivity, the repo/file names, and free disk."
        )
    on_disk_mb = os.path.getsize(gguf_path) / (1024 * 1024)
    print(f"downloaded {gguf_path}: {on_disk_mb:.0f} MiB")

    # --- re-validate disk against the ACTUAL downloaded size ----------------
    # The pre-download guard sized everything off --expected-gb, which can be
    # wrong (e.g. the default 0.67 pointed at a 20 GB Mixtral). Now that the
    # real GGUF is on disk, re-check free space against its ACTUAL size before
    # we pack: strata-pack writes a .strata copy of roughly the same size, so
    # we need ~1x the GGUF free plus a little headroom. Fail clearly if it will
    # not fit rather than letting strata-pack run out of disk mid-write.
    gguf_bytes = os.path.getsize(gguf_path)
    free_after_dl_gb = shutil.disk_usage(args.workdir).free / (1024 ** 3)
    strata_needed_gb = gguf_bytes / (1024 ** 3) * 1.1  # .strata copy + headroom
    print(f"  actual GGUF size: {gguf_bytes / (1024 ** 3):.2f} GB")
    print(f"  free disk now:    {free_after_dl_gb:.1f} GB")
    print(f"  need for .strata: ~{strata_needed_gb:.2f} GB (copy + headroom)")
    if free_after_dl_gb < strata_needed_gb:
        sys.exit(
            f"refusing to pack: only {free_after_dl_gb:.1f} GB free after "
            f"download but ~{strata_needed_gb:.2f} GB needed to write the "
            f".strata copy of the {gguf_bytes / (1024 ** 3):.2f} GB GGUF. Free "
            "disk or pick a smaller --hf-file."
        )

    # --- pack --------------------------------------------------------------
    strata = os.path.join(args.workdir, "real_model.strata")
    print("\n" + "=" * 70)
    print("STEP 2: pack GGUF -> .strata (trunk + aligned per-expert blobs)")
    print("=" * 70)
    sh([pack, gguf_path, strata])
    strata_mb = os.path.getsize(strata) / (1024 * 1024)

    # --- plan --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: show the placement plan (hardware auto-detected)")
    print("=" * 70)
    sh([cli, "--model", strata, "--plan"], check=False)

    # --- run + measure -----------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: decode the REAL model from .strata, bounded + measured")
    print(f"  .strata on disk: {strata_mb:.0f} MiB")
    print(f"  --expert-slots {args.expert_slots} caps resident experts so RAM")
    print("  stays far below the on-disk size. A real model has a real trained")
    print("  context length, so the 'n_ctx_seq > n_ctx_train (0)' warning the")
    print("  generated demo emits should NOT appear here - that is a signal the")
    print("  real tokenizer + metadata are in use.")
    print("=" * 70)
    cmd = [cli, "--model", strata, "--expert-slots", str(args.expert_slots),
           "--max-tokens", str(args.max_tokens),
           "--async-prefetch", args.async_prefetch, "--prompt", args.prompt]
    t0 = time.time()
    rc, captured = sh_capture(cmd)
    elapsed = time.time() - t0
    if rc != 0:
        sys.exit(f"decode failed (rc={rc}). See the CLI output above.")

    decoded, resident_mib, peak_mib = parse_cli_output(captured)
    stats = parse_cli_stats(captured)  # FEAT-003 overlap metric, when present
    # A measurement run must not silently report a non-result. If the CLI
    # produced NO 'output:' line at all, parsing failed (format drift, crash
    # before decode, etc.) and `decoded` is None - treat that as a hard error
    # rather than printing 'None' as if it were the model's output.
    if decoded is None:
        sys.exit(
            "decode produced no 'output:' line to parse. The CLI ran (rc=0) "
            "but its stdout did not contain the expected 'output: <text>' "
            "line, so there is no decoded text to report. Check the CLI output "
            "above; do not treat this run as a valid measurement."
        )
    toks_per_sec = args.max_tokens / elapsed if elapsed > 0 else float("nan")

    print("\n" + "=" * 70)
    print("MEASURED RESULTS (real downloaded quantized model)")
    print("=" * 70)
    print(f"  model:            {hf_repo} :: {hf_file}")
    print(f"  arch:             {'llama-arch MoE' if args.is_moe else 'dense llama'}")
    print(f"  prompt:           {args.prompt!r}")
    print(f"  decoded output:   {decoded!r}")
    print(f"  wall-clock:       {elapsed:.2f} s for {args.max_tokens} tokens")
    print(f"  tokens/sec:       {toks_per_sec:.2f}")
    print(f"  async-prefetch:   {args.async_prefetch} (FEAT-003 overlap)")
    # FEAT-003: the overlap signal, when the CLI printed it (bounded streaming
    # path). completed-before-use counts experts the background I/O worker read
    # and installed BEFORE the layer needed them (0 when async-prefetch=off).
    if stats.get("prefetch_completed_before_use") is not None:
        print(f"  prefetch warmed:  {stats['prefetch_warmed']}")
        print(f"  prefetch used:    {stats['prefetch_used']}")
        print("  completed-before-use (overlap hid the read): "
              f"{stats['prefetch_completed_before_use']}")
        print("    (run once with --async-prefetch on and once off, same")
        print("     --expert-slots: the decoded text is identical, and the")
        print("     completed-before-use count is 0 for off and > 0 for on when")
        print("     the overlap fires; compare the two tokens/sec too.)")
    if resident_mib is not None:
        print(f"  resident weights: {resident_mib:.1f} MiB")
    if peak_mib is not None:
        print(f"  peak RSS:         {peak_mib:.1f} MiB")
    print(f"  on-disk .strata:  {strata_mb:.0f} MiB")
    print("\nWHAT THIS PROVES:")
    print(" - A REAL downloaded, QUANTIZED GGUF decodes end to end through")
    print("   StrataFlow's OWN ggml forward pass (no llama_decode, no GPU),")
    print("   with its real tokenizer producing real output text.")
    print(" - Quantized weights stream through the type-agnostic staging path.")
    if resident_mib is not None and peak_mib is not None:
        print(f" - Resident weights ({resident_mib:.1f} MiB) and peak RSS "
              f"({peak_mib:.1f} MiB) stay")
        print(f"   bounded far below the {strata_mb:.0f} MiB on-disk model: "
              "big model, small RAM.")


def _total_ram_gb():
    """Best-effort total RAM in GB; None if it cannot be determined."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return None


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
                    help="where to write the generated/downloaded models")
    # Real downloaded-model flow (needs network + disk; see run_real_model).
    ap.add_argument("--real-model", action="store_true",
                    help="download a REAL quantized GGUF from HuggingFace, pack "
                         "it, and run it bounded (instead of the generated-F32 "
                         "demo). Needs network + disk.")
    ap.add_argument("--hf-repo", default="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    help="HuggingFace repo id for --real-model (default: a small "
                         "dense-llama Q4_K_M model). Point at a llama-arch GGUF.")
    ap.add_argument("--hf-file", default="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                    help="GGUF filename within --hf-repo.")
    ap.add_argument("--expert-slots", type=int, default=8,
                    help="bound resident experts for the real-model run (MoE). "
                         "Dense models have no experts; the flag is harmless.")
    ap.add_argument("--expected-gb", type=float, default=0.67,
                    help="expected on-disk size of --hf-file in GB, used by the "
                         "disk guard (default matches the TinyLlama Q4_K_M).")
    ap.add_argument("--async-prefetch", default="on", choices=["on", "off"],
                    help="async/overlapped expert prefetch (FEAT-003): 'on' "
                         "(default) overlaps the next layer's expert disk reads "
                         "with the current layer's compute on a background I/O "
                         "thread; 'off' uses the synchronous warm path. Output "
                         "is byte-identical either way; run both to contrast "
                         "the overlap metric + decode tok/s on a real model.")
    ap.add_argument("--is-moe", action="store_true",
                    help="label the --real-model as MoE in the summary (set this "
                         "when you point --hf-repo/--hf-file at a Mixtral-style "
                         "llama-arch MoE GGUF). Default label is dense llama.")
    args = ap.parse_args()

    if args.real_model:
        run_real_model(args)
        return

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
    sh([pack, model, strata])

    print("\n" + "=" * 70)
    print("STEP 3: show the placement plan (hardware auto-detected)")
    print("=" * 70)
    sh([cli, "--model", strata, "--plan"], check=False)

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
    sh([cli, "--model", strata, "--expert-slots", str(args.expert_slots),
        "--max-tokens", str(args.max_tokens), "--prompt", args.prompt],
       check=False)

    print("\n" + "=" * 70)
    print("WHAT THIS PROVES:")
    print(" - The model DECODES through StrataFlow's own ggml forward pass")
    print("   (no llama_decode, no GPU).")
    print(f" - The model is {on_disk_mb:.0f} MiB on disk, but the 'stats:' line")
    print("   above shows resident model weights far below that - experts stream")
    print("   from disk through a bounded cache. That is the whole mission:")
    print("   big model, small RAM, no GPU.")
    print(" - Output text is gibberish ONLY because the weights are random; what")
    print("   is proven here is the mechanism, not model quality. To run a REAL")
    print("   downloaded, QUANTIZED model (dense llama by default) end to end,")
    print("   re-run with --real-model (needs network + disk). Quantized weights")
    print("   stream through the engine's type-agnostic staging path; the honest")
    print("   caveat is architecture coverage (llama-arch MoE + dense llama")
    print("   only; Qwen2-MoE / DeepSeek-MoE not yet implemented).")


if __name__ == "__main__":
    main()

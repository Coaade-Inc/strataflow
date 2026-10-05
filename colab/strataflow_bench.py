#!/usr/bin/env python3
# StrataFlow benchmark harness - runs a MATRIX of configs and emits real
# measured numbers as a human-readable table plus machine-readable JSON + CSV.
#
# What this measures, per run (one (model-size x expert-slots) config):
#   - gen tok/s    : END-TO-END generate throughput = max_tokens / full
#                    sf_generate wall-clock. This INCLUDES prompt processing
#                    and the first-token latency, so it is NOT steady-state
#                    decode. Reported honestly as end-to-end generate rate.
#   - decode tok/s : STEADY-STATE decode throughput with the first token (and
#                    its TTFT) removed: (max_tokens - 1) / (wall-clock - TTFT).
#                    This is the figure the harness actually targets; it is
#                    only computable when the CLI reports TTFT and max_tokens>1.
#   - TTFT (ms)    : time-to-first-token, reported by the CLI (FEAT-002).
#   - resident MiB : resident model weights held in RAM (sf_runtime_stats).
#   - peak RSS MiB : peak process RSS during the run.
#   - on-disk MiB  : size of the packed .strata file.
#   - streamed MiB : SSD bytes streamed through StrataReader (FEAT-002).
#   - bytes/token  : exact streamed_bytes / max_tokens. Uses the raw uint64
#                    streamed_bytes the CLI reports (not the 1-decimal MiB
#                    display value), so the integer is exact, not quantized.
#
# DEFAULT matrix: a small GENERATED-F32 MoE swept across model sizes and
# --expert-slots. It is CPU-only, offline, fits the free Colab tier (~12 GB
# RAM, ~70-100 GB disk) and finishes in a few minutes. It runs in the offline
# sandbox too. Grow it with --layers-list / --experts-list / --slots-list /
# --max-tokens. The optional --real-model row downloads a real TinyLlama GGUF
# (network, Colab only); offline it is SKIPPED with a clear message, and the
# generated rows still produce a full table.
#
# What the generated numbers DO and DO NOT prove:
#   - They prove the MECHANISM and the bounded-RAM property: a model many
#     hundreds of MiB on disk decodes with StrataFlow's RESIDENT model weights
#     held far below the on-disk size, and the SSD bytes STREAMED per token move
#     with --expert-slots. Resident weights + streamed bytes are the bounded
#     quantities (the Phase 3 "big model, small RAM" exit criterion,
#     PHASE3_PLAN.md). Whole-process peak RSS is a separate figure: it carries a
#     fixed llama/ggml backend + vocab floor (hundreds of MiB) plus a per-layer
#     staging buffer sized to one layer's FULL expert count, so it does NOT move
#     with --expert-slots and can exceed on-disk for tiny models. The
#     exit-criteria block below reports both and attributes the mission to the
#     bounded quantities, never to a peak-RSS percentage.
#   - They do NOT prove the absolute tokens/sec ladder in PLAN.md section 2:
#     that ladder is for LARGE REAL models on NVMe. The generated toy has
#     random weights and tiny dimensions, so its decode tok/s is only a
#     mechanism proxy, not a comparable throughput figure. We say so and do not
#     fabricate
#     a comparison where no target applies.
#
# Usage:
#   python3 colab/strataflow_bench.py                 # default generated matrix
#   python3 colab/strataflow_bench.py --real-model    # add one real TinyLlama row
#   python3 colab/strataflow_bench.py --layers-list 8,12 --slots-list 2,8,0
# Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
import argparse
import csv
import json
import os
import sys
import time

# Reuse the shared helpers so parsing + generation logic lives in ONE place and
# cannot drift from the main Colab driver.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strataflow_colab as sfc  # noqa: E402

SCHEMA_VERSION = 1


def default_workdir():
    """/content on Colab, /projects/sandbox in the sandbox, else cwd.

    Matches the --workdir convention in strataflow_colab.py (where models are
    written) while being usable offline in the sandbox.
    """
    if os.path.isdir("/content"):
        return "/content"
    if os.path.isdir("/projects/sandbox"):
        return "/projects/sandbox"
    return os.getcwd()


def model_label(cfg):
    return f"moe-L{cfg['layers']}-E{cfg['experts']}"


def ensure_gguf(args, cfg):
    """Generate the F32 MoE GGUF for this model size if not already present.

    One GGUF per (layers, experts, n_embd, n_ff) model size is reused across
    the expert-slots sweep so we do not regenerate it for every row.
    """
    label = model_label(cfg)
    gguf_path = os.path.join(args.workdir, f"bench_{label}.gguf")
    if not os.path.exists(gguf_path):
        print("\n" + "-" * 70)
        print(f"generate GGUF for {label} "
              f"(L={cfg['layers']} E={cfg['experts']} "
              f"n_embd={args.n_embd} n_ff={args.n_ff})")
        print("-" * 70)
        sfc.make_moe_gguf(gguf_path, cfg["layers"], cfg["experts"],
                          n_embd=args.n_embd, n_ff=args.n_ff)
    return gguf_path


def ensure_strata(args, gguf_path, cfg):
    """Pack the GGUF to .strata once per model size."""
    label = model_label(cfg)
    strata_path = os.path.join(args.workdir, f"bench_{label}.strata")
    if not os.path.exists(strata_path):
        print("\n" + "-" * 70)
        print(f"pack {label} GGUF -> .strata")
        print("-" * 70)
        sfc.sh([args.pack, gguf_path, strata_path])
    return strata_path


def _decode_tok_per_sec(max_tokens, elapsed, ttft_ms):
    """Steady-state decode throughput: first token (and its TTFT) removed.

    (max_tokens - 1) / (elapsed_seconds - ttft_seconds). Returns None when it
    cannot be computed honestly: no TTFT reported (old CLI build / no token),
    max_tokens <= 1, or a non-positive post-TTFT window. This is the figure the
    harness actually targets; the end-to-end 'gen tok/s' folds in prompt
    processing + TTFT and is reported separately.
    """
    if ttft_ms is None or max_tokens <= 1 or elapsed is None:
        return None
    decode_window = elapsed - ttft_ms / 1000.0
    if decode_window <= 0:
        return None
    return round((max_tokens - 1) / decode_window, 2)


def run_one(args, cfg, slots, async_prefetch="on"):
    """Run ONE generated-MoE config and return a result record (dict).

    Steps: generate GGUF (if needed) -> pack to .strata (if needed) -> run the
    CLI capturing stdout, timing the full sf_generate wall-clock -> parse the
    stats line. A missing 'output:' line is a HARD error for that run: the row
    is marked failed rather than emitting a fabricated measurement (mirrors
    run_real_model's guard in strataflow_colab.py).

    `async_prefetch` is 'on' (default) or 'off' and is threaded straight to the
    CLI's --async-prefetch flag so the harness can contrast the SAME bounded
    config with the background I/O overlap on versus off (FEAT-003).

    Throughput is reported two honest ways: 'gen tok/s' is end-to-end
    (max_tokens / full wall-clock, includes prompt + TTFT) and 'decode tok/s'
    is steady-state ((max_tokens - 1) / (wall-clock - TTFT), first token
    removed) when TTFT is available.
    """
    label = model_label(cfg)
    gguf_path = ensure_gguf(args, cfg)
    strata_path = ensure_strata(args, gguf_path, cfg)
    on_disk_mib = os.path.getsize(strata_path) / (1024 * 1024)

    print("\n" + "=" * 70)
    print(f"RUN {label}  expert-slots={slots}  "
          f"async-prefetch={async_prefetch}  max-tokens={args.max_tokens}")
    print("=" * 70)
    cmd = [args.cli, "--model", strata_path,
           "--expert-slots", str(slots),
           "--max-tokens", str(args.max_tokens),
           "--async-prefetch", async_prefetch,
           "--prompt", args.prompt]
    t0 = time.time()
    rc, captured = sfc.sh_capture(cmd)
    elapsed = time.time() - t0

    stats = sfc.parse_cli_stats(captured)
    record = {
        "model": label,
        "arch": "generated-f32-moe",
        "layers": cfg["layers"],
        "experts": cfg["experts"],
        "n_embd": args.n_embd,
        "n_ff": args.n_ff,
        "expert_slots": slots,
        "async_prefetch": async_prefetch,
        "max_tokens": args.max_tokens,
        "on_disk_mib": round(on_disk_mib, 1),
        "wall_seconds": round(elapsed, 3),
        "gen_tokens_per_sec": None,
        "decode_tokens_per_sec": None,
        "ttft_ms": stats["ttft_ms"],
        "resident_mib": stats["resident_mib"],
        "peak_rss_mib": stats["peak_mib"],
        "streamed_mib": stats["streamed_mib"],
        "streamed_bytes": stats["streamed_bytes"],
        "prefetch_warmed": stats["prefetch_warmed"],
        "prefetch_used": stats["prefetch_used"],
        "prefetch_completed_before_use":
            stats["prefetch_completed_before_use"],
        "decoded_text": stats["decoded"],
        "bytes_per_token": None,
        "ok": False,
        "error": None,
    }

    if rc != 0:
        record["error"] = f"CLI exited rc={rc}"
        print(f"  FAILED: {record['error']}")
        return record
    if stats["decoded"] is None:
        # No 'output:' line at all -> parse failure / crash before decode.
        # Do not treat this as a valid measurement.
        record["error"] = "no 'output:' line in CLI stdout"
        print(f"  FAILED: {record['error']}")
        return record

    _fill_throughput_and_bytes(record, args.max_tokens, elapsed, stats)
    record["ok"] = True
    return record


def _fill_throughput_and_bytes(record, max_tokens, elapsed, stats):
    """Populate the end-to-end + decode throughput and the EXACT bytes/token.

    gen tok/s is end-to-end (max_tokens / wall-clock). decode tok/s removes the
    first token + its TTFT. bytes/token uses the raw uint64 streamed_bytes when
    the CLI reports it (exact), falling back to the 1-decimal MiB display value
    only for old CLI builds that do not print streamed_bytes.
    """
    record["gen_tokens_per_sec"] = (round(max_tokens / elapsed, 2)
                                    if elapsed > 0 else None)
    record["decode_tokens_per_sec"] = _decode_tok_per_sec(
        max_tokens, elapsed, stats["ttft_ms"])
    if stats.get("streamed_bytes") is not None:
        # Exact: raw uint64 byte count straight from sf_runtime_stats.
        record["bytes_per_token"] = round(
            stats["streamed_bytes"] / max_tokens, 1)
    elif stats["streamed_mib"] is not None:
        # Fallback for old CLI builds: derived from the rounded MiB string, so
        # only approximate (quantized to ~0.05 MiB by the display rounding).
        streamed_bytes = stats["streamed_mib"] * 1024 * 1024
        record["bytes_per_token"] = round(streamed_bytes / max_tokens, 1)


def run_real_row(args):
    """Append ONE real downloaded TinyLlama (dense llama) row.

    Reuses strataflow_colab.run_real_model's download + disk/RAM guard + pack
    logic indirectly by replicating the capture-and-parse here with the SAME
    shared helpers. The download needs network; offline (sandbox) it is SKIPPED
    with a clear message rather than crashing the whole harness.
    """
    label = "real-tinyllama-dense"
    record = {
        "model": label,
        "arch": "real-dense-llama",
        "layers": None,
        "experts": None,
        "n_embd": None,
        "n_ff": None,
        "expert_slots": args.expert_slots,
        "async_prefetch": args.async_list[0],
        "max_tokens": args.max_tokens,
        "on_disk_mib": None,
        "wall_seconds": None,
        "gen_tokens_per_sec": None,
        "decode_tokens_per_sec": None,
        "ttft_ms": None,
        "resident_mib": None,
        "peak_rss_mib": None,
        "streamed_mib": None,
        "streamed_bytes": None,
        "prefetch_warmed": None,
        "prefetch_used": None,
        "prefetch_completed_before_use": None,
        "bytes_per_token": None,
        "ok": False,
        "error": None,
        "skipped": False,
    }

    print("\n" + "=" * 70)
    print("OPTIONAL REAL-MODEL ROW (needs network + disk; Colab only)")
    print("=" * 70)
    print(f"  model: {args.hf_repo} :: {args.hf_file}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        record["skipped"] = True
        record["error"] = ("huggingface_hub not installed - skipping the "
                           "real-model row (install it + run on Colab)")
        print(f"  SKIPPED: {record['error']}")
        return record

    # Disk guard, mirroring run_real_model: need the GGUF + a .strata copy.
    import shutil
    free_gb = shutil.disk_usage(args.workdir).free / (1024 ** 3)
    needed_gb = args.expected_gb * 2.1
    if free_gb < needed_gb:
        record["skipped"] = True
        record["error"] = (f"only {free_gb:.1f} GB free, need ~{needed_gb:.1f} "
                           "GB - skipping real-model row")
        print(f"  SKIPPED: {record['error']}")
        return record

    try:
        gguf_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file,
                                    local_dir=args.workdir)
    except Exception as exc:  # offline / network / not-found all land here
        record["skipped"] = True
        record["error"] = (f"download failed ({exc}) - skipping real-model row. "
                           "This path needs network (Colab), not an offline "
                           "sandbox.")
        print(f"  SKIPPED: {record['error']}")
        return record

    strata_path = os.path.join(args.workdir, "bench_real_model.strata")
    if not os.path.exists(strata_path):
        sfc.sh([args.pack, gguf_path, strata_path])
    on_disk_mib = os.path.getsize(strata_path) / (1024 * 1024)
    record["on_disk_mib"] = round(on_disk_mib, 1)

    cmd = [args.cli, "--model", strata_path,
           "--expert-slots", str(args.expert_slots),
           "--max-tokens", str(args.max_tokens),
           "--async-prefetch", args.async_list[0],
           "--prompt", args.prompt]
    t0 = time.time()
    rc, captured = sfc.sh_capture(cmd)
    elapsed = time.time() - t0
    stats = sfc.parse_cli_stats(captured)
    if rc != 0 or stats["decoded"] is None:
        record["error"] = (f"CLI rc={rc}" if rc != 0
                           else "no 'output:' line in CLI stdout")
        print(f"  FAILED: {record['error']}")
        return record
    record["wall_seconds"] = round(elapsed, 3)
    record["ttft_ms"] = stats["ttft_ms"]
    record["resident_mib"] = stats["resident_mib"]
    record["peak_rss_mib"] = stats["peak_mib"]
    record["streamed_mib"] = stats["streamed_mib"]
    record["streamed_bytes"] = stats["streamed_bytes"]
    record["prefetch_warmed"] = stats["prefetch_warmed"]
    record["prefetch_used"] = stats["prefetch_used"]
    record["prefetch_completed_before_use"] = \
        stats["prefetch_completed_before_use"]
    _fill_throughput_and_bytes(record, args.max_tokens, elapsed, stats)
    record["ok"] = True
    return record


def _fmt(value, spec="{}"):
    return "-" if value is None else spec.format(value)


def print_table(records):
    """Print a fixed-width ASCII results table, one row per config."""
    # 'gen tok/s' = end-to-end (max_tokens / full wall-clock, includes prompt +
    # TTFT). 'decode tok/s' = steady-state ((max_tokens-1)/(wall-clock-TTFT),
    # first token removed) - the figure the harness targets. See the README.
    header = ["model", "on-disk MiB", "slots", "async", "gen tok/s",
              "decode tok/s", "TTFT ms", "resident MiB", "peak RSS MiB",
              "streamed MiB", "bytes/tok", "cbu", "status"]
    rows = []
    for r in records:
        if r.get("skipped"):
            status = "SKIPPED"
        elif r["ok"]:
            status = "ok"
        else:
            status = "FAILED"
        slots = r["expert_slots"]
        slots_s = "auto" if slots == 0 else str(slots)
        rows.append([
            r["model"],
            _fmt(r["on_disk_mib"], "{:.1f}"),
            slots_s,
            _fmt(r.get("async_prefetch")),
            _fmt(r["gen_tokens_per_sec"], "{:.2f}"),
            _fmt(r["decode_tokens_per_sec"], "{:.2f}"),
            _fmt(r["ttft_ms"], "{:.1f}"),
            _fmt(r["resident_mib"], "{:.1f}"),
            _fmt(r["peak_rss_mib"], "{:.1f}"),
            _fmt(r["streamed_mib"], "{:.1f}"),
            _fmt(r["bytes_per_token"], "{:.0f}"),
            _fmt(r.get("prefetch_completed_before_use"), "{:d}"),
            status,
        ])

    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))
    print("\nnote: each row is a SINGLE run (no warmup/repeat), so gen tok/s,")
    print("decode tok/s and TTFT are single noisy samples. gen tok/s is")
    print("end-to-end (includes prompt + TTFT); decode tok/s removes the first")
    print("token and is the steady-state figure the harness targets.")
    print("'async' = --async-prefetch (background I/O overlap on/off, FEAT-003).")
    print("'cbu' = prefetch_completed_before_use: experts whose disk read the")
    print("background worker FINISHED and installed BEFORE the layer needed them")
    print("(the overlap that hid the read). It is 0 by construction when")
    print("async=off, so a nonzero cbu on async=on is the mechanism firing.")


def print_exit_criteria(records):
    """Report measured numbers against the applicable exit criteria, honestly.

    The 'big model, small RAM' mission is bounded by StrataFlow's OWN resident
    model weights and the SSD bytes it streams per token - BOTH move with
    --expert-slots. Whole-process peak RSS is a DIFFERENT quantity: it carries a
    fixed llama/ggml backend + vocab floor (hundreds of MiB, reproduced even by
    a 3.9 MiB model and a --plan-only run) PLUS a per-layer expert staging
    buffer sized to one layer's FULL expert count (not --expert-slots). So peak
    RSS does NOT move with --expert-slots and can exceed on-disk for small
    models. We report both, attribute the mission to the bounded quantities,
    and never present a peak-RSS percentage as the proof.
    """
    gen_ok = [r for r in records
              if r["ok"] and r["arch"] == "generated-f32-moe"]
    print("\n" + "=" * 70)
    print("AGAINST THE EXIT CRITERIA (honest)")
    print("=" * 70)
    if not gen_ok:
        print("  no successful generated rows to report.")
        return

    print("PHASE3_PLAN.md criterion 1 - 'big model, small RAM'. The quantities")
    print("that are BOUNDED and MOVE with --expert-slots are StrataFlow's own")
    print("resident model weights and the SSD bytes streamed per decode. These")
    print("are the mechanism. Measured on the generated MoE (watch resident +")
    print("streamed shrink/grow as slots change for a given model):")
    print(f"  {'model':<16} {'slots':<5} {'on-disk MiB':>11}  "
          f"{'resident MiB':>12}  {'streamed MiB':>12}  {'peak RSS MiB':>12}")
    for r in gen_ok:
        if r["on_disk_mib"] in (None, 0):
            continue
        slots_s = "auto" if r["expert_slots"] == 0 else str(r["expert_slots"])
        print(f"  {r['model']:<16} {slots_s:<5} "
              f"{r['on_disk_mib']:>11.1f}  "
              f"{_fmt(r['resident_mib'], '{:.1f}'):>12}  "
              f"{_fmt(r['streamed_mib'], '{:.1f}'):>12}  "
              f"{_fmt(r['peak_rss_mib'], '{:.1f}'):>12}")

    print("\nResident weights stay FAR below on-disk (experts are metadata-only")
    print("and stream through a bounded cache), and the streamed-MiB column")
    print("moves with --expert-slots: that movement IS the bound working.")

    print("\nWhole-process peak RSS (getrusage ru_maxrss) is reported for")
    print("transparency, NOT as the proof. It carries a FIXED llama/ggml backend")
    print("+ vocab floor (hundreds of MiB; reproduced by a 3.9 MiB model and by")
    print("a --plan-only run) plus a per-layer staging buffer sized to one")
    print("layer's FULL expert count. So it does NOT track --expert-slots (the")
    print("peak-RSS column above does not shrink as slots shrink, unlike the")
    print("streamed-MiB column) and can EXCEED on-disk for small models. That is")
    print("EXPECTED here, not a failure: these generated models are tiny, so the")
    print("fixed floor dominates.")

    # Explicitly flag, per row, where peak RSS >= on-disk instead of printing a
    # >100% figure as if it proved the mission.
    flagged = [r for r in gen_ok
               if r["peak_rss_mib"] is not None and r["on_disk_mib"]
               and r["peak_rss_mib"] >= r["on_disk_mib"]]
    if flagged:
        print("\n  NOTE: peak RSS >= on-disk on these tiny rows (fixed floor")
        print("  dominates; this is the floor, not an unbounded leak):")
        for r in flagged:
            slots_s = "auto" if r["expert_slots"] == 0 else str(r["expert_slots"])
            print(f"    {r['model']} slots={slots_s}: "
                  f"peak RSS {r['peak_rss_mib']:.1f} MiB "
                  f">= on-disk {r['on_disk_mib']:.1f} MiB")

    # Make the slots-vs-peak-RSS invariance explicit so a reader does not expect
    # peak RSS to track --expert-slots (Defect 2).
    _print_peak_rss_flatness(gen_ok)

    # Auto vs smallest fixed-slot contrast, per model. This is the FEAT-003
    # exit-criteria point: for the SAME generated model, the auto row (slots=0)
    # sizes residency from the free-RAM budget + the model's expert layout, so
    # when RAM holds the whole working set it caches it and re-reads far fewer
    # SSD bytes per token than a deliberately-too-small fixed slot count, which
    # evicts and re-streams experts every token.
    _print_auto_vs_small_contrast(gen_ok)

    # FEAT-003: async-prefetch overlap on vs off at the same bounded config.
    _print_async_on_vs_off_contrast(gen_ok)

    print("\nPLAN.md section 2 - absolute tokens/sec ladder: this ladder is for")
    print("LARGE REAL models on NVMe. The generated toy has random weights and")
    print("tiny dimensions, so its decode tok/s is a MECHANISM proxy only, NOT a")
    print("comparable throughput figure. No comparison is fabricated here.")
    print("Run the optional --real-model row (or a Mixtral quant on a larger")
    print("runtime) to measure throughput against that ladder on real weights.")


def _print_auto_vs_small_contrast(gen_ok):
    """Contrast the auto row against the SMALLEST fixed-slot row per model.

    For each generated model that was run at BOTH slots=auto (0) and at least
    one positive fixed slot count, report the smallest fixed-slot row next to
    the auto row so a reader sees, for the SAME model, that auto streams fewer
    SSD bytes per token (and is typically faster). This is a MECHANISM proxy on
    random weights, stated honestly: the win comes from auto caching the whole
    layer working set when RAM allows, so experts are not re-read every token.
    """
    by_model = {}
    for r in gen_ok:
        by_model.setdefault(r["model"], []).append(r)

    lines = []
    for model, rows in by_model.items():
        auto = next((r for r in rows if r["expert_slots"] == 0), None)
        fixed = [r for r in rows if r["expert_slots"] and r["expert_slots"] > 0]
        if auto is None or not fixed:
            continue
        smallest = min(fixed, key=lambda r: r["expert_slots"])
        if (auto.get("bytes_per_token") is None
                or smallest.get("bytes_per_token") is None):
            continue
        a_bpt = auto["bytes_per_token"]
        s_bpt = smallest["bytes_per_token"]
        ratio = (s_bpt / a_bpt) if a_bpt > 0 else None
        ratio_s = f"{ratio:.1f}x more" if ratio is not None else "-"
        lines.append(
            f"    {model}: auto streams {a_bpt:.0f} bytes/tok vs "
            f"slots={smallest['expert_slots']} streaming {s_bpt:.0f} "
            f"bytes/tok ({ratio_s} on the too-small fixed count).")
        a_dec = auto.get("decode_tokens_per_sec")
        s_dec = smallest.get("decode_tokens_per_sec")
        if a_dec is not None and s_dec is not None:
            faster = ("faster" if a_dec >= s_dec else "slower")
            lines.append(
                f"      decode tok/s: auto {a_dec:.2f} vs "
                f"slots={smallest['expert_slots']} {s_dec:.2f} "
                f"(auto is {faster} here; single noisy sample).")

    if not lines:
        return
    print("\nAUTO vs a deliberately-too-small fixed slot count (same model).")
    print("The auto row (slots=auto) sizes expert residency per model from the")
    print("measured free-RAM budget + the model's expert layout. When RAM holds")
    print("the whole layer working set it CACHES it, so it re-reads far fewer")
    print("SSD bytes per token than a too-small fixed count that evicts and")
    print("re-streams experts every token. Mechanism proxy (random weights), so")
    print("the decode tok/s is indicative, not an absolute throughput claim:")
    for ln in lines:
        print(ln)


def _print_async_on_vs_off_contrast(gen_ok):
    """Contrast async-prefetch ON vs OFF at the SAME (model, slots) config.

    For every (model, expert-slots) pair that was run at BOTH async=on and
    async=off, print them side by side so a reader sees the mechanism directly:
      - streamed bytes/token: the AUTHORITATIVE experts loaded are identical
        either way (ensure_layer_experts_resident is unchanged), so the DECODED
        TEXT is byte-identical on vs off - the invariant to confirm. Note the
        streamed-bytes COUNTER can be HIGHER with async on, because the
        background worker issues speculative reads that race eviction at a tight
        slot count; that is extra I/O work, not a correctness change.
      - completed-before-use (cbu): 0 when off (the warm is inline, not
        overlapped) and > 0 when on IF the worker finished a predicted read
        before the layer needed it. A nonzero cbu on the ON row is the overlap
        firing - prefetches completed ahead of use and hid the read latency.
      - decode tok/s + TTFT: reported honestly as single noisy samples on a tiny
        sandbox CPU where the per-layer compute window is small, so wall-clock
        gains are hardware-dependent; the MECHANISM (cbu > 0) is the proof here,
        not the absolute speed.
    """
    def key(r):
        return (r["model"], r["expert_slots"])

    pairs = {}
    for r in gen_ok:
        ap = r.get("async_prefetch")
        if ap in ("on", "off"):
            pairs.setdefault(key(r), {})[ap] = r

    contrasted = {k: v for k, v in pairs.items() if "on" in v and "off" in v}
    if not contrasted:
        return

    print("\nASYNC PREFETCH on vs off (same model, same expert-slots) - FEAT-003.")
    print("The authoritative expert load is unchanged, so the decoded text is")
    print("byte-identical on vs off (correctness invariant). The overlap signal")
    print("is completed-before-use (cbu): experts the background I/O worker read")
    print("and installed BEFORE the layer's acquire needed them (0 when off).")
    print("Absolute decode tok/s is hardware-dependent (tiny sandbox CPU => tiny")
    print("compute window to hide reads behind); the mechanism is cbu > 0:")
    for (model, slots), v in sorted(contrasted.items()):
        on, off = v["on"], v["off"]
        slots_s = "auto" if slots == 0 else str(slots)
        same_text = (on.get("decoded_text") == off.get("decoded_text"))
        print(f"    {model} slots={slots_s}:")
        print(f"      decoded text identical on-vs-off: "
              f"{'yes' if same_text else 'NO (unexpected!)'}")
        cbu_on = on.get("prefetch_completed_before_use")
        cbu_off = off.get("prefetch_completed_before_use")
        used_on = on.get("prefetch_used")
        print(f"      completed-before-use: on={_fmt(cbu_on)} "
              f"off={_fmt(cbu_off)} (of prefetch_used={_fmt(used_on)} on 'on')")
        print(f"      streamed bytes/tok:   on={_fmt(on['bytes_per_token'], '{:.0f}')} "
              f"off={_fmt(off['bytes_per_token'], '{:.0f}')} "
              f"(authoritative experts identical; 'on' may read more "
              f"speculatively)")
        d_on = on.get("decode_tokens_per_sec")
        d_off = off.get("decode_tokens_per_sec")
        print(f"      decode tok/s:         on={_fmt(d_on, '{:.2f}')} "
              f"off={_fmt(d_off, '{:.2f}')} "
              f"(single noisy sample; hardware-dependent)")
        t_on = on.get("ttft_ms")
        t_off = off.get("ttft_ms")
        print(f"      TTFT ms:              on={_fmt(t_on, '{:.1f}')} "
              f"off={_fmt(t_off, '{:.1f}')}")


def _print_peak_rss_flatness(gen_ok):
    """Show that peak RSS stays ~flat across the --expert-slots sweep per model.

    Groups the successful rows by model size and, when a model was swept across
    more than one slot setting, prints the peak-RSS spread so the reader SEES
    peak RSS not moving with slots (the EXPECTED consequence of the fixed floor
    + one-layer staging), while resident/streamed do move (shown above).
    """
    by_model = {}
    for r in gen_ok:
        if r["peak_rss_mib"] is not None:
            by_model.setdefault(r["model"], []).append(r)
    lines = []
    for model, rows in by_model.items():
        if len(rows) < 2:
            continue
        peaks = [r["peak_rss_mib"] for r in rows]
        spread = max(peaks) - min(peaks)
        lines.append(f"    {model}: peak RSS {min(peaks):.1f}-{max(peaks):.1f} "
                     f"MiB across {len(rows)} slot settings "
                     f"(spread {spread:.1f} MiB)")
    if lines:
        print("\n  peak RSS across the --expert-slots sweep (it does NOT track")
        print("  the slot budget - the fixed floor + one-layer staging dominate,")
        print("  so any spread is from the compute/staging buffer, not the slot")
        print("  count):")
        for ln in lines:
            print(ln)


def write_artifacts(args, records):
    """Write the matrix results to BOTH a JSON file and a CSV file."""
    meta = {
        "schema_version": SCHEMA_VERSION,
        "tool": "strataflow_bench",
        "generated_matrix": {
            "layers_list": args.layers_list,
            "experts_list": args.experts_list,
            "slots_list": args.slots_list,
            "async_list": args.async_list,
            "n_embd": args.n_embd,
            "n_ff": args.n_ff,
            "max_tokens": args.max_tokens,
            "prompt": args.prompt,
        },
        "records": records,
    }
    json_path = os.path.join(args.workdir, "strataflow_bench.json")
    csv_path = os.path.join(args.workdir, "strataflow_bench.csv")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    # CSV: a flat row per record with a stable column order.
    fields = ["model", "arch", "layers", "experts", "n_embd", "n_ff",
              "expert_slots", "async_prefetch", "max_tokens", "on_disk_mib",
              "wall_seconds", "gen_tokens_per_sec", "decode_tokens_per_sec",
              "ttft_ms", "resident_mib", "peak_rss_mib", "streamed_mib",
              "streamed_bytes", "prefetch_warmed", "prefetch_used",
              "prefetch_completed_before_use", "bytes_per_token", "ok", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    return json_path, csv_path


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip() != ""]


def parse_async_list(text):
    """Parse the --async-list value into a list of 'on'/'off' strings.

    Accepts 'on', 'off' (and the aliases 1/true, 0/false) in any order, e.g.
    'on,off'. Rejects anything else loudly so a typo cannot silently run only
    one side of the contrast.
    """
    out = []
    for raw in text.split(","):
        tok = raw.strip().lower()
        if tok == "":
            continue
        if tok in ("on", "1", "true"):
            out.append("on")
        elif tok in ("off", "0", "false"):
            out.append("off")
        else:
            sys.exit(f"--async-list: expected on/off tokens, got '{raw}'")
    return out or ["on"]


def main():
    ap = argparse.ArgumentParser(
        description="StrataFlow benchmark harness (matrix -> table + JSON/CSV).")
    ap.add_argument("--repo", default=None,
                    help="StrataFlow repo root (default: parent of this script).")
    ap.add_argument("--workdir", default=None,
                    help="where models + artifacts are written (default: "
                         "/content on Colab, /projects/sandbox in the sandbox).")
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-ff", type=int, default=512)
    ap.add_argument("--prompt", default="hello world from the strataflow bench")
    ap.add_argument("--max-tokens", type=int, default=16,
                    help="generation cap per run; gen tok/s = max-tokens / full "
                         "wall-clock, decode tok/s = (max-tokens-1)/(wall-clock"
                         "-TTFT) (default 16, free-tier quick).")
    # DEFAULT matrix: two model sizes x three expert-slots settings. Small
    # enough for the free Colab tier (a few hundred MiB on disk each) and quick.
    ap.add_argument("--layers-list", default="8,12",
                    help="comma list of layer counts (default 8,12).")
    ap.add_argument("--experts-list", default="16,32",
                    help="comma list of expert counts, paired with layers-list "
                         "by index (default 16,32).")
    ap.add_argument("--slots-list", default="2,8,0",
                    help="comma list of --expert-slots to sweep; 0 = auto "
                         "(default 2,8,0).")
    ap.add_argument("--async-list", default="on",
                    help="comma list of async-prefetch settings to sweep per "
                         "config ('on', 'off', or 'on,off' to contrast the "
                         "background I/O overlap on vs off at the SAME bounded "
                         "slot count; default 'on'). FEAT-003.")
    # Optional real-model row (Colab only).
    ap.add_argument("--real-model", action="store_true",
                    help="append ONE real downloaded TinyLlama (dense) row. "
                         "Needs network + disk; skipped cleanly when offline.")
    ap.add_argument("--hf-repo", default="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    ap.add_argument("--hf-file", default="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    ap.add_argument("--expert-slots", type=int, default=8,
                    help="expert-slots for the real-model row (default 8).")
    ap.add_argument("--expected-gb", type=float, default=0.67,
                    help="expected on-disk size of --hf-file (GB) for the disk "
                         "guard (default matches TinyLlama Q4_K_M).")
    args = ap.parse_args()

    if args.repo is None:
        args.repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.workdir is None:
        args.workdir = default_workdir()
    os.makedirs(args.workdir, exist_ok=True)

    build = os.path.join(args.repo, "build", "release")
    args.cli = os.path.join(build, "bin", "strataflow")
    args.pack = os.path.join(build, "bin", "strata-pack")
    for binp in (args.cli, args.pack):
        if not os.path.exists(binp):
            sys.exit(f"missing build artifact: {binp}\n"
                     "Build the release target first (see colab/README.md).")

    args.layers_list = parse_int_list(args.layers_list)
    args.experts_list = parse_int_list(args.experts_list)
    args.slots_list = parse_int_list(args.slots_list)
    args.async_list = parse_async_list(args.async_list)
    if len(args.layers_list) != len(args.experts_list):
        sys.exit("--layers-list and --experts-list must have the same length "
                 "(they pair by index to define each model size).")

    model_cfgs = [{"layers": l, "experts": e}
                  for l, e in zip(args.layers_list, args.experts_list)]

    print("=" * 70)
    print("STRATAFLOW BENCHMARK HARNESS")
    print("=" * 70)
    print(f"  workdir:      {args.workdir}")
    print(f"  model sizes:  {[model_label(c) for c in model_cfgs]}")
    print(f"  expert-slots: {args.slots_list} (0 = auto)")
    print(f"  async-list:   {args.async_list} (--async-prefetch sweep)")
    print(f"  max-tokens:   {args.max_tokens}")
    print(f"  real-model:   {'yes (optional row)' if args.real_model else 'no'}")

    records = []
    for cfg in model_cfgs:
        for slots in args.slots_list:
            for ap in args.async_list:
                records.append(run_one(args, cfg, slots, async_prefetch=ap))

    if args.real_model:
        records.append(run_real_row(args))

    print_table(records)
    print_exit_criteria(records)
    json_path, csv_path = write_artifacts(args, records)

    print("\n" + "=" * 70)
    print("ARTIFACTS")
    print("=" * 70)
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("  (download these from the Colab file browser to keep the numbers.)")

    n_ok = sum(1 for r in records if r["ok"])
    n_fail = sum(1 for r in records
                 if not r["ok"] and not r.get("skipped"))
    print(f"\n  {n_ok} ok, {n_fail} failed, "
          f"{len(records) - n_ok - n_fail} skipped, {len(records)} total.")
    # Non-zero exit only if a NON-skipped row failed, so an offline run (where
    # only the real-model row is skipped) still exits 0.
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

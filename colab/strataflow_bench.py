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
#     hundreds of MiB on disk decodes with resident weights + peak RSS held far
#     below the on-disk size, and that budget moves with --expert-slots. That
#     is the Phase 3 "big model, small RAM" exit criterion (PHASE3_PLAN.md).
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


def run_one(args, cfg, slots):
    """Run ONE generated-MoE config and return a result record (dict).

    Steps: generate GGUF (if needed) -> pack to .strata (if needed) -> run the
    CLI capturing stdout, timing the full sf_generate wall-clock -> parse the
    stats line. A missing 'output:' line is a HARD error for that run: the row
    is marked failed rather than emitting a fabricated measurement (mirrors
    run_real_model's guard in strataflow_colab.py).

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
    print(f"RUN {label}  expert-slots={slots}  max-tokens={args.max_tokens}")
    print("=" * 70)
    cmd = [args.cli, "--model", strata_path,
           "--expert-slots", str(slots),
           "--max-tokens", str(args.max_tokens),
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
    header = ["model", "on-disk MiB", "slots", "gen tok/s", "decode tok/s",
              "TTFT ms", "resident MiB", "peak RSS MiB", "streamed MiB",
              "bytes/tok", "status"]
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
            _fmt(r["gen_tokens_per_sec"], "{:.2f}"),
            _fmt(r["decode_tokens_per_sec"], "{:.2f}"),
            _fmt(r["ttft_ms"], "{:.1f}"),
            _fmt(r["resident_mib"], "{:.1f}"),
            _fmt(r["peak_rss_mib"], "{:.1f}"),
            _fmt(r["streamed_mib"], "{:.1f}"),
            _fmt(r["bytes_per_token"], "{:.0f}"),
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


def print_exit_criteria(records):
    """Report measured numbers against the applicable exit criteria, honestly."""
    gen_ok = [r for r in records
              if r["ok"] and r["arch"] == "generated-f32-moe"]
    print("\n" + "=" * 70)
    print("AGAINST THE EXIT CRITERIA (honest)")
    print("=" * 70)
    if not gen_ok:
        print("  no successful generated rows to report.")
        return

    print("PHASE3_PLAN.md criterion 1 - 'big model, small RAM': peak RSS and")
    print("resident weights must stay FAR BELOW the on-disk model size, and")
    print("move with --expert-slots. Measured on the generated MoE:")
    for r in gen_ok:
        if r["peak_rss_mib"] is None or r["on_disk_mib"] in (None, 0):
            continue
        ratio = r["peak_rss_mib"] / r["on_disk_mib"] if r["on_disk_mib"] else 0
        slots_s = "auto" if r["expert_slots"] == 0 else str(r["expert_slots"])
        print(f"  {r['model']:<16} slots={slots_s:<4} "
              f"on-disk={r['on_disk_mib']:.1f} MiB  "
              f"resident={_fmt(r['resident_mib'], '{:.1f}')} MiB  "
              f"peak RSS={r['peak_rss_mib']:.1f} MiB  "
              f"(peak = {ratio * 100:.0f}% of on-disk)")

    print("\nPLAN.md section 2 - absolute tokens/sec ladder: this ladder is for")
    print("LARGE REAL models on NVMe. The generated toy has random weights and")
    print("tiny dimensions, so its decode tok/s is a MECHANISM proxy only, NOT a")
    print("comparable throughput figure. No comparison is fabricated here.")
    print("Run the optional --real-model row (or a Mixtral quant on a larger")
    print("runtime) to measure throughput against that ladder on real weights.")


def write_artifacts(args, records):
    """Write the matrix results to BOTH a JSON file and a CSV file."""
    meta = {
        "schema_version": SCHEMA_VERSION,
        "tool": "strataflow_bench",
        "generated_matrix": {
            "layers_list": args.layers_list,
            "experts_list": args.experts_list,
            "slots_list": args.slots_list,
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
              "expert_slots", "max_tokens", "on_disk_mib", "wall_seconds",
              "gen_tokens_per_sec", "decode_tokens_per_sec", "ttft_ms",
              "resident_mib", "peak_rss_mib", "streamed_mib", "streamed_bytes",
              "bytes_per_token", "ok", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    return json_path, csv_path


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip() != ""]


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
    print(f"  max-tokens:   {args.max_tokens}")
    print(f"  real-model:   {'yes (optional row)' if args.real_model else 'no'}")

    records = []
    for cfg in model_cfgs:
        for slots in args.slots_list:
            records.append(run_one(args, cfg, slots))

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

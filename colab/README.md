# Running StrataFlow on Google Colab

A step-by-step guide to build StrataFlow on a free Colab CPU runtime and run
models through its streaming engine - proving the "big model, bounded RAM, no
GPU" path on a real machine. There are two flows:

1. **Generated-F32 demo** (default, no download): generate a larger-than-toy
   F32 MoE and stream it. Works offline, exercises the streaming path.
2. **Real downloaded quantized model** (`--real-model`): download a real
   quantized GGUF from HuggingFace, pack it to `.strata`, run it bounded, and
   record output text + tokens/sec + peak RSS.

## Before you start

- Use a **CPU runtime** (Runtime -> Change runtime type -> CPU). StrataFlow's
  engine is CPU-only today; a GPU runtime is not needed and is not used.
- Honest limits (see [`../docs/ROADMAP.md`](../docs/ROADMAP.md)):
  - Quantized weights **run**. The engine stages each per-layer expert tensor
    at the SOURCE tensor's real ggml type, so quantized experts stream and
    compute via `ggml_mul_mat_id` natively. **Q8_0** is validated against the
    libllama oracle; **K-quant (Q4_K / Q6_K)** is validated by the
    generated-fixture oracle gate. The remaining honest caveat is
    **architecture coverage**: the engine implements **llama-arch MoE and dense
    llama only**. Qwen2-MoE and DeepSeek-MoE are different architectures (shared
    experts, per-expert gate, group-limited sigmoid gating) that are not yet
    implemented, so do not point the real-model flow at them.
  - First build compiles llama.cpp/ggml: a few minutes.
  - Free Colab is ~12 GB RAM, ~70-100 GB ephemeral disk. The limiter here is
    **disk, not RAM**: keep the model (generated or downloaded) under the disk
    size, and `--expert-slots` keeps resident weights far below the on-disk size
    (see "Big model, small RAM" just below).

### Big model, small RAM (how a model bigger than your RAM runs)

This is the exact question people ask: "this model is 24 GB, how can it run on
12 GB (or 8 GB) of RAM?" The answer is that **RAM holds only the working set,
not the whole model.**

For a Mixture-of-Experts model, only a small fraction of the weights (the trunk
plus the top-k active experts per layer) is used for any one token. StrataFlow
keeps just that working set resident and streams the cold experts from disk per
token. The full model only has to be **reachable on disk**, not held in memory.
StrataFlow's **resident model weights** (trunk only) stay far below the on-disk
size, and the **SSD bytes streamed per token** are bounded by `--expert-slots`:
a tighter slot budget streams fewer bytes. Those two quantities are the bound.

What this does and does NOT mean for whole-process peak RSS. Measured in this
repo (see [`../docs/ROADMAP.md`](../docs/ROADMAP.md) and
[`../CHANGELOG.md`](../CHANGELOG.md)): a 785 MiB model and a 1177 MiB model both
run at ~132 MiB peak RSS. That ~132 MiB is essentially a **fixed** llama/ggml
backend + vocab process floor (reproduced by a 3.9 MiB model and by a
`--plan-only` run with no decode), plus a one-layer expert staging buffer sized
to a layer's full expert count, plus the ggml compute buffer. So peak RSS is
roughly flat across model size for similar layer counts, but it does **not**
move with `--expert-slots` and can **exceed** on-disk for small models - that is
the fixed floor, not an unbounded leak, and not proof of the mission by itself.
The mission is carried by the bounded resident weights + streamed bytes above.
So on an 8 GB machine the constraints are (1) enough disk for the model file and
its `.strata` copy, and (2) streaming speed (SSD read throughput sets tok/s).
**More memory buys speed, not capability; the output is identical at every
memory size.**

### Which real model, and why

The engine runs **llama-architecture** models only. That drives the choice:

- The natural real llama-arch MoE is **Mixtral** (mistralai, Apache-2.0, genuine
  llama arch). Its ~24 GB+ figure (even at a low K-quant) is a **disk/download**
  number, NOT a resident-RAM number. StrataFlow holds only the trunk resident
  and streams the experts from disk per token, so the resident model weights and
  the streamed bytes per token are bounded by `--expert-slots`, not by model
  size - a Mixtral quant does NOT need ~24 GB of resident RAM (whole-process peak
  RSS still carries the fixed backend + vocab floor plus one-layer staging, as
  above). The real free-tier limiters are **disk space** (you
  need room for the ~24 GB GGUF plus a similar-size `.strata` copy, roughly
  ~48 GB, which free Colab's ~70-100 GB ephemeral disk CAN hold) and
  **download time + streaming throughput** (SSD read speed sets tok/s). So a
  Mixtral quant IS runnable on free Colab, just slow: a long download and pack,
  disk-heavy, with `--expert-slots` keeping resident RAM low. See Cell 8. You can
  also point the flow at a Mixtral quant on a larger runtime via
  `--hf-repo/--hf-file --is-moe`.
- **Qwen2-MoE** and **DeepSeek-MoE** are different architectures the engine does
  not implement; do not use them (they would hit the unsupported-arch path).
- So the **default** `--real-model` is a small, real, downloaded, **quantized
  dense llama** model: **TinyLlama-1.1B-Chat at Q4_K_M** (~0.67 GB), from
  `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (base model Apache-2.0). The engine
  runs real downloaded dense-llama models through the exact same `.strata`
  streaming + type-agnostic quant staging path as a MoE, so this is a real proof
  of "downloaded quantized model decodes through StrataFlow's own forward pass",
  honestly labeled **dense**, not MoE.

## The cells

Paste each block into its own Colab cell and run top to bottom.

### Cell 1 - clone the repo (with the vendored submodule)

```bash
%%bash
cd /content
rm -rf strataflow
git clone --recurse-submodules https://github.com/Coaade-Inc/strataflow.git
cd strataflow
git submodule update --init --recursive
echo "cloned at $(git -C /content/strataflow rev-parse --short HEAD)"
```

> If the repo is private, you will be prompted for credentials, or clone over an
> authenticated URL. If `--recurse-submodules` did not fetch llama.cpp, the
> explicit `git submodule update --init --recursive` line covers it.

### Cell 2 - install build tools + Python deps

```bash
%%bash
# Colab usually already has cmake/ninja/build-essential. If so you can skip the
# apt lines. The "W: Skipping acquire of configured file .../Sources" warning
# from Colab's preconfigured R2U/CRAN apt repo is HARMLESS - the install still
# succeeds; ignore it. Use `apt-get update ... || true` so a flaky mirror never
# fails the cell.
apt-get -qq update || true
apt-get -qq install -y cmake ninja-build build-essential > /dev/null || true
# gguf + numpy drive the generated-F32 demo; huggingface_hub drives the real
# downloaded-model flow (Cells 7-8). Installing all three up front is harmless.
pip -q install gguf numpy huggingface_hub
cmake --version | head -1
```

### Cell 3 - build StrataFlow (Release; CPU backend)

```bash
%%bash
cd /content/strataflow
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j"$(nproc)"
echo "built:"; ls -lh build/release/bin
```

This compiles llama.cpp/ggml + StrataFlow. Expect a few minutes the first time.

### Cell 4 - run the demo (generate a real-size MoE, pack, stream-decode)

```bash
%%bash
cd /content/strataflow
# --layers / --experts control the model size. Defaults make a ~0.5-1 GB F32
# MoE: big enough that streaming matters, small enough for Colab's disk.
python3 colab/strataflow_colab.py --layers 24 --experts 32 --max-tokens 16
```

Watch for, in the output:
- `engine: loaded .strata trunk (... experts stream via StrataReader)` - the
  engine is streaming experts, not holding them resident.
- The decoded tokens (garbage text - the weights are random; what matters is it
  runs end to end through our own forward pass).
- The final `stats: resident model weights = ... MiB, peak RSS = ... MiB` line.
  The model is ~1.2 GB on disk, but the resident model weights stay near the
  trunk size (tens of MiB) because experts stream from disk through a bounded
  cache (`--expert-slots`). That gap is the bounded-RAM point - no external
  timing tool needed.

### Cell 5 (optional) - make it bigger and watch RAM stay bounded

```bash
%%bash
cd /content/strataflow
ls -lh /content/colab_moe.strata
# Push the model larger (more experts = more on-disk weight, same trunk):
python3 colab/strataflow_colab.py --layers 24 --experts 64 --max-tokens 8
ls -lh /content/colab_moe.strata
```

As `--experts` grows, the `.strata` file (on disk) grows a lot while
StrataFlow's resident model weights stay close to the trunk size - that is
StrataFlow doing its job. (Whole-process peak RSS carries the fixed backend +
vocab floor plus one-layer staging, so it does not shrink to the trunk; the
resident-weights line is the bounded quantity. See "Big model, small RAM".)

### Cell 6 (optional) - run the unit tests on Colab's real hardware

```bash
%%bash
cd /content/strataflow
# Generate the committed fixtures the guarded tests look for.
uv() { python3 -m pip -q install gguf numpy >/dev/null 2>&1; python3 "$@"; }
python3 tools/testdata/make_tiny_moe_gguf.py /content/moe.gguf
python3 tools/testdata/make_tiny_dense_gguf.py /content/dense.gguf
cd build/release
STRATAFLOW_TEST_MOE_GGUF=/content/moe.gguf \
STRATAFLOW_TEST_GGUF=/content/moe.gguf \
STRATAFLOW_TEST_DENSE_GGUF=/content/dense.gguf \
  ctest --output-on-failure
```

### Cell 7 - run a REAL downloaded quantized model (default: dense TinyLlama)

```bash
%%bash
cd /content/strataflow
# Downloads a real quantized GGUF from HuggingFace, packs it to .strata, runs it
# bounded, and prints the decoded text + tokens/sec + peak RSS. The default is
# TinyLlama-1.1B-Chat Q4_K_M (~0.67 GB, dense llama, Apache-2.0 base). The flow
# checks free disk first and refuses clearly if the model will not fit.
python3 colab/strataflow_colab.py --real-model \
  --prompt "The capital of France is" --max-tokens 24 --expert-slots 8
```

Watch for, in the output:
- The download size and the `GUARDS` block (free disk vs needed disk, total RAM).
- No `n_ctx_seq > n_ctx_train (0)` warning - a real model carries a real trained
  context length, so that noisy warning from the generated demo should be gone.
  Its absence is a signal the real tokenizer + metadata are in use.
- The `MEASURED RESULTS` block: the real decoded output text, the measured
  `tokens/sec` (wall-clock around the decode divided by `--max-tokens`), the
  `resident weights` and `peak RSS`, and the on-disk `.strata` size. Resident
  weights and peak RSS stay bounded far below the on-disk size.

### Cell 8 (optional) - point it at a real llama-arch MoE (Mixtral)

Mixtral's ~24 GB is a **disk** number, not a resident-RAM number: `--expert-slots`
keeps StrataFlow's resident model weights and streamed bytes per token bounded
regardless of model size, so this is NOT gated by free Colab's ~12 GB RAM.
(Whole-process peak RSS still carries the fixed llama/ggml backend + vocab floor
plus one-layer staging and does not move with `--expert-slots`; see "Big model,
small RAM" above.) The real limiter is **disk** - you need room for the
~24 GB GGUF plus a similar-size `.strata` copy (~48 GB), which free Colab's
~70-100 GB ephemeral disk CAN hold - plus the long download/pack and SSD
streaming throughput (which sets tok/s). So this runs on free Colab, just
slowly. Check free disk FIRST, because the flow's disk guard will refuse clearly
if it will not fit:

```bash
%%bash
# How much ephemeral disk is free on this runtime? Mixtral needs ~48 GB free
# (the ~24 GB GGUF plus a similar-size .strata copy).
df -h /content
```

```bash
%%bash
cd /content/strataflow
# Mixtral is a genuine llama-arch MoE (Apache-2.0). It is disk-heavy, not
# RAM-heavy: the ~24 GB+ GGUF is a download/disk cost, while --expert-slots
# bounds resident RAM far below that. The flow's disk guard will refuse clearly
# if the GGUF + .strata copy will not fit the free disk. Set --is-moe so the
# summary labels it correctly, and raise --expected-gb to match the chosen quant.
python3 colab/strataflow_colab.py --real-model --is-moe \
  --hf-repo TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
  --hf-file mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf \
  --expected-gb 20.4 --expert-slots 2 \
  --prompt "Explain mixture of experts in one sentence:" --max-tokens 24
```

As with the generated demo, `--expert-slots` bounds how many experts are held
resident at once, so a multi-GB MoE decodes with StrataFlow's resident model
weights and streamed bytes far below the on-disk size - the whole mission on a
real, downloaded, quantized MoE. On the free tier expect it to be slow (long
download + pack, disk-bound streaming), not blocked by RAM; a larger/faster
runtime mainly buys you speed. Nothing in the sandbox or CI executes this cell
(both are offline), so **this Mixtral run is the one remaining Colab-only step**:
the packer, the loader, the fail-loud classification check, the bounded resident
weights/streamed bytes, both-compiler builds and the full test suite are all
proven in-sandbox on a Mixtral-faithful generated fixture, but the real 19 GB
download + end-to-end decode can only be re-verified on Colab.

The exact copy-pasteable re-verification sequence is: run Cell 1 (clone), Cell 2
(deps), Cell 3 (build Release), then run the two cells above (`df -h /content`
to confirm ~48 GB free, then the `--real-model --is-moe` command). Expect one of
two honest outcomes:

- **It streams.** You see the `GUARDS` block, the pack step, then decoded text
  with a `stats: resident model weights = ... MiB, peak RSS = ... MiB, ...
  streamed = ... MiB` line - resident weights far below the ~19 GB on-disk size.
- **It fails LOUDLY at pack time** (NOT an OOM at decode). With the packer fix on
  this branch, if Mixtral's expert tensors do not classify (metadata says MoE but
  zero experts were matched), `strata-pack` now aborts with a diagnostic listing
  the expert-tensor names/shapes it found, instead of silently writing a
  0-expert `.strata` that the engine then tries to hold whole-model-resident and
  gets SIGKILLed (rc=-9). So a classification miss is a clear, actionable error
  at pack time, not a mysterious decode-time OOM.

### Cell 9 - benchmark harness (real numbers, free-tier sized)

```bash
%%bash
cd /content/strataflow
# Runs a MATRIX of generated-F32 MoE configs (two model sizes x three
# --expert-slots settings), measures real numbers per run, prints a results
# table, and writes strataflow_bench.json + strataflow_bench.csv under the
# workdir for download. CPU-only, offline, free-tier sized, finishes in a few
# minutes. Grow it with --layers-list / --experts-list / --slots-list /
# --max-tokens.
python3 colab/strataflow_bench.py --workdir /content
```

The table has one row per `(model size x expert-slots)` config, with columns:

- `on-disk MiB` - size of the packed `.strata` file.
- `slots` - `--expert-slots` for the run (`auto` = 0, fully resident).
- `gen tok/s` - END-TO-END generate throughput (`max-tokens / full sf_generate
  wall-clock`). This INCLUDES prompt processing and the first-token latency, so
  it is not steady-state decode.
- `decode tok/s` - STEADY-STATE decode throughput with the first token and its
  TTFT removed (`(max-tokens - 1) / (wall-clock - TTFT)`). This is the figure
  the harness targets; it is only reported when the CLI gives TTFT and
  `max-tokens > 1`.
- `TTFT ms` - time-to-first-token.
- `resident MiB` / `peak RSS MiB` - RAM held by the run.
- `streamed MiB` / `bytes/tok` - SSD bytes streamed through `StrataReader`.
  `bytes/tok` is exact: it comes from the raw `streamed_bytes` uint64 the CLI
  reports, not the 1-decimal `streamed MiB` display value.

Each row is a SINGLE run (no warmup or repeat), so `gen tok/s`, `decode tok/s`
and `TTFT ms` are single noisy samples; treat them as indicative, not
statistically tight.

What the generated numbers prove and do not prove: the **bounded-RAM**
property is real - with `--expert-slots` set below the expert count,
StrataFlow's resident model weights stay far below the on-disk size and the
SSD **streamed bytes per token move with the slot count** (the Phase 3 "big
model, small RAM" exit criterion). Whole-process **peak RSS** is reported too,
but it does NOT move with `--expert-slots`: it carries a fixed llama/ggml
backend + vocab floor plus a one-layer staging buffer, so it stays roughly flat
across the slots sweep and can exceed on-disk for these tiny models. The
exit-criteria block flags that explicitly and does not present a peak-RSS
percentage as the proof. The absolute
**decode tok/s** is only a MECHANISM proxy: the generated model has random
weights and tiny dimensions, so its throughput is NOT comparable to the PLAN.md
section 2 ladder, which is for large real models on NVMe. The harness says so
and does not fabricate a comparison. For real throughput, run Cell 10 (or a
Mixtral quant on a larger runtime). The JSON/CSV artifacts are written for
download so you can keep the measured numbers.

### Cell 10 (optional) - include the real downloaded TinyLlama row

```bash
%%bash
cd /content/strataflow
# Appends ONE real downloaded TinyLlama (dense llama Q4_K_M) row to the matrix.
# This row needs network + disk (Colab), and is cleanly SKIPPED (not a crash)
# when offline; the generated rows still produce a full table either way.
python3 colab/strataflow_bench.py --workdir /content --real-model
```

The real-model row reuses the same download + disk guard + pack + bounded-run
logic as Cell 7, so its tok/s and TTFT are measured on real weights with the
real tokenizer. In the offline sandbox and in CI the row is skipped with a
clear message; only Cell 10 on a networked Colab produces that row.

## What this does and does not prove

- **Proves:** StrataFlow builds and runs on a real CPU-only Linux box, decodes
  through its own ggml forward pass (no `llama_decode`, no GPU), and streams
  experts from a `.strata` file so the on-disk model is far larger than
  StrataFlow's resident model weights, with the streamed bytes per token bounded
  by `--expert-slots` (whole-process peak RSS additionally carries a fixed
  backend + vocab floor plus one-layer staging; see "Big model, small RAM").
  With `--real-model` it does this on a **real downloaded, quantized** GGUF with
  its real tokenizer producing real output text, and reports measured
  tokens/sec and peak RSS. Quantized weights run through the engine's
  type-agnostic staging path (Q8_0 validated against the oracle, K-quant
  Q4_K/Q6_K validated by the generated-fixture oracle gate).
- **Honest caveat - architecture coverage.** The engine implements **llama-arch
  MoE and dense llama only**. The default real-model run uses a **dense** llama
  (TinyLlama) because it is small to download and exercises the same download ->
  pack -> stream -> bounded-RAM mechanism end to end. Running a real llama-arch
  **MoE** (Mixtral, Cell 8) is also possible on the free tier - it is **disk-
  bound, not RAM-bound** (its ~24 GB is a disk/download cost; `--expert-slots`
  keeps resident RAM bounded) - just slow (long download/pack + SSD streaming),
  so a larger runtime mainly buys speed. Qwen2-MoE / DeepSeek-MoE are different
  architectures and are not yet implemented. See
  [`../docs/ROADMAP.md`](../docs/ROADMAP.md).
- **Benchmark harness (Cells 9-10).** A benchmark harness now exists
  (`colab/strataflow_bench.py`) and emits measured numbers - tok/s, TTFT, peak
  RSS, resident weights, on-disk size, SSD streamed MiB and bytes/token - as a
  table plus JSON/CSV across a config matrix. The generated-F32 path (offline,
  sandbox/CI-verified) proves the MECHANISM and the bounded-RAM property; the
  absolute tok/s ladder in PLAN.md section 2 is for large real models on NVMe,
  so the generated toy's throughput is a proxy only. Measure real throughput
  with the optional real-model row (Cell 10) or a Mixtral quant on a larger
  runtime.
- **Not yet:** published tok/s benchmarks against the section 2 ladder on large
  real models (the harness measures them; the large-model numbers themselves
  are Colab/HW-demonstrated, not proven in the offline sandbox/CI), and the
  non-llama MoE architectures above.

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

Run it TWICE for a before/after on the SAME model. Run A relies on StrataFlow's
AUTO expert-residency (`--expert-slots 0`): the engine sizes the resident expert
pool per model from this runtime's free RAM plus Mixtral's own expert layout, so
on a runtime with enough RAM it caches the whole layer working set and re-reads
far fewer SSD bytes per token. Run B forces a deliberately-small fixed pool
(`--expert-slots 4`) so you can SEE the difference: the fixed-too-small pool
evicts and re-streams experts every token, so its `streamed = ... MiB` and
bytes/token are much larger (and tok/s lower) than the auto run on the same
model. Set `--is-moe` so the summary labels it correctly, and raise
`--expected-gb` to match the chosen quant.

```bash
%%bash
cd /content/strataflow
# RUN A - AUTO residency (recommended). --expert-slots 0 lets StrataFlow size
# the resident expert pool per model from measured free RAM + Mixtral's expert
# layout. On a runtime with enough RAM this caches the whole working set, so it
# re-reads the fewest SSD bytes per token. --cache-gb X would cap that budget;
# precedence is explicit --expert-slots > --cache-gb > auto-from-free-RAM.
python3 colab/strataflow_colab.py --real-model --is-moe \
  --hf-repo TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
  --hf-file mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf \
  --expected-gb 19 --expert-slots 0 --max-tokens 16 \
  --prompt "The city of"
```

```bash
%%bash
cd /content/strataflow
# RUN B - deliberately-small fixed pool for comparison. --expert-slots 4 forces
# a too-small resident pool on the SAME model, so experts are evicted and
# re-streamed every token. Compare its `streamed = ... MiB` / bytes-per-token
# and tok/s against RUN A: auto streams far fewer bytes and is faster because it
# caches the working set when RAM allows. (The .strata is already packed from
# RUN A, so this run skips the download/pack and only re-decodes.)
python3 colab/strataflow_colab.py --real-model --is-moe \
  --hf-repo TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
  --hf-file mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf \
  --expected-gb 19 --expert-slots 4 --max-tokens 16 \
  --prompt "The city of"
```

#### Residency + prefetch gating BEFORE/AFTER (sandbox-measured)

Two levers shipped together: (B) layer-stratified, frequency-aware per-layer
expert residency replacing the single global plain-LRU, and (A) prefetch
free-slot install gating + adaptive self-backoff so async prefetch never
increases bytes streamed. Both are measured IN-SANDBOX on the bounded
multi-layer generated fixtures (byte-identical decoded text throughout), with
the absolute wall-clock tok/s left as a Colab/hardware-dependent companion. The
table below is the generated `bench_moe-L8-E16` fixture (8 layers, 16x4 experts,
full working set = 128 bundles), max-tokens 8, release build.

Lever B (residency), async off, at the shape-derived tight budget slots=32
(= n_layer * n_expert_used), measured via the CLI apples-to-apples (same slot
count, same peak RSS, decoded text byte-identical both ways):

| residency policy        | streamed bytes      | streamed MiB | peak RSS  |
| ----------------------- | ------------------- | ------------ | --------- |
| global plain-LRU (before) | 522,190,848       | 498.0        | 132.4 MiB |
| layer-stratified (after)  | 327,155,712       | 312.0        | 132.4 MiB |

That is 37.3% fewer SSD bytes streamed at the same slot budget and peak RSS,
byte-identical output. (Misses as the engine-level proxy: 975 before vs 626
after, 35.8% fewer.)

Lever A (prefetch gating), async-on vs async-off at the bounded slot count
slots=24, from `colab/strataflow_bench.py --layers-list 8 --experts-list 16
--slots-list 24,0 --async-list on,off --max-tokens 8`:

| config                 | bytes/tok    | streamed MiB | peak RSS  | cbu | decoded text |
| ---------------------- | ------------ | ------------ | --------- | --- | ------------ |
| slots=24 async=off (before regression)* | 63,700,992 | 486.0 | 132.5 MiB | 0 | identical |
| slots=24 async=on (before regression)*  | 94,371,840 | 720.0 | 132.4 MiB | 0 | identical |
| slots=24 async=off (after) | 61,341,696 | 468.0  | 132.4 MiB | 0 | identical |
| slots=24 async=on (after)  | 46,792,704 | 357.0  | 132.4 MiB | 0 | identical |

`*` the "before regression" rows are the FEAT-001 baseline on current `main`,
where async-on streamed ~1.48x MORE than async-off (prefetch HURT). After the
gating, async-on streams 46,792,704 bytes/tok, which is 0.76x async-off
(61,341,696) and LESS than before - async prefetch no longer hurts. Decoded
text is byte-identical across every row and matches the before run exactly (the
policy change did not alter output). `cbu` (completed-before-use) stays 0 on
this fast sandbox CPU, mirroring the 2-core Colab 0-overlap case, which is why
self-backoff engages. The decode tok/s per row is a MECHANISM proxy only (tiny
random-weight fixture, fast sandbox CPU) and is NOT a throughput claim.

#### Async prefetch overlap on vs off (FEAT-003), real Mixtral

Async prefetch overlaps the next layer's predicted-expert disk reads with the
current layer's compute on a background I/O thread, hiding read latency behind
compute. It only has work to do on a **bounded** cache (a fully-resident working
set never streams), so these commands use auto residency (`--expert-slots 0`)
**capped by `--cache-gb`** so the resident pool is smaller than Mixtral's full
working set and experts stream every layer. Run the SAME config twice - once
with `--async-prefetch on` and once `off` - and compare `tokens/sec` and the
`completed-before-use` overlap metric the driver prints. The decoded text is
byte-identical between the two (the authoritative load is unchanged); only
`completed-before-use` (0 when off, > 0 when the overlap fires) and the
wall-clock differ.

```bash
%%bash
cd /content/strataflow
# RUN C - async prefetch ON (default). Auto residency capped by --cache-gb so
# the pool is BOUNDED below Mixtral's full working set and experts stream every
# layer (so overlap has something to hide). Tune --cache-gb down if this runtime
# has enough RAM to cache the whole set (otherwise there is nothing to prefetch).
python3 colab/strataflow_colab.py --real-model --is-moe \
  --hf-repo TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
  --hf-file mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf \
  --expected-gb 19 --expert-slots 0 --cache-gb 6 --max-tokens 32 \
  --async-prefetch on \
  --prompt "The city of"
```

```bash
%%bash
cd /content/strataflow
# RUN D - async prefetch OFF (synchronous warm path), SAME --cache-gb so it is a
# true before/after. The .strata is already packed from RUN C, so this run skips
# the download/pack and only re-decodes. Compare against RUN C:
#   * decoded text: IDENTICAL (correctness invariant);
#   * completed-before-use: 0 here (off) vs > 0 in RUN C (overlap firing);
#   * tokens/sec: RUN C should be >= RUN D when the overlap hides real read
#     latency (hardware/runtime-dependent - the mechanism is the overlap metric,
#     the absolute speed depends on this runtime's disk vs compute balance).
python3 colab/strataflow_colab.py --real-model --is-moe \
  --hf-repo TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
  --hf-file mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf \
  --expected-gb 19 --expert-slots 0 --cache-gb 6 --max-tokens 32 \
  --async-prefetch off \
  --prompt "The city of"
```

To measure the effect of the residency + prefetch-gating change itself, run the
SAME commands (RUN A/RUN B and RUN C/RUN D) on **current `main` (before)** and
on **this branch (after)** and compare the `streamed = ... MiB` / bytes-per-token
and tok/s each prints. The sandbox-proven expectation is that the after branch
streams `<=` bytes/token with async on at a bounded cache (prefetch no longer
hurts) and the per-layer-stratified residency re-reads fewer SSD bytes/token
than the before. Absolute tok/s stays hardware-dependent on the 2-core Colab
runtime, so do not expect a fixed tok/s number - compare the streamed MiB and
bytes/token first, and read tok/s as the hardware-dependent companion.

This async on-vs-off contrast on the real ~19 GB Mixtral is **Colab/hardware-
dependent and the offline sandbox cannot run it** (no network to download
Mixtral, and absolute tok/s depends on the runtime's disk and CPU). The sandbox
proves the MECHANISM in-sandbox instead: `colab/strataflow_bench.py
--layers-list 8 --experts-list 16 --slots-list 24,0 --async-list on,off` decodes
a byte-identical sequence on vs off while streamed bytes/token with async on
stays `<=` async off at the bounded slot count (the prefetch-no-longer-hurts
invariant), and `completed-before-use` is reported (it stays ~0 on the fast
sandbox CPU, which is why self-backoff engages). The CLI's own stats line also
reports it directly: `build/release/bin/strataflow --model
bench_moe-L8-E16.strata --expert-slots 24 --max-tokens 16 --async-prefetch on`
vs `off`.

Note on `streamed_bytes` in this tiny-fixture contrast: with the FEAT-003
free-slot install gating + self-backoff, async-on no longer increases streamed
bytes vs async-off at a bounded slot count - it is now `<=` off. A completed
speculative read is only installed into a genuinely FREE per-layer slot
(wired through the FEAT-002 layer-stratified pool's `has_free_slot_in_layer`),
so a prefetch can never evict a still-needed authoritative resident and can
never amplify the authoritative working set's miss/stream count; and when the
observed `completed-before-use` rate stays ~0 over a shape-derived window the
worker backs off and stops issuing speculative reads entirely. Measured
in-sandbox on this fixture at `--expert-slots 24` over 8 tokens: async-on
streams 46,792,704 bytes/tok vs async-off 61,341,696 bytes/tok (on is 0.76x
off), at the same 132.4 MiB peak RSS, byte-identical decoded text. This
reverses the earlier behavior, where speculative reads racing eviction made
async-on stream MORE than off (~1.5 GiB vs ~1.0 GiB over 16 tokens).

The honest caveat remains: `completed-before-use` stays ~0 on this fast sandbox
CPU (the per-layer compute window is too small to hide the background read),
which mirrors the 2-core Colab 0-overlap case and is exactly why self-backoff
engages. So the sandbox-proven result is that async prefetch no longer HURTS
(streamed bytes `<=` off, output byte-identical); the absolute wall-clock
speedup from overlap is still hardware-dependent and reserved for the Colab
Mixtral run, where a slower disk relative to compute gives the overlap real
read latency to hide.

Sandbox-proven vs Colab-only, to be exact about what each part demonstrates:

- **Sandbox-proven (offline, in CI):** the per-model AUTO residency policy, its
  precedence, oracle byte-identity (the auto slot count decodes a byte-identical
  greedy sequence to the fully-resident and fixed-slot runs), and the
  streamed-bytes delta between auto and a too-small fixed count. The last is
  measured by `colab/strataflow_bench.py --layers-list 8 --experts-list 16
  --slots-list 2,16,0` on a GENERATED MoE, where the auto row streams ~8x fewer
  SSD bytes/token than `slots=2` on the same model (8.06 MB/tok vs 64.39 MB/tok)
  and decodes faster (mechanism proxy on random weights). ALSO sandbox-proven:
  (lever B) the layer-stratified, frequency-aware per-layer residency re-reads
  fewer SSD bytes/token than the old global plain-LRU at the SAME bounded slot
  count (L8-E16 at slots=32: 327,155,712 vs 522,190,848 streamed bytes, 37.3%
  fewer, byte-identical decode at equal peak RSS), and (lever A) async-on no
  longer increases streamed bytes vs async-off at a bounded slot count (L8-E16
  at slots=24: 46,792,704 vs 61,341,696 bytes/tok, on is 0.76x off, decoded text
  byte-identical, same 132.4 MiB peak RSS), reversing the earlier behavior where
  async-on streamed MORE than off.
- **Colab-only (this cell):** the real ~19 GB Mixtral download + end-to-end
  decode and its absolute wall-clock tok/s. The offline sandbox cannot download
  Mixtral, so RUN A vs RUN B (auto vs small fixed residency) and RUN C vs RUN D
  (async on vs off) above are the real-model before/after you reproduce on
  Colab. Absolute tok/s is hardware-dependent on the 2-core free runtime and is
  deliberately NOT promised as a fixed number; compare streamed MiB /
  bytes-per-token first, with tok/s as the hardware-dependent companion.

This TheBloke Q3_K_M GGUF uses the LEGACY per-expert tensor naming
(`blk.N.ffn_{gate,down,up}.E.weight`). As of the legacy per-expert packer fix,
`strata-pack` recognizes those names, GROUPS them into the stacked `.strata`
expert region (verbatim bytes), and synthesizes stacked `_exps` metadata, so the
engine streams and decodes it the same as any stacked MoE - it no longer fails
to classify. The in-sandbox proof is a generated legacy fixture (plus a Q3_K
variant mirroring this exact file) packed, streamed, and decoded byte-identical
to the libllama oracle on a matched stacked model; the real ~19 GB download +
end-to-end decode stays Colab-only.

As with the generated demo, `--expert-slots` bounds how many experts are held
resident at once, so a multi-GB MoE decodes with StrataFlow's resident model
weights and streamed bytes far below the on-disk size - the whole mission on a
real, downloaded, quantized MoE. On the free tier expect it to be slow (long
download + pack, disk-bound streaming), not blocked by RAM; a larger/faster
runtime mainly buys you speed. Nothing in the sandbox or CI executes this cell
(both are offline), so **this Mixtral run is the one remaining Colab-only step**:
the packer, the loader, the fail-loud classification check, the bounded resident
weights/streamed bytes, both-compiler builds and the full test suite are all
proven in-sandbox on a Mixtral-faithful generated fixture AND on a generated
legacy per-expert fixture (incl. a Q3_K variant mirroring this exact file,
packed and decoded byte-identical to the libllama oracle on a matched stacked
model), but the real 19 GB download + end-to-end decode can only be re-verified
on Colab.

The exact copy-pasteable re-verification sequence is: run Cell 1 (clone), Cell 2
(deps), Cell 3 (build Release), then run the cells above (`df -h /content` to
confirm ~48 GB free, then RUN A with `--expert-slots 0` for the auto-residency
decode, and optionally RUN B with `--expert-slots 4` for the before/after
contrast on the same model). Expect one of two honest outcomes:

- **It streams.** You see the `GUARDS` block, the pack step, then decoded text
  with a `stats: resident model weights = ... MiB, peak RSS = ... MiB, ...
  streamed = ... MiB` line - resident weights far below the ~19 GB on-disk size.
- **It fails LOUDLY at pack time** (NOT an OOM at decode). The packer now
  classifies BOTH stacked `_exps` and legacy per-expert
  (`blk.N.ffn_{gate,down,up}.E.weight`) expert tensors, so a correctly-formed
  Mixtral quant streams. If some OTHER expert layout is encountered such that
  the metadata declares a MoE (`llama.expert_count` > 0) but zero experts match
  either form, `strata-pack` aborts with a diagnostic listing the expert-tensor
  names/shapes it found, instead of silently writing a 0-expert `.strata` that
  the engine then tries to hold whole-model-resident and gets SIGKILLed (rc=-9).
  So a classification miss is a clear, actionable error at pack time, not a
  mysterious decode-time OOM.

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
- `slots` - `--expert-slots` for the run. `auto` = 0 = StrataFlow auto-sizes the
  resident expert pool per model from free RAM + the model's expert layout
  (holds the whole working set when RAM allows, else bounds it); a positive value
  forces that many resident expert bundles.
- `gen tok/s` - END-TO-END generate throughput (`max-tokens / full sf_generate
  wall-clock`). This INCLUDES prompt processing and the first-token latency, so
  it is not steady-state decode.
- `decode tok/s` - STEADY-STATE decode throughput with the first token and its
  TTFT removed (`(max-tokens - 1) / (wall-clock - TTFT)`). This is the figure
  the harness targets; it is only reported when the CLI gives TTFT and
  `max-tokens > 1`.
- `async` - `--async-prefetch` for the run (`on`/`off`, FEAT-003). Sweep both
  with `--async-list on,off` to contrast the background I/O overlap on vs off at
  the SAME bounded slot count.
- `TTFT ms` - time-to-first-token.
- `resident MiB` / `peak RSS MiB` - RAM held by the run.
- `streamed MiB` / `bytes/tok` - SSD bytes streamed through `StrataReader`.
  `bytes/tok` is exact: it comes from the raw `streamed_bytes` uint64 the CLI
  reports, not the 1-decimal `streamed MiB` display value. Note: with the
  FEAT-003 free-slot install gating + self-backoff, `--async-prefetch on` at a
  tight slot count now streams `<=` as many bytes/tok as `off` (a completed
  prefetch is only installed into a genuinely free per-layer slot so it never
  evicts a still-needed resident, and the worker backs off issuing speculative
  reads once the observed overlap stays ~0). The authoritative experts loaded,
  and thus the decoded text, are identical either way.
- `cbu` - `completed-before-use` (FEAT-003 overlap signal): of the prefetched
  experts the routing then used, how many the background I/O worker had fully
  read AND installed BEFORE the layer's authoritative acquire needed them, i.e.
  the overlap actually hid the disk read. `0` by construction when `async=off`;
  `> 0` on `async=on` is the mechanism firing.

When the sweep includes `--async-list on,off`, the exit-criteria block adds an
ASYNC PREFETCH on-vs-off contrast per `(model, slots)`: it confirms the decoded
text is byte-identical on vs off and reports `completed-before-use`, streamed
bytes/token, decode tok/s and TTFT side by side. On a small sandbox CPU the
per-layer compute window is tiny, so wall-clock gains are hardware-dependent and
the overlap metric (`cbu > 0`) is the honest proof of the mechanism.

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
percentage as the proof. When the sweep includes both `auto` (0) and a smaller
fixed slot count for a model (e.g. `--slots-list 2,16,0`), the exit-criteria
report also contrasts the AUTO row against the smallest fixed-slot row for that
model, showing that auto streams far fewer SSD bytes per token (and typically
decodes faster) because it caches the whole layer working set when RAM allows -
the per-model auto-residency policy at work, stated honestly as a mechanism
proxy. The absolute
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

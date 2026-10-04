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
  - Free Colab is ~12 GB RAM, ~70-100 GB ephemeral disk. Keep the model
    (generated or downloaded) well under the disk size, and use
    `--expert-slots` to keep resident weights far below the on-disk size.

### Which real model, and why

The engine runs **llama-architecture** models only. That drives the choice:

- The natural real llama-arch MoE is **Mixtral** (mistralai, Apache-2.0, genuine
  llama arch). But the smallest useful Mixtral-8x7B GGUF is ~24 GB+ even at a low
  K-quant, which strains the Colab free tier. You can still point the flow at a
  Mixtral quant on a larger runtime via `--hf-repo/--hf-file --is-moe`.
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

As `--experts` grows, the `.strata` file (on disk) grows a lot while peak RAM
stays close to the trunk size - that is StrataFlow doing its job.

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

```bash
%%bash
cd /content/strataflow
# Mixtral is a genuine llama-arch MoE (Apache-2.0). The smallest useful GGUF is
# ~24 GB+, so this needs a LARGE runtime (not the free tier). The flow's disk
# guard will refuse if it will not fit. Set --is-moe so the summary labels it
# correctly, and raise --expected-gb to match the chosen quant.
python3 colab/strataflow_colab.py --real-model --is-moe \
  --hf-repo TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
  --hf-file mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf \
  --expected-gb 20.4 --expert-slots 2 \
  --prompt "Explain mixture of experts in one sentence:" --max-tokens 24
```

As with the generated demo, `--expert-slots` bounds how many experts are held
resident at once, so a multi-GB MoE decodes with peak RAM far below the on-disk
size - that is the whole mission on a real, downloaded, quantized MoE.

## What this does and does not prove

- **Proves:** StrataFlow builds and runs on a real CPU-only Linux box, decodes
  through its own ggml forward pass (no `llama_decode`, no GPU), and streams
  experts from a `.strata` file so the on-disk model is larger than peak RAM.
  With `--real-model` it does this on a **real downloaded, quantized** GGUF with
  its real tokenizer producing real output text, and reports measured
  tokens/sec and peak RSS. Quantized weights run through the engine's
  type-agnostic staging path (Q8_0 validated against the oracle, K-quant
  Q4_K/Q6_K validated by the generated-fixture oracle gate).
- **Honest caveat - architecture coverage.** The engine implements **llama-arch
  MoE and dense llama only**. The default real-model run uses a **dense** llama
  (TinyLlama) because the smallest llama-arch MoE (Mixtral) is too large for the
  free tier; the dense path exercises the same download -> pack -> stream ->
  bounded-RAM mechanism. Running a real llama-arch **MoE** (Mixtral) needs a
  larger runtime (Cell 8). Qwen2-MoE / DeepSeek-MoE are different architectures
  and are not yet implemented. See [`../docs/ROADMAP.md`](../docs/ROADMAP.md).
- **Not yet:** published tok/s benchmarks against the exit criteria, and the
  non-llama MoE architectures above.

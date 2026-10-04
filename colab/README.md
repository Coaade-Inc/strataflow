# Running StrataFlow on Google Colab

A step-by-step guide to build StrataFlow on a free Colab CPU runtime and run a
larger-than-toy Mixture-of-Experts model through its streaming engine - proving
the "big model, bounded RAM, no GPU" path on a real machine.

## Before you start

- Use a **CPU runtime** (Runtime -> Change runtime type -> CPU). StrataFlow's
  engine is CPU-only today; a GPU runtime is not needed and is not used.
- Honest limits (see [`../docs/ROADMAP.md`](../docs/ROADMAP.md)):
  - The engine runs **F32** llama-arch MoE and dense llama only. A downloaded
    **quantized** real model (Q4/Q6) will not run yet. So this demo generates a
    larger F32 MoE (hundreds of MB) - that is the real exercise of the streaming
    path on real hardware, not a toy.
  - First build compiles llama.cpp/ggml: a few minutes.
  - Free Colab is ~12 GB RAM, ~70-100 GB ephemeral disk. Keep the generated
    model well under the disk size (the defaults make a ~0.5-1 GB model).

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
pip -q install gguf numpy
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
- `Maximum resident set size` from `/usr/bin/time -v` - peak RAM. Compare it to
  the `.strata` file size on disk (`ls -lh /content/colab_moe.strata`): the
  model on disk is larger than the peak RAM, which is the bounded-RAM point.

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

## What this does and does not prove

- **Proves:** StrataFlow builds and runs on a real CPU-only Linux box, decodes a
  real-size MoE through its own ggml forward pass, and streams experts from a
  `.strata` file so the on-disk model is larger than peak RAM.
- **Does not prove yet:** running a real *pretrained, quantized* model, a
  user-facing memory-budget flag, or published tok/s benchmarks.

> **Top-priority next work: quantized-weight support.** Real models ship
> quantized (Q4_K, Q6_K, Q8_0, IQ families). The engine is **F32-only** today,
> so a downloaded real model will not run yet - this is the #1 item in
> [`../docs/ROADMAP.md`](../docs/ROADMAP.md). StrataFlow's goal is to run **all**
> GGUF weight types, quantized and full-precision alike; this demo uses a
> generated F32 model only because that is what the engine handles right now.

<div align="center">

# StrataFlow

**Run very large models on the hardware you already own.**

A local inference engine that streams trillion-scale Mixture-of-Experts models
from disk, caches the hot parts in RAM and VRAM, and uses whatever GPU it finds
to go faster - on Windows, Linux and macOS.

*A Coaade Inc. project. Source-available. Free for personal, non-commercial use.*

</div>

---

> **Status: engine core working (CPU).** StrataFlow runs its own forward pass
> and decodes Mixture-of-Experts models byte-identically to the reference, with
> only the active experts resident in a bounded memory budget. See
> [Current status](#current-status) below, the design in
> [`docs/ENGINE_CORE_DESIGN.md`](docs/ENGINE_CORE_DESIGN.md), and the roadmap in
> [`docs/PLAN.md`](docs/PLAN.md).

## The idea

Model weights live in *strata* - layers of memory - and *flow* between them on
demand:

```
   VRAM  (fastest, smallest)   <- hot experts, trunk, KV cache
    ^v
   RAM   (fast, medium)        <- warm experts, resident trunk
    ^v
   SSD   (large, slowest)      <- the full model; cold experts stream in per token
```

For a Mixture-of-Experts model only a small fraction of the weights are used per
token, so the rest only has to be *reachable on disk*, not held in memory. That
is what lets a trillion-parameter model run on a low-RAM PC: **more memory only
buys speed, not capability - the output is identical at every memory size.**

## What StrataFlow adds over existing engines

- **Its own forward pass over ggml.** StrataFlow builds and runs the compute
  graph itself, so it decides exactly which weights are resident and when. It is
  not a wrapper that hands the model to another engine.
- **One tiered weight store** across VRAM, RAM and SSD with a shared caching
  policy - not three separate mechanisms bolted together.
- **True per-expert streaming.** Each layer's router is read first, then only
  the top-k active experts are made resident before the expert matmul runs - so
  resident expert memory is bounded to what a token actually uses, not the whole
  layer.
- **Automatic hardware tuning**: profiles your machine on first run and decides
  what goes where. No hand-tuning placement flags.
- **A streaming-friendly model format** (`.strata`) built from standard GGUF, so
  one file holds the always-resident trunk plus aligned per-expert blobs.
- **Prediction-driven prefetch** (in progress): guess the next layer's experts
  and load them while the current layer computes, so disk reads hide behind
  compute.

Primary goal: run **Coaade Inc.'s models** fast and locally. The public
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) project (which runs
a 2.78T model in ~8 GB RAM, CPU-only) is used as a stand-in to validate the
trillion-scale streaming path.

## Approach

StrataFlow **reuses [ggml](https://github.com/ggml-org/llama.cpp) as its kernel
and tensor library** (CPU SIMD now; CUDA / HIP / Vulkan / Metal / SYCL to come)
and its GGUF reader, then builds **its own** model loader, compute graph,
scheduler, tiered weight store, and `.strata` format on top. `llama.cpp` is
vendored as a pinned submodule.

Importantly, **`llama.cpp` is not in the forward pass.** StrataFlow constructs
and executes the ggml graph itself. `libllama` is kept only for the tokenizer /
vocabulary and as a correctness *oracle* in tests: every engine change is gated
on reproducing `llama_decode`'s logits bit-for-bit (within floating-point
noise). That is what makes StrataFlow a real engine rather than a skin over
another one.

## Current status

Working today, verified on CPU and green in CI on Linux (GCC + Clang), macOS,
and Windows (MSVC):

- **Own forward pass** for the llama-arch Mixture-of-Experts family, built from
  ggml ops (RMSNorm, RoPE, attention, `mul_mat_id` MoE), matching the oracle.
- **Engine-owned KV cache** and multi-token greedy decode matching the oracle
  token sequence.
- **Bounded per-expert streaming**: with a slot pool smaller than the expert
  count, decode is byte-identical to the fully-resident run while holding only
  the top-k experts per layer (measured: resident expert RAM cut to a fraction,
  with real cache eviction and reload).
- **`.strata` single-file loader**: our loader reads the trunk and streams
  experts directly; decode is byte-identical to the source GGUF.
- **The engine is the default and only inference path.**
- **Predictor-driven prefetch**: the router read-back feeds a predictor that
  warms the next layer's likely experts into the cache ahead of use, raising the
  cache hit rate without changing results.
- **Two architectures**: the llama-arch MoE and dense llama, each gated on
  reproducing the oracle.

In progress / next:

- GPU and multi-backend execution (CPU plus a GPU buffer type via
  `ggml_backend_sched`); the engine already runs on a swappable ggml backend.
- More architectures (shared-expert and grouped-gating MoE families), each a new
  oracle-gated graph builder.
- Asynchronous prefetch overlap (background I/O loading while compute runs) and
  per-layer graph-build reuse across tokens.

See [`docs/ENGINE_CORE_DESIGN.md`](docs/ENGINE_CORE_DESIGN.md) for the engine
design and the EC-1..EC-7 breakdown, and [`docs/PLAN.md`](docs/PLAN.md) for the
overall roadmap.

## Building

Requires CMake >= 3.20, a C++20 compiler (GCC 11+, Clang 14+, or MSVC 2022),
Ninja, and the vendored submodule.

```sh
git submodule update --init --recursive
cmake --preset debug
cmake --build build/debug
ctest --preset debug
```

The `strataflow` CLI loads a `.gguf` or `.strata` model; `strata-pack` converts
a GGUF into the streaming-friendly `.strata` layout. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the build presets and conventions.

## License & commercial use

StrataFlow is **source-available**, under the **Coaade Source-Available License,
Version 1.0** (see [`LICENSE`](LICENSE)). In short:

- Free for **personal, non-commercial use** - read it, learn from it, run it,
  modify it for yourself.
- **Not** for commercial use, and **not** for reselling, rebranding,
  white-labeling, hosting as a service, or building a competing product on it.
- This is **not** an OSI open-source license, and it **does not** convert to
  Apache/MIT over time.

Companies, teams, and organizations - including for internal use - need a
separate commercial license. **Want to use StrataFlow commercially, or discuss
terms? Contact [contact@coaade.com](mailto:contact@coaade.com).**

StrataFlow is built on [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
(MIT), which remains under its own license - see [`NOTICE`](NOTICE). This repo
contains **no model weights**.

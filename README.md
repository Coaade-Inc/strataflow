<div align="center">

# StrataFlow

**Run very large models on the hardware you already own.**

A local inference engine that streams trillion-scale Mixture-of-Experts models
from disk - quantized or full-precision - caches the hot parts in RAM and VRAM,
and uses whatever GPU it finds to go faster, on Windows, Linux and macOS.
(Quantized weights now run and are validated against the reference on Q8_0 and
the K-quant families Q4_K/Q6_K; running an actual downloaded pretrained model is
the next proof point - see [`docs/ROADMAP.md`](docs/ROADMAP.md).)

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
  what goes where. Expert residency is now **auto-sized per model** - StrataFlow
  picks how many expert bundles to keep resident from your measured free RAM plus
  the model's own expert layout (`n_expert`, `n_expert_used`, `expert_bytes`,
  layer count), so it adapts to any MoE shape with no hand-tuned slot flag. When
  RAM holds the whole working set it caches it; when it cannot, residency is
  bounded (never below one layer's top-k). `--cache-gb` caps the expert-RAM
  budget and `--expert-slots` overrides the choice (precedence: `--expert-slots`
  > `--cache-gb` > auto-from-free-RAM).
- **A streaming-friendly model format** (`.strata`) built from standard GGUF, so
  one file holds the always-resident trunk plus aligned per-expert blobs. The
  packer also handles **real Mixtral GGUFs that use the legacy per-expert tensor
  layout** (`blk.N.ffn_{gate,down,up}.E.weight`): `strata-pack` groups them into
  the stacked `.strata` expert region (verbatim bytes, no requant) and
  synthesizes the stacked metadata, so a real downloaded Mixtral packs and runs.
- **Prediction-driven prefetch with async overlap.** The router read-back feeds
  a predictor, and a background I/O worker loads the next layer's predicted
  experts while the current layer computes, so disk reads hide behind compute.
  The synchronous load stays the authoritative backstop, so decode is
  byte-identical to the oracle even under eviction; ggml compute stays
  single-threaded (only an I/O thread is added), the path is thread-safe /
  TSan-clean, and the overlap is default-ON (`--async-prefetch on|off`) with a
  `completed-before-use` overlap metric surfaced on the stats line.

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
- **`.strata` single-file loader with bounded RAM**: our loader holds only the
  always-resident trunk in memory and streams experts from disk on demand - the
  full expert footprint is never allocated. On the tiny MoE this is proven by a
  test: 103 KB trunk resident, 393 KB of experts NOT resident. For a real
  trillion-scale model this is the difference between needing all the RAM and
  needing only the trunk plus a small expert cache. Decode is byte-identical to
  the source GGUF. This is the core "run big models on everyday hardware without
  a GPU" path.
- **The engine is the default and only inference path.**
- **Predictor-driven prefetch**: the router read-back feeds a predictor that
  warms the next layer's likely experts into the cache ahead of use, raising the
  cache hit rate without changing results.
- **Two architectures**: the llama-arch MoE and dense llama, each gated on
  reproducing the oracle.

Also shipped since the engine core (see [`CHANGELOG.md`](CHANGELOG.md) and
[`docs/ROADMAP.md`](docs/ROADMAP.md)):

- **Quantized-weight support** - the engine stages each expert tensor at its
  source ggml type, so quantized experts stream and compute natively. Q8_0 is
  validated against the libllama oracle; the **K-quant families (Q4_K/Q6_K)**
  are validated too (byte-identical greedy sequence) (#26, #30).
- **Memory-budget CLI + peak-RSS reporting** - `--expert-slots` plus a `stats:`
  line that reports resident model-weight bytes and peak RSS, so a user can run
  a model bounded and see the result (#25).
- **RAM-aware, per-model auto expert-residency** - by default the engine
  auto-sizes the resident expert pool from the measured free RAM and the model's
  expert layout (no model constants, so it adapts to any MoE shape); `--cache-gb`
  caps the expert-RAM budget and `--expert-slots` overrides, with precedence
  `--expert-slots` > `--cache-gb` > auto-from-free-RAM. The placement-plan line
  now reports the slot count the engine actually uses with the correct
  stream/resident state and a realistic peak, and the auto choice stays
  byte-identical to the fully-resident and fixed-slot runs against the oracle.
  Sandbox-proven: the bench harness shows the auto row streaming ~8x fewer
  SSD bytes/token than a deliberately-too-small fixed slot count on the same
  model (the real large-MoE speedup is Colab-demonstrated).
- **Bounded peak process RAM** - a single reused expert-staging buffer caps
  staging to one layer's footprint, so peak RSS no longer tracks model size: a
  785 MiB model runs in 132 MiB peak RSS, and a 1177 MiB model also runs in
  132 MiB (peak RSS flat as on-disk size grows) (#27).
- **Legacy per-expert Mixtral packing** - real TheBloke Mixtral GGUFs name
  routed experts per expert (`blk.N.ffn_{gate,down,up}.E.weight`) rather than
  stacked `_exps`; `strata-pack` now recognizes the legacy names, groups them
  per `(layer, expert)` into the stacked `.strata` layout (verbatim bytes, no
  requant) and synthesizes stacked metadata, so a real Mixtral packs and runs.
  The engine, reader, format and `mul_mat_id` addressing are unchanged (they
  still see only stacked names), and the legacy-sourced `.strata` decodes
  byte-identically to the oracle on generated legacy fixtures (incl. a Q3_K
  variant mirroring real Mixtral Q3_K_M) (#34).
- **Async prefetch overlap** - a background I/O worker loads the next layer's
  predicted experts while the current layer computes, hiding disk-read latency
  on the bounded streaming path. The synchronous load stays the authoritative
  backstop (a mispredicted or in-flight prefetch is a plain miss corrected
  inline), so decode is byte-identical to the oracle even under eviction; ggml
  compute stays single-threaded (only an I/O thread is added), the worker never
  touches the non-thread-safe slot pool (thread-safe / TSan-clean), the overlap
  is default-ON (`--async-prefetch on|off`), and a `completed-before-use`
  overlap metric is surfaced via the stats line. In-sandbox the mechanism is
  proven (`completed-before-use` goes from 0 with async off to > 0 with async
  on, byte-identical output); absolute wall-clock speedup is hardware-dependent
  and re-verified on Colab / real Mixtral (#36).
- **Benchmark harness with real numbers** - reports measured tok/s, TTFT, peak
  RSS, resident weights, on-disk size and bytes-per-token across a config
  matrix, with JSON/CSV artifacts (#31).
- **Real downloaded models run end to end** via the Colab flow: download a GGUF,
  pack to `.strata`, decode bounded. A real quantized **dense** model
  (TinyLlama-1.1B-Chat Q4_K_M) and a real large quantized **MoE**
  (Mixtral-8x7B-Instruct Q3_K_M, ~19 GB on disk) have both run end to end
  through StrataFlow on a free Colab CPU box (12 GB RAM, no GPU) with coherent
  output - the "big model, small RAM, no GPU" proof on a real large MoE. This is
  demonstrated on Colab / real hardware (the offline sandbox and CI cannot
  download multi-GB models); free-tier throughput was initially disk-bound,
  which is exactly what the auto-residency and async-prefetch work targets, so
  the large-MoE result is framed as "runs end to end, bounded RAM, coherent
  output; throughput improvements from auto residency + async prefetch are being
  measured on real hardware" rather than a settled tok/s number.

In progress / next (see [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full,
honest list of what is done and what is open):

- **Capture settled real large-MoE throughput** - the real Mixtral MoE runs end
  to end with bounded RAM and coherent output, but the throughput gains from
  auto residency + async prefetch are still being measured on real hardware
  (Colab / real-Mixtral), not pinned to a product tok/s number in the offline
  sandbox/CI.
- **Broaden quant coverage** - the IQ quant families are not yet validated
  (Q8_0 and the K-quants Q4_K/Q6_K are).
- **More architectures** - Qwen2-MoE and DeepSeek-MoE are different, unimplemented
  architectures; only llama-arch MoE and dense llama are supported today.
- **GPU / multi-backend execution** - optional accelerator, structurally
  reachable but needs real GPU hardware to validate (not required for the no-GPU
  mission).
- **Release engineering** - no release tag and no OpenAI-compatible HTTP server
  front-end yet; only the CLI is wired to the engine. Speculative decoding and
  the low-bit CPU kernels (Phases 5-6) are not started.

> **Honest status:** the engine works on CPU with the bounded-RAM mechanism
> proven, quantized weights (Q8_0 + K-quant Q4_K/Q6_K) validated, legacy and
> stacked Mixtral expert layouts both packing and streaming, async prefetch
> overlap shipped (default-ON), a memory-budget CLI with per-model auto
> residency, a benchmark harness shipping real numbers, and both a real dense
> model and a real ~19 GB Mixtral MoE running end to end on a free Colab CPU box
> (no GPU, coherent output). Still open: pinning settled real large-MoE
> throughput (Colab/HW-demonstrated, not sandbox-proven), IQ-quant validation,
> non-llama MoE architectures, GPU backends (need hardware), and a release tag /
> HTTP server. See [`docs/ROADMAP.md`](docs/ROADMAP.md) and
> [`CHANGELOG.md`](CHANGELOG.md).

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

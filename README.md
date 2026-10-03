<div align="center">

# StrataFlow

**Run very large models on the hardware you already own.**

A local inference engine that streams trillion-scale Mixture-of-Experts models
from disk, caches the hot parts in RAM and VRAM, and uses whatever GPU it finds
to go faster — on Windows, Linux and macOS.

*A Coaade Inc. project · Source-available · Free for personal, non-commercial use*

</div>

---

> **Status: planning / design.** No engine code yet. This repo currently holds
> the research and the architecture plan. See [`docs/PLAN.md`](docs/PLAN.md) and
> [`docs/RESEARCH.md`](docs/RESEARCH.md).

## The idea

Model weights live in *strata* — layers of memory — and *flow* between them on
demand:

```
   VRAM  (fastest, smallest)   ← hot experts, trunk, KV cache
    ↑↓
   RAM   (fast, medium)        ← warm experts, resident trunk
    ↑↓
   SSD   (large, slowest)      ← the full model; cold experts stream in per token
```

For a Mixture-of-Experts model only a small fraction of the weights are used per
token, so the rest only has to be *reachable on disk*, not held in memory. That
is what lets a trillion-parameter model run on a low-RAM PC: **more memory only
buys speed, not capability — the output is identical at every memory size.**

## What StrataFlow adds over existing engines

- **One tiered weight store** across VRAM, RAM and SSD with a shared caching
  policy — not three separate mechanisms bolted together.
- **Prediction-driven prefetch**: guess the next layer's experts and load them
  while the current layer computes, so disk reads hide behind compute.
- **Automatic hardware tuning**: profiles your machine on first run and decides
  what goes where. No hand-tuning placement flags.
- **Speculative decoding** that also tells the loader which experts to fetch —
  the single biggest win on disk-bound machines.
- **A streaming-friendly model format** (`.strata`) built from standard GGUF.

Primary goal: run **Coaade Inc.'s models** fast and locally. The public
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) project (which runs
a 2.78T model in ~8 GB RAM, CPU-only) is used as a stand-in to validate the
trillion-scale streaming path.

## Approach

StrataFlow is a **fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)**: we
reuse its kernels and GPU backends (CPU SIMD, CUDA, HIP, Vulkan, Metal, SYCL) and
the GGUF/model code, and add the tiered runtime, scheduler, predictor and
`.strata` format on top.

## Roadmap

See [`docs/PLAN.md §6`](docs/PLAN.md). In short: baseline benchmarks → fork +
minimal engine → auto placement → **tiered weight store (the core)** → predictor
+ prefetch → speculative decoding → low-bit quant & kernels.

## License & commercial use

StrataFlow is **source-available**, under the **Coaade Source-Available License,
Version 1.0** (see [`LICENSE`](LICENSE)). In short:

- ✅ **Free for personal, non-commercial use** — read it, learn from it, run it,
  modify it for yourself.
- ❌ **Not** for commercial use, and **not** for reselling, rebranding,
  white-labeling, hosting as a service, or building a competing product on it.
- ❌ This is **not** an OSI open-source license, and it **does not** convert to
  Apache/MIT over time.

Companies, teams, and organizations — including for internal use — need a
separate commercial license. **Want to use StrataFlow commercially, or discuss
terms? Contact [contact@coaade.com](mailto:contact@coaade.com).**

StrataFlow is built on [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)
(MIT), which remains under its own license — see [`NOTICE`](NOTICE). This repo
contains **no model weights**.

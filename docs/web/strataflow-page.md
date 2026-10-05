<!--
Website content for the StrataFlow page on www.coaade.com.
This is landing-page copy in Markdown for the web team to render on the site.
It is not a repo README. Keep claims consistent with docs/ROADMAP.md,
CHANGELOG.md, and README.md. ASCII only, regular hyphens, no em dashes.
-->

# StrataFlow

## Run very large models on the hardware you already own - no GPU required.

StrataFlow runs very large Mixture-of-Experts models - quantized or
full-precision - on everyday hardware **without a GPU**, inside a **bounded RAM
budget**, by streaming experts from disk. The model lives on your SSD; only the
parts a token actually uses are held in memory.

---

## What it is

StrataFlow is a local inference engine for large Mixture-of-Experts (MoE)
models. A MoE model is huge on disk but only uses a small slice of its weights
for any single token. StrataFlow takes advantage of that: it keeps the always-on
trunk plus a bounded number of active experts resident, and streams the rest in
from disk per token. The result is that a model far bigger than your RAM runs on
a normal laptop or desktop CPU.

It runs its own forward pass. StrataFlow is not a wrapper that hands your model
to another engine - it builds and executes the compute graph itself, so it
decides exactly which weights are in memory and when.

## How can a model bigger than my RAM run?

This is the question everyone asks: "the model is 24 GB, how can it run on 12 GB
(or 8 GB) of RAM?"

The answer: **RAM holds only the working set, not the whole model.**

Model weights live in *strata* - layers of memory - and *flow* between them on
demand:

```
   VRAM  (fastest, smallest)   <- hot experts, trunk, KV cache (optional GPU)
    ^v
   RAM   (fast, medium)        <- warm experts, resident trunk
    ^v
   SSD   (large, slowest)      <- the full model; cold experts stream in per token
```

For each token, only the trunk and the top-k active experts per layer are
needed. StrataFlow keeps just that working set in RAM and streams the cold
experts from disk. Everything else only has to be **reachable on disk**, not
held in memory. You set the resident budget with a single knob (`--expert-slots`),
and resident RAM stays bounded even as the on-disk model grows.

That means **more memory buys speed, not capability** - the output is identical
at every memory size. A bigger RAM budget just lets more experts stay cached, so
fewer disk reads are needed per token. The answer the model produces does not
change.

On an 8 GB machine the real constraints are therefore (1) enough disk space for
the model file, and (2) streaming speed (your SSD read throughput sets tokens
per second). RAM size does not cap what you can run.

## Why it is different

- **Its own forward pass over ggml.** StrataFlow builds and runs the compute
  graph itself, so it controls residency directly. It is not a skin over another
  engine.
- **One tiered weight store** across VRAM, RAM and SSD with a shared caching
  policy - not three separate mechanisms bolted together.
- **True per-expert streaming.** Each layer's router is read first, then only
  the top-k active experts are made resident before the expert matmul runs - so
  resident expert memory is bounded to what a token actually uses, not the whole
  layer.
- **Automatic hardware tuning.** It profiles your machine on first run and
  decides what goes where. No hand-tuning placement flags.
- **A streaming-friendly model format (`.strata`).** Built from standard GGUF,
  one file holds the always-resident trunk plus aligned per-expert blobs for
  fast per-expert reads.
- **Predictor-driven prefetch.** The router read-back feeds a predictor that
  warms the next layer's likely experts into the cache ahead of use, raising the
  cache hit rate without changing results.

## Proof points (measured)

StrataFlow's correctness and bounded-RAM behavior are measured, not asserted:

- **Byte-identical to the reference engine.** Every engine change is gated on
  reproducing the reference logits bit-for-bit (within floating-point noise), so
  StrataFlow's own forward pass matches a standard decode.
- **Bounded RAM, flat as the model grows.** A 785 MiB model runs at ~132 MiB
  peak RSS, and a 1177 MiB model also runs in ~132 MiB - peak memory stays flat
  as the on-disk model grows.
- **Quantized weights validated.** Q8_0 and the K-quant families Q4_K/Q6_K are
  validated against the oracle (identical greedy token sequence) and stream
  through the `.strata` path.
- **A benchmark harness reports real numbers** - tokens per second, time to
  first token, peak RSS, resident weights and bytes per token - across a matrix
  of model sizes and memory budgets.

The bounded-RAM mechanism is proven, and a real downloaded quantized model runs
end to end through it. Numbers for a real *large* MoE are being gathered on real
hardware; see Current status.

## Who it is for

- Developers and researchers who want to run large MoE models locally, on CPU,
  without renting GPUs.
- People on everyday laptops and desktops who have the disk space for a big
  model but not the RAM to hold it all at once.
- Anyone who wants predictable, bounded memory use with identical output at any
  memory budget.

## Current status

StrataFlow has a working engine core on CPU, green in CI on Linux, macOS and
Windows:

- Its own forward pass for the llama-architecture MoE family and dense llama,
  matching the reference engine byte-for-byte.
- Bounded per-expert streaming from the `.strata` format, with peak RAM proven
  flat as on-disk model size grows.
- Quantized-weight support validated for Q8_0 and K-quant Q4_K/Q6_K.
- A memory-budget knob plus resident/peak-RSS reporting.
- A benchmark harness that emits real tok/s, TTFT, peak RSS and bytes-per-token.
- A real downloaded quantized model running end to end via the Colab guide
  (a dense TinyLlama proven on free Colab).

Honest open items:

- Running a real *large* MoE (as opposed to dense) end to end with captured
  numbers is demonstrated on Colab/real hardware, not yet proven in the offline
  test sandbox.
- The IQ quantization families are not yet validated.
- Other MoE architectures (Qwen2-MoE, DeepSeek-MoE) are not yet implemented;
  only llama-arch MoE and dense llama are supported today.
- GPU backends are structurally reachable but need real GPU hardware to
  validate.
- No tagged release and no OpenAI-compatible HTTP server front-end yet.

The mechanism is proven; real large-model numbers are being gathered.

## Platforms

- Windows, Linux and macOS.
- CPU today (the no-GPU mission). A GPU, if present, is an optional accelerator
  for speed - never a requirement. GPU backends are coming.

## Try it

StrataFlow runs on CPU. The easiest way to see the "big model, bounded RAM, no
GPU" path on a real machine is the Colab guide, which builds the engine, packs a
model to `.strata`, and streams it with a bounded memory budget:

- Colab guide: `colab/README.md` in the repository.

## Licensing and commercial use

StrataFlow is **source-available**, under the **Coaade Source-Available License,
Version 1.0**. In short:

- Free for **personal, non-commercial use** - read it, learn from it, run it,
  and modify it for yourself.
- **Not** for commercial use, and **not** for reselling, rebranding,
  white-labeling, hosting as a service, or building a competing product on it.
- This is **not** an OSI open-source license, and it **does not** convert to
  Apache, MIT, or any other license over time.

Companies, teams, and organizations - including for internal use - need a
separate commercial license.

Want to use StrataFlow commercially, or discuss terms? Contact
[contact@coaade.com](mailto:contact@coaade.com).

The owner and licensor is Coaade Inc., a Delaware C corporation. StrataFlow is
built on llama.cpp / ggml (MIT), which remains under its own license. The
repository contains no model weights.

<!--
Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
-->

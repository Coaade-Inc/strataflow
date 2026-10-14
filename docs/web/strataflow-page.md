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
  decides what goes where, with no hand-tuning placement flags. Expert residency
  is auto-sized per model from your measured free RAM plus the model's own expert
  layout (no model-specific constants, so it adapts to any MoE shape): when RAM
  holds the whole working set it caches it, and when it cannot, residency is
  bounded (never below one layer's top-k). A `--cache-gb` budget caps it and
  `--expert-slots` overrides it, with precedence `--expert-slots` > `--cache-gb`
  > auto-from-free-RAM.
- **One dial for the fit-vs-speed tradeoff.** A single target-RAM setting drives
  everything that decides what stays in memory - the KV window, the resident
  expert cache, and the streamed trunk - from one decision, in three modes.
  **Auto** is the default and the good-neighbor choice: it takes only a
  conservative share of your RAM (leaving a safe headroom so your other apps keep
  running while the model decodes) and streams the rest from disk. **A number**
  (`--target-ram-gb 2`) sets an exact ceiling: StrataFlow fits the model under
  that budget automatically, or tells you clearly, naming the floor, if the
  budget is too small - no silent out-of-memory kill. **Max** (`--ram-mode max`)
  does the opposite: it dedicates the machine to the model, holding the whole
  working set resident for the fastest decode. (Max sizes from all physical RAM,
  so it holds at least as many experts resident and streams no more than auto on
  the same machine - never fewer residents or more streaming.) More budget
  buys speed, not a
  different answer - the output is identical in every mode. If a GPU is present
  the planner budgets placement into VRAM as the fastest tier; GPU-accelerated
  compute itself is planned enterprise work, not a shipped feature, so these
  modes run correctly on CPU and the RAM side is the proven, measured mechanism.
- **A streaming-friendly model format (`.strata`).** Built from standard GGUF,
  one file holds the always-resident trunk plus aligned per-expert blobs for
  fast per-expert reads. The packer handles both the stacked `_exps` expert
  layout and the legacy per-expert layout that real Mixtral GGUFs use
  (`blk.N.ffn_{gate,down,up}.E.weight`), grouping the legacy tensors into the
  stacked `.strata` region with no requant, so a real downloaded Mixtral packs
  and runs.
- **Streamed edge tensors (stream the trunk).** The two biggest trunk tensors -
  the input embedding and the output head - are touched only once per token, at
  the input and output edges. StrataFlow streams them from disk like experts
  instead of holding them resident, while the small, hot per-layer attention and
  norm weights stay in memory. On a large model these vocab-sized tensors are
  several GB, so moving them off the resident floor is the biggest single memory
  win: a model whose resident trunk did not fit before can fit now. It is
  lossless - the same bytes read on demand - and the `.strata` format stays
  back-compatible (older files simply keep the trunk resident).
- **Predictor-driven prefetch with async overlap.** The router read-back feeds a
  predictor, and a background I/O worker loads the next layer's predicted experts
  while the current layer computes, so disk reads hide behind compute. The
  synchronous load stays the authoritative backstop, so results do not change and
  stay byte-identical to the reference engine even under eviction; compute stays
  single-threaded (only an I/O thread is added) and the overlap is on by default.
- **Optional compression of the streamed experts (opt-in, lossless).** The
  packer can compress the on-disk expert blobs so fewer bytes have to be read
  from the SSD per token; StrataFlow decompresses them transparently as it reads,
  so the output is byte-for-byte identical - it is a pure transport optimization,
  not a quality or accuracy change. It is off by default and opt-in at pack time.
  The win depends on the model's weight entropy: on already-quantized experts the
  reduction is modest, so we report it honestly rather than quoting a speedup
  number.

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
  through the `.strata` path. The IQ families are the remaining quant-coverage
  gap.
- **Streaming the trunk is lossless and lowers resident RAM.** With the input
  embedding and output head streamed instead of resident, the resident weight
  bytes drop with no change to the output: the streamed decode is bit-for-bit
  identical to the fully-resident decode and to the reference engine, on both
  full-precision and K-quant models. On the generated test models resident
  weights fall measurably (for example 103,040 to 69,248 bytes on the tiny
  full-precision model); on a real large model the saving is multiple GB.
- **Both Mixtral expert layouts stream.** Real Mixtral GGUFs use a legacy
  per-expert tensor layout as well as the stacked form; the packer handles both,
  so a real downloaded Mixtral packs into `.strata` and streams, decoding
  byte-identically to the reference on the matched model.
- **A benchmark harness reports real numbers** - tokens per second, time to
  first token, peak RSS, resident weights and bytes per token - across a matrix
  of model sizes and memory budgets.
- **Real models run end to end on free hardware.** A real quantized dense model
  (TinyLlama-1.1B-Chat Q4_K_M) and a real large quantized MoE
  (Mixtral-8x7B-Instruct Q3_K_M, about 19 GB on disk) have both run end to end
  through StrataFlow on a free Colab CPU box (12 GB RAM, no GPU) with coherent
  output - the "big model, small RAM, no GPU" proof on a real large MoE.

The bounded-RAM mechanism is proven, and both a real dense model and a real
large MoE run end to end through it on real hardware. The FIT is proven on the
real large MoE: Mixtral-8x7B-Instruct Q3_K_M decoded byte-identically on the free
12 GB Colab CPU box at about 2.4 GiB peak RSS. Throughput on that box is
I/O-bound, not RAM-bound: on a 2-core machine with a slow SSD the limiter is how
many bytes stream from disk per token and how fast the SSD can serve them, so
decode is slow there and gets faster with more CPU cores and a faster NVMe drive.
The large-MoE result is therefore "runs end to end, bounded RAM, coherent output,
fit proven; throughput is I/O-bound on low-core / slow-SSD hardware and is being
measured on real hardware", not a settled tokens-per-second number - we do not
quote a tok/s figure as a product claim. See Current status.

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
- Both the stacked and the legacy per-expert Mixtral expert layouts packing and
  streaming, so a real downloaded Mixtral runs.
- Per-model automatic expert residency sized from free RAM plus the model's own
  expert layout, with a `--cache-gb` cap and an `--expert-slots` override.
- Async prefetch overlap (default-ON): a background I/O worker hides disk reads
  behind compute while output stays byte-identical to the reference.
- A memory-budget knob plus resident/peak-RSS reporting.
- A benchmark harness that emits real tok/s, TTFT, peak RSS and bytes-per-token.
- Real downloaded models running end to end via the Colab guide: a real
  quantized dense model (TinyLlama Q4_K_M) and a real large quantized MoE
  (Mixtral-8x7B-Instruct Q3_K_M, about 19 GB on disk) have both run on a free
  12 GB Colab CPU box, no GPU, with coherent output.

Honest open items:

- Settled throughput for the real large MoE is still being measured on real
  hardware. It runs end to end with bounded RAM and coherent output (fit proven
  at about 2.4 GiB peak RSS on the free Colab box), but throughput there is
  I/O-bound: on a 2-core / slow-SSD machine the bytes streamed per token and the
  SSD read speed set the pace, so decode is slow on that box and improves with
  more CPU cores and a faster NVMe drive. There is no settled tokens-per-second
  product number yet, and we do not quote one.
- The IQ quantization families are not yet validated (Q8_0 and K-quant
  Q4_K/Q6_K are).
- Other MoE architectures (Qwen2-MoE, DeepSeek-MoE) are not yet implemented;
  only llama-arch MoE and dense llama are supported today.
- GPU backends are structurally reachable but need real GPU hardware to
  validate. The VRAM-placement decision an eventual GPU path consumes is already
  built and CPU-unit-tested (GPU discovery, VRAM-first trunk/expert placement,
  and the pure plan-to-`n_gpu_layers` translation), and the engine's compute
  backend is now SELECTABLE: a `STRATAFLOW_GPU=off|cuda|metal|vulkan` build
  option (default `off`) wires the vendored ggml backends, a runtime selector
  picks the first available GPU and always falls back to CPU, resident buffers
  move to the selected backend's buffer type, and a single host-or-device
  `stage_write` primitive turns the per-token expert "fault bytes resident" step
  into a host->device upload on a GPU backend. On the default CPU-only build the
  whole GPU path compiles out and decode is byte-identical to the oracle, proven
  by `test_backend_selection`. What remains, and is hardware-gated, is the actual
  GPU-COMPUTE validation on real hardware. Enterprise GPU acceleration -
  single-GPU first, then multi-GPU scale-out for high-throughput serving - is
  planned future work, not a shipped feature; the no-GPU CPU path remains the
  proven core.
- No tagged release and no OpenAI-compatible HTTP server front-end yet.

The mechanism is proven on real hardware; settled real large-model throughput is
being measured.

## Platforms

- Windows, Linux and macOS.
- CPU today (the no-GPU mission). A GPU, if present, is an optional accelerator
  for speed - never a requirement. The proven path today is CPU / no-GPU.
- Enterprise GPU deployment is planned, not shipped. Single-GPU acceleration
  (routing hot experts, trunk and KV into VRAM as the fastest tier of the same
  streaming engine) and multi-GPU scale-out (expert / tensor / pipeline
  parallelism, topology-aware device placement) are on the roadmap for when GPU
  hardware is present. This is an additional enterprise tier and does not change
  the no-GPU mission; GPUs stay an optional accelerator.

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

# StrataFlow: Architecture and Implementation Plan

**Owner: Coaade Inc.**

Goal: one local inference engine that does two things:
1. Runs **very large MoE models (100B up to 800B+, trillion-scale) on a low-RAM PC with no GPU** at usable speed, streaming experts from disk.
2. **Automatically uses any GPU it finds** (NVIDIA, AMD, Intel, Apple, iGPU) to go faster.

See [RESEARCH.md](./RESEARCH.md) for the background and sources.

### Confirmed decisions (from the project owner)
- **Base:** fork llama.cpp's libraries (ggml + llama) and build the tiered runtime on top. We take the kernels/backends and the GGUF/model code; the scheduler, tiered weight store, predictor and `.strata` format are ours.
- **Platforms:** ship for **Windows, Linux and macOS** from day one. This matches both llama.cpp and the kimi-k3-in-c reference, which already port `O_DIRECT`/`pread`/`posix_memalign` to all three.
- **Primary target: run Coaade's own models.** StrataFlow is the inference runtime for Coaade Inc.'s models — not tied to any one external checkpoint. The Kimi work (K2 at 1T-A32B, the K3 line at 2.78T-A104B per the [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) reference) is used only as a **public stand-in** to validate the trillion-scale streaming path until Coaade models are ready. Design must handle any large MoE whose experts live on disk, so the CPU+SSD streaming path is a **v1 must-have**.
  - *Implication:* Coaade controls the model architecture, so StrataFlow can **co-design the model and the runtime** — e.g. ship Coaade models in the streaming-friendly `.strata` layout natively, train prerouter/draft heads into the checkpoint, and pick expert counts/sizes that cache well. This is a real advantage external engines don't have.
- **Reference to mirror and beat:** [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) (Apache-2.0). It already runs a 2.78T model on 8.24 GB RAM, CPU-only, byte-identically across machine sizes. Strata's job is to reproduce that trunk/expert streaming result **and** add GPU tiers, routing-prediction prefetch, auto hardware tuning, and speculative decoding on top. License: **Apache-2.0** to stay compatible with both llama.cpp and the reference.

---

## 1. Design principles

1. **Measure bytes per token, not FLOPs.** Every feature must cut the bytes read per token, move them to a faster tier, or hide the read behind compute.
2. **Build on ggml/GGUF, don't rewrite it.** Kernels and backends (CPU SIMD, CUDA, Vulkan, Metal, HIP, SYCL) are a solved, large problem. Our value is the **memory-tier runtime and scheduler** above them.
3. **Zero-config by default.** The engine profiles the machine on first run and builds a placement plan. Manual overrides are available but optional.
4. **Exact by default, approximate only when chosen.** Predicted routing and activation sparsity are opt-in quality/speed settings with measured perplexity cost.
5. **Degrade smoothly.** The same model runs at every hardware level, just faster on better hardware. It should never refuse to run.

---

## 2. Target performance (estimates to check in Phase 0)

| Machine | Model | Target decode |
|---|---|---|
| 16 GB DDR4, no GPU, NVMe Gen3 | Qwen3-30B-A3B class, Q4 | 10–15 tok/s |
| 16 GB DDR4, no GPU, NVMe Gen3 | gpt-oss-120b class (about 5B active) | 2–5 tok/s |
| 16 GB DDR4, no GPU, NVMe Gen4 | 397B-A17B class, Q3 | 0.5–1.5 tok/s (more with speculative decoding) |
| 32 GB DDR5 + 8–12 GB GPU | 120B-class MoE | 15–30 tok/s |
| 64 GB DDR5 + 24 GB GPU | DeepSeek/Kimi class (671B–1T) Q2–Q3, mostly from SSD | 3–8 tok/s |
| 8–16 GB, no GPU, NVMe | **800B–2.78T Kimi MoE, experts streamed from disk** | 0.3–1.5 tok/s (I/O-bound; matches the kimi-k3-in-c ladder, we aim to beat it) |
| 16 GB, no GPU | Dense 70B Q4 | **0.1–0.3 tok/s.** Bandwidth limit; see RESEARCH §1 |

The owner's **800B Kimi test** lands in the trillion-scale streamed row: on a no-GPU box the limit is SSD read speed, so the primary levers are (a) the resident expert cache, (b) prediction-driven prefetch, and (c) speculative decoding. On a box with a GPU, the trunk and the hot expert cache move to VRAM and speed climbs substantially.

Success means beating stock llama.cpp with good manual tuning on the same hardware by **at least 1.5×** in the SSD and hybrid settings, and **matching it** when everything fits in memory.

---

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Front-ends: CLI · OpenAI-compatible HTTP server · C API · Python  │
├────────────────────────────────────────────────────────────────────┤
│  Session layer: tokenizer, chat templates, sampling, grammar,      │
│  prompt cache, multi-request batching                              │
├────────────────────────────────────────────────────────────────────┤
│  Decode accelerators: speculative decoding (EAGLE-3 / draft /      │
│  n-gram / MTP heads), tree verification                            │
├────────────────────────────────────────────────────────────────────┤
│  ★ Scheduler: builds a per-token execution plan across devices,    │
│    overlaps compute with transfers, CPU/GPU expert split           │
├──────────────────────────────┬─────────────────────────────────────┤
│  ★ Predictor service         │  ★ Tiered Weight Store (TWS)        │
│  - next-layer router predict │  - VRAM pool ⇄ RAM pool ⇄ SSD file  │
│  - draft-token expert predict│  - unit = expert / neuron-bundle /  │
│  - activation-sparsity masks │    layer block                      │
│  - access statistics         │  - eviction: freq+recency (learned) │
│                              │  - async I/O: io_uring / IOCP /     │
│                              │    F_RDADVISE + O_DIRECT            │
├──────────────────────────────┴─────────────────────────────────────┤
│  ★ KV store: quantized KV (q8 / 4-bit Hadamard), paged,            │
│    optional RAM/SSD spill for very long context                    │
├────────────────────────────────────────────────────────────────────┤
│  ggml compute graph + backends: CPU (AVX2/AVX-512/AMX/NEON + LUT   │
│  kernels), CUDA, HIP, Vulkan, Metal, SYCL                          │
├────────────────────────────────────────────────────────────────────┤
│  ★ Hardware profiler: measures bandwidth of each tier, PCIe speed, │
│    core count/ISA, SSD QD/latency, free memory → hardware profile  │
└────────────────────────────────────────────────────────────────────┘
★ = new work by Strata
```

### 3.1 Hardware profiler
- Runs about 10 seconds on first start and caches the result as `~/.strata/hw.json`.
- Measures: RAM read bandwidth for each thread count, and VRAM size and bandwidth for each device.
- Also measures: host→device copy speed (pinned and pageable), SSD sequential and random-4 MB throughput at queue depth 1/8/32, and free RAM after leaving a safety margin for the OS.
- Detects ISA support (AVX2, AVX-512, AMX, NEON, SVE) and P-cores vs E-cores.

### 3.2 Placement planner
Inputs: model graph metadata, hardware profile, context length, and a quality preset.

- Treats placement as a knapsack problem. Each weight unit gets a value of (expected accesses per token × bytes) ÷ (bandwidth of its tier).
- Rules, in priority order:
  1. **Always-active weights** go on the fastest device: attention, shared experts, embeddings/head, norms, routers, the KV cache, and the draft model. This is the KTransformers / llama.cpp-offload insight.
  2. **Routed experts** go into the VRAM pool, then the RAM pool, then SSD. Hot experts are placed first, using a calibration trace collected per model.
  3. Pick dense-layer splits by comparing two options: (a) compute the layer on the CPU in RAM, or (b) copy it to the GPU and compute there. Choose whichever is cheaper, as in KTransformers' CPU-expert approach.
- Output: a placement plan file. The planner re-plans when memory pressure changes.

### 3.3 Tiered Weight Store (core)
- **Weight unit**: one expert, meaning its gate, up and down matrices for one layer. For dense models the unit is a PowerInfer-style neuron bundle, or a whole layer block.
- **Slots**: fixed-size pools in VRAM (device buffers) and RAM (pinned, hugepage-backed). A per-layer remap table sends expert IDs to slot IDs, the same trick as the llama.cpp RFC, so kernels don't change.
- **SSD path**: async direct I/O that bypasses the page cache, which otherwise competes with our RAM pool.
  - Linux: `io_uring` + `O_DIRECT`
  - Windows: IOCP + `FILE_FLAG_NO_BUFFERING`
  - macOS: `F_NOCACHE` + parallel `pread`
  - Plain `mmap` remains as a simple fallback.
- **Eviction**: start with LFU+LRU with frequency-gated admission. Later, a small learned policy (FlashMoE-style).
- **Prefill rule**: if a micro-batch touches more experts than the pool has slots, it streams them without caching, so prefill can't wipe out the decode hot set.
- **Residency hints from the predictor**: `prefetch(layer, expert_ids, deadline)`.

### 3.4 Predictor service
Prediction sources, from cheapest to most accurate:
1. **Statistical**: per-layer expert frequency and co-activation tables from calibration plus the live session.
2. **Hidden-state look-ahead (training-free)**: run layer N+1's router on layer N's hidden state. Residual streams change slowly, which makes this accurate for many models ([Speculating Experts](https://arxiv.org/html/2603.19289v1)). Used only for **prefetching**; real routing is still exact.
3. **Trained prerouter heads ([Edge0](https://arxiv.org/html/2609.18063))**: tiny per-layer heads trained once per model, shipped as a small sidecar file. Exact mode uses them only for prefetching. "Fast mode" uses the prediction as the routing, plus an optional recovery LoRA.
4. **Draft-token look-ahead**: during speculative decoding, the draft tokens' hidden states predict which experts verification will need ([MoE-SpeQ](https://arxiv.org/html/2511.14102v1)).

Predictor accuracy and prefetch hit rate are always tracked and reported as metrics.

### 3.5 Scheduler
- Builds a DAG for each token: per-layer compute nodes plus transfer nodes (SSD→RAM, RAM→VRAM).
- Runs a double- or triple-buffered pipeline. While layer N computes, layer N+1's predicted experts are being loaded.
- Splits expert work between devices, choosing CPU or GPU per expert:
  - If an expert's weights are in RAM and it serves only a few tokens, the CPU computes it there.
  - If the expert is in VRAM, or the batch is large (prefill), the GPU computes it.
  - Based on HybriMoE.
- Thread pinning puts compute on P-cores and I/O on E-cores. NUMA-aware on multi-socket machines.
- Prefill: large batches on the GPU when one exists, with layer-wise weight streaming. Prefill is compute-bound, so the transfers amortize.

### 3.6 Speculative decoding
- On SSD- or RAM-bound setups, verifying K tokens costs about the same bytes as generating 1. **This is the biggest single gain for low-end PCs.**
- Draft sources:
  - EAGLE-3 heads, where available
  - Built-in MTP heads (DeepSeek, Qwen-Next style)
  - A small same-family draft model
  - Prompt-lookup / n-gram drafting with no extra model (great for code and RAG)
- K (draft length) adapts to the measured acceptance rate and the current cost per byte.
- The draft model is pinned in the fastest tier.

### 3.7 Quantization and kernels
- Load all existing GGUF quants. Prefer IQ/IQK-style and K-quants. Add QTIP/trellis-type support for 2–3 bit (later phase).
- **Mixed-precision export**: the "strata-pack" tool gives attention, shared experts and the head higher bits (Q6/Q8) and cold experts lower bits (Q2–Q3), using importance matrices ([unsloth-style dynamic quants](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF)).
- CPU kernels:
  - Reuse ggml/ik_llama kernels.
  - Add T-MAC LUT kernels for ≤3-bit decode on AVX2/NEON.
  - Use AMX on Intel Xeon/Core Ultra when present.

### 3.8 Activation sparsity (dense models, opt-in)
- TEAL-style magnitude thresholding at 25–50% sparsity, using calibrated per-layer thresholds.
- Sparse GEMV kernels that skip zeroed input channels, so fewer weight bytes are read.
- Combined with the TWS: for SSD-resident dense layers, read only the needed neuron bundles. This is "LLM in a Flash" windowing plus row-column bundling.

### 3.9 KV cache
- Paged KV store. Default q8 K/V; optional 4-bit Hadamard-rotated KV (TurboQuant/KIVI class).
- Optional spill of old KV pages to RAM or SSD for very long contexts, plus an attention-sink + recent-window eviction mode (SnapKV/StreamingLLM class) as a lossy option.
- Native MLA support (DeepSeek/Kimi) so long context fits in a small VRAM budget.

### 3.10 Strata pack file format (`.strata`)
GGUF stays the import format. `strata-pack` converts it into a streaming-friendly layout:
- Header + metadata are GGUF-compatible, so tooling still works.
- **Always-active block**: contiguous, loaded up front.
- **Expert blocks**: one contiguous, 4 KiB-aligned (direct-I/O-safe) blob per (layer, expert), holding gate+up+down together. This is row-column bundling.
- Experts that are often used together are placed next to each other on disk, using the co-activation tables.
- Sidecar files: calibration stats, prerouter heads, draft/EAGLE heads, and optional recovery LoRA.
- Original GGUF files are also supported directly through an index; they're just slower on SSD.

---

## 4. Tech stack

| Area | Choice | Why |
|---|---|---|
| Core language | C++20 | Same language as ggml, so there's no FFI cost |
| Base engine | **Fork of llama.cpp** (ggml + llama libs), pinned, rebased monthly | Confirmed decision. We get CPU/CUDA/HIP/Vulkan/Metal/SYCL and GGUF/model code; Strata code lives in new modules beside it to keep rebases clean |
| Async I/O | io_uring (Linux), IOCP (Windows), GCD/pread (macOS) | Best SSD throughput on each OS |
| Build | CMake + presets, CI release binaries for each backend | Same as the llama.cpp ecosystem |
| Server | Reuse llama-server patterns: OpenAI-compatible `/v1/chat/completions` | Works with existing tools |
| Bindings | C API → Python (nanobind) | Scripting and calibration tools |
| Calibration/training tools | Python + PyTorch (offline only) | Prerouter heads, thresholds, importance matrices |
| Benchmarks | Own harness + `llama-bench` comparison + perplexity/KLD | Fair comparison against the baseline |

---

## 5. Repository layout

```
strata/
  docs/                 RESEARCH.md, PLAN.md, design notes
  third_party/ggml/     pinned submodule
  src/
    hw/                 profiler
    plan/               placement planner
    tws/                tiered weight store, slot pools, async IO
    predict/            router predictors, stats
    sched/              per-token DAG scheduler, CPU/GPU split
    spec/               speculative decoding
    kv/                 paged + quantized KV store
    model/              architectures (llama, qwen-moe, deepseek, gpt-oss, ...)
    kernels/            extra CPU kernels (LUT, sparse GEMV)
    api/                C API
  tools/
    strata-cli/  strata-server/  strata-pack/  strata-bench/  strata-calibrate/
  python/               bindings + calibration/training scripts
  tests/                unit, kernel parity, perplexity regression
```

---

## 6. Phased roadmap

Each phase ends with a measurable exit criterion. **Benchmarks come first.**

### Phase 0: Baseline and benchmark harness (2–3 weeks)
- Reference machines: (A) 16 GB DDR4 no GPU, (B) 32 GB DDR5 + 8 GB GPU, (C) 64 GB + 24 GB GPU, (D) Apple 24–48 GB.
- Benchmark set:
  - Qwen3-30B-A3B, gpt-oss-120b, one 235B–400B MoE, Llama-3.x-8B/70B dense
  - Prompts: chat, code, long-context RAG
- Measure stock llama.cpp and ik_llama.cpp with the best manual flags. Record tok/s, TTFT, peak RSS, SSD bytes per token, PPL/KLD.
- Instrument expert-access traces (hit rates, reuse distance, co-activation).
- **Exit**: reproducible baseline table plus trace data showing what a VRAM/RAM cache *could* achieve.

### Phase 1: Fork llama.cpp + minimal Strata engine (3–4 weeks)
- Fork llama.cpp, pin it, add Strata modules beside `ggml`/`llama` so rebases stay clean.
- Validate the fork builds on Windows/Linux/macOS with CPU + CUDA + Vulkan + Metal.
- CLI and OpenAI-compatible server pass through (reuse `llama-server`), sampling, chat templates.
- **Exit**: perplexity matches upstream llama.cpp to ≤0.1% on identical quants; speed within 5% when everything fits; clean build on all three OSes in CI.

### Phase 2: Hardware profiler + placement planner (2–3 weeks)
- Auto-placement replaces manual `-ngl` / `-ot` / `--n-cpu-moe` flags.
- **Exit**: auto plan is ≥95% of the best hand-tuned llama.cpp config on all reference machines.

### Phase 3: Tiered Weight Store + trillion-scale streaming (5–6 weeks), the core of the project
- Slot pools (VRAM + RAM), ID remap, LFU/LRU eviction, prefill bypass.
- Async direct-I/O SSD path (`O_DIRECT`/io_uring, IOCP, `F_NOCACHE`) on all three OSes.
- `strata-pack` with expert-contiguous aligned layout (generalizes the kimi-k3-in-c "packed trunk").
- Trunk/expert split with a `--trunk-gb` / `--cache-gb`-style memory dial.
- Trace capture + offline cache-size simulator (like `sim_cache.py`).
- **Exit**:
  - **The owner's 800B Kimi model runs on an 8–16 GB no-GPU box with bounded RSS** (no OS swapping), byte-identical output across memory budgets.
  - ≥1.5× llama.cpp mmap on machine A for gpt-oss-120b.
  - VRAM cache gives ≥+50% on machine B.
  - On the 800B Kimi run, match or beat kimi-k3-in-c tok/s at an equal RAM budget on the same disk.

### Phase 4: Predictor + pipelined prefetch (4–5 weeks)
- Statistical and hidden-state look-ahead predictors (training-free).
- Per-token DAG scheduler with transfer/compute overlap; CPU/GPU expert split.
- **Exit**: prefetch hit rate ≥70% on the benchmark set; SSD stall time per token cut by ≥50% versus Phase 3.

### Phase 5: Speculative decoding (3–4 weeks)
- n-gram/prompt-lookup, draft model, MTP, then EAGLE-3. Adaptive K. Expert prefetch from draft tokens.
- **Exit**: ≥1.8× on code/RAG prompts on machine A for a large MoE, with output identical to non-speculative decoding (greedy).

### Phase 6: Quantization and kernels (4–6 weeks, can run alongside Phases 4–5)
- Mixed-precision `strata-pack` presets (always-active high-bit, cold-expert low-bit).
- T-MAC LUT kernels for 2–3 bit on AVX2/NEON; AMX path.
- Quantized/paged KV (q8, 4-bit Hadamard).
- **Exit**:
  - 2-bit LUT kernels are ≥1.5× ggml's on CPU decode.
  - 32k context fits on machine A for a 30B MoE.

### Phase 7: Advanced and opt-in features (ongoing)
- Trained prerouter heads + "fast mode" approximate routing + recovery LoRA (Edge0-style).
- TEAL activation sparsity + neuron-bundle SSD reads for dense models.
- QTIP/trellis 2-bit formats.
- Learned cache eviction.
- Multi-device home cluster (prima.cpp piped-ring / exo-style).
- NPU backends (Intel/AMD/Qualcomm) once they're mature.

**Rough total to a strong v1 (Phases 0–6): about 6–8 months for 2–3 engineers.**

---

## 7. Test and quality plan
- **Kernel parity tests** for every new kernel against ggml reference output: bit-exact or within a tolerance.
- **Perplexity/KLD regression** in CI on small models for every quant and mode. Approximate modes must report their measured quality cost.
- **Determinism**: under greedy decoding, cache, prefetch and speculative paths must produce output identical to the plain path.
- **Soak tests**: multi-hour sessions with no memory growth, no throughput decay, and no SSD write traffic (reads only).
- **Cross-platform CI**: Linux/Windows/macOS × CPU/CUDA/Vulkan/Metal.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Users expect dense 70B+ to be fast on 16 GB | Communicate the bandwidth limits clearly in docs and in a CLI "expected speed" estimate before loading |
| Keeping up with ggml upstream | Pin a version, keep Strata changes outside ggml, rebase monthly |
| Each new model architecture needs work | Use a generic graph builder for llama/MoE families first; add architectures by usage |
| SSD speeds vary widely (SATA, DRAM-less NVMe) | The profiler measures them; the planner falls back to RAM-only plans and warns |
| Windows direct I/O and pinned memory quirks | Keep an mmap fallback path and test Windows early |
| Router prediction accuracy varies by model | Exact mode never depends on it (prefetch only); trained heads are optional |
| SSD energy and heat on laptops | Power-saver preset that limits I/O depth |
| Licensing | ggml/llama.cpp are MIT. Check each paper's code license before porting (PowerInfer, KTransformers, T-MAC are permissive, but verify) |

---

## 9. Decisions

Resolved by the owner:
- **Fork llama.cpp libs** (not a from-scratch engine). ✓
- **All three OSes** (Windows/Linux/macOS) from v1. ✓
- **Owner: Coaade Inc.**, license **Apache-2.0**. ✓
- **Primary validation: an ~800B Kimi MoE** with disk-streamed experts; CPU+SSD streaming is a v1 must-have. ✓

Still open:
1. **First dev platform for the fastest loop** (Linux has the easiest `io_uring` path) — build order only; all three still ship.
2. **Approximate "fast mode"** (predicted routing / activation sparsity): in scope for v1, or exact-only first? kimi-k3-in-c is exact-only; recommend exact-first, fast-mode opt-in later.
3. **Team size and timeline** — affects whether Phase 6 runs in parallel.
4. **GitHub org/repo** under Coaade Inc. to push this to.

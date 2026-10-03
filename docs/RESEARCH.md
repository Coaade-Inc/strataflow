# Research: Running Huge LLMs on Low-End and Mid-Range PCs

Working name for the project: **StrataFlow**. The name comes from the core idea: model weights sit in memory *layers*. Those layers are VRAM, RAM, and SSD.

*Content from external sources has been rephrased for compliance with licensing restrictions. Sources are linked inline.*

---

## 1. Hardware limits (read this first)

For single-user chat, **decode speed is limited by memory bandwidth, not by compute.** Every generated token has to read every *active* weight once:

```
tokens/sec  ≈  effective_bandwidth  /  bytes_read_per_token
```

| Tier | Typical bandwidth |
|---|---|
| SATA SSD | ~0.5 GB/s |
| NVMe Gen3 / Gen4 / Gen5 | ~3 / ~7 / ~12 GB/s |
| PCIe 4.0 x16 (RAM → GPU) | ~25 GB/s |
| DDR4 dual-channel (practical) | ~35–45 GB/s |
| DDR5 dual-channel (practical) | ~60–80 GB/s |
| Apple M-series unified | 100–800 GB/s |
| Consumer GPU VRAM | 250–1800 GB/s |

What this means for the project:

- **Dense 70B at 4-bit on a 16 GB machine:** each token reads about 40 GB, and about 25 GB of that has to come from SSD. Speed is around **0.1 tok/s**. No framework can change this. All you can do is read fewer bytes.
- **MoE models get around the limit.** A 397B-A17B model reads only about 17B parameters per token. At 3–4 bits that's about 7–9 GB per token. Most of that hits a RAM or VRAM cache, so only a small part comes from SSD.
- So "hundreds of billions of parameters on 16 GB at usable speed" is **only possible for sparse models** (MoE, or dense models with activation sparsity). Strata's design should start from that fact.

**Every byte not read is a speed gain.** The techniques below fall into five groups:
1. Store fewer bits (quantization).
2. Read fewer weights (MoE routing, activation sparsity).
3. Read from a faster tier (caching, placement).
4. Hide reads behind compute (prefetching, pipelining).
5. Produce more tokens per read (speculative decoding).

---

## 2. Existing frameworks and what to take from each

| Project | Key idea | What we reuse |
|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | Portable C/C++ engine, GGUF format, mmap loading, CPU SIMD + CUDA/Vulkan/Metal/SYCL/HIP backends, `--n-cpu-moe` tensor overrides | GGUF format, ggml kernels, backends. This is the foundation. |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | llama.cpp fork with stronger quant types, faster CPU and hybrid paths, fused MoE ops, MLA/FlashMLA, row-interleaved quant packing | IQK quant types and the fused MoE kernel ideas |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) ([paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)) | CPU/GPU hybrid for MoE. Attention and shared parts run on the GPU, routed experts run on the CPU with AMX/AVX-512 kernels. Ran DeepSeek 671B Q4 with about 14 GB VRAM plus a lot of DRAM. | The hybrid split rule: the always-active parts go on the fastest device |
| [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) ([paper](https://arxiv.org/abs/2312.12456)) | Neuron activations follow a power law. "Hot" neurons are kept on the GPU, "cold" neurons are computed on the CPU, and small predictors choose which neurons to compute. | Neuron-level hot/cold placement and activation predictors |
| [PowerInfer-2](https://arxiv.org/abs/2406.06282) | Ran a 47B model on a phone at about 11.7 tok/s. Splits matrices into neuron clusters, caches them by segment, and pipelines I/O against compute. | Neuron-cluster I/O pipelining and segmented cache |
| [Apple "LLM in a Flash"](https://arxiv.org/abs/2312.11514) | Weights stay on flash and are loaded on demand. "Windowing" reuses recently active neurons. "Row-column bundling" makes reads larger and contiguous. | Disk layout co-designed with access pattern. Sliding-window reuse. |
| [flash-moe](https://github.com/danveloper/flash-moe) | Plain C/Metal engine. Streams experts of Qwen3.5-397B-A17B from SSD on a 48 GB MacBook at about 4.4 tok/s. | Shows that SSD expert streaming works for very large MoE models |
| [AirLLM](https://github.com/lyogavin/airllm) | Loads one layer at a time, so a 70B model fits on a 4 GB GPU | Fallback mode only: it fits anything, but it's very slow (reads the whole model every token) |
| [prima.cpp](https://arxiv.org/abs/2504.08791) | Runs 30–70B models across several home devices over Wi-Fi. Uses mmap, "piped-ring" parallelism, and prefetching. A device-aware solver places layers. | A solver that assigns layers to devices based on measured speeds. Multi-device support later. |
| [exo](https://github.com/exo-explore/exo) | Clusters home devices for inference | Multi-device support later |
| [bitnet.cpp](https://github.com/microsoft/BitNet) / [T-MAC](https://arxiv.org/abs/2407.00088) | Lookup-table matmul for 1–4 bit weights, with no dequantization step. Reports 2–6× CPU speedups and large energy savings. | LUT kernels for very-low-bit CPU decode |
| [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE) | Dynamic CPU-GPU scheduling of MoE experts plus cache management | Balancing expert work between CPU and GPU |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | Uses request-level activation traces to drive expert prefetch and caching | Trace-driven prefetch |
| [Edge0](https://arxiv.org/html/2609.18063) (Sep 2026) | A trained per-layer "prerouter" predicts the next layer's routing one token ahead, so SSD reads overlap compute. Serves a 35B MoE at about 20 tok/s within about 3 GiB active memory. | **Most relevant recent result.** Predicted routing makes SSD streaming fast. |
| [llama.cpp expert slot pool RFC](https://github.com/ggml-org/llama.cpp/discussions/28248) | Persistent LRU pool of experts in VRAM, built only by remapping expert IDs. Reports up to +84% decode on a 4090. Prefill skips the pool so it doesn't evict the hot set. | The VRAM expert cache design, including the rule that keeps prefill out of the cache |
| [FlashMoE (edge)](https://arxiv.org/html/2601.17063v1) | ML-based expert cache replacement that mixes recency and frequency | Smarter eviction than plain LRU |
| **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** (Apache-2.0) — **primary reference** | Portable C99, no BLAS/framework/GPU. Runs the **2.78T-param Kimi K3** (1.56 TB on disk) on **one CPU in 8.24 GB RAM** by keeping the always-on "trunk" in memory and streaming the 1.45 TB of routed experts off disk. Output is byte-identical from the 8 GB machine to a 128 GB+ one; **more RAM only buys speed.** | The whole streaming strategy, concretely: trunk-vs-expert split, the dial-shaped memory budget, the LRU expert cache, trace-driven cache sizing, `O_DIRECT` reads, and the honest reporting of I/O share of wall-clock. Confirms Strata's core thesis is real. |
| [MoE-SpeQ](https://arxiv.org/html/2511.14102v1), [Speculating Experts](https://arxiv.org/html/2603.19289v1) | A draft model or internal hidden states predict which experts upcoming tokens will need | Prefetching experts ahead of time |
| [TEAL](https://github.com/FasterDecoding/TEAL) | Training-free activation sparsity of 40–50%, giving about 1.5–1.8× decode on dense models | Speeds up dense models with no retraining |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) | A drafter reads the target model's hidden states and proposes tokens. Up to 6.5× reported on GPU, and llama.cpp now supports it. | Speculative decoding. On CPU or SSD-bound setups, verifying K tokens costs about the same as producing 1. |
| [QTIP](https://arxiv.org/abs/2406.11235) / EXL3 | Trellis-coded quantization. Currently the best published 2-bit post-training quality. | High-quality 2–3 bit formats for when the model barely fits |
| TurboQuant / KIVI / SnapKV ([benchmark](https://arxiv.org/html/2607.05399v1)) | KV cache compression: 3–4 bit KV (Hadamard rotation + quantization), plus token eviction | Long context without running out of memory |
| [Vulkan backend](https://www.technolynx.com/post/vulkan-vs-cuda-llama-cpp/) | One binary for AMD, Intel, NVIDIA and integrated GPUs. Peak speed is lower than CUDA. | Default GPU path for non-NVIDIA cards and iGPUs |

---

## 3. The kimi-k3-in-c recipe (what the 8 GB / 2.78T claim actually is)

This repo is the clearest proof that the goal is reachable, so its mechanics are worth stating exactly:

- **MoE sparsity is the whole trick.** Kimi K3 has 93 layers; 92 of them route and each picks **16 of 896 experts** per token. About **104B of the 2.78T parameters (3.7%) are active per token.** The other 96.3% must merely be *reachable*, not resident.
- **Split the model into "trunk" and "experts."** The always-on trunk (attention, shared parts, router, norms, embeddings/head) stays in memory to whatever depth you choose. The ~1.45 TB of routed experts are **never resident** — they're read from disk on demand and multiplied straight out of their packed form.
- **Memory becomes a dial, not a floor.** 8 GB runs it slowly (whole trunk streams every step); 128 GB+ holds everything and the disk wait disappears. **The answer is byte-identical at every size; only the clock changes.**
- **An LRU cache for routed experts**, sized from a replayed access trace (`sim_cache.py`). Reports distinguish *true resident hits* from experts the prefetcher pulled off disk moments earlier.
- **Honest bottleneck accounting.** On a streamed run, I/O was ~41–71% of wall-clock. The run report prints the I/O share, because disk — not arithmetic — is the limit.
- **`O_DIRECT` + `posix_memalign` + `pread`**, with a Windows (MSYS2/MinGW) and macOS/arm64 port. Experts ship at ~half a byte (sub-4-bit), the trunk is kept at higher precision.
- **License: Apache-2.0.** No model weights included.

Google has published related work (Gemma, and the Apple "LLM in a Flash" line is the closest public method), but no public Google repo matches the "trillions-of-params on 8 GB" framing; kimi-k3-in-c is the concrete reference. The Strata plan adopts its trunk/expert streaming model as the baseline and extends it with GPU tiers, prediction-driven prefetch, auto hardware tuning, and speculative decoding.

## 4. Gaps no current project covers

Each of these ideas exists in some project, but **none combines them, auto-tunes them to the user's hardware, works across GPUs, and stays easy to use:**

1. **One tiered weight store across VRAM, RAM and SSD.** llama.cpp has the GPU/RAM split and mmap, flash-moe and kimi-k3-in-c have CPU+SSD streaming, and the RFC has a VRAM cache. These are separate mechanisms, not one cache hierarchy with a shared policy. kimi-k3-in-c is CPU-only — it leaves all GPU acceleration on the table.
2. **Routing prediction that drives prefetch on all backends.** Edge0 and MoE-SpeQ show it works, but no mainstream engine ships it. kimi-k3-in-c prefetches but does not predict routing ahead of the layer.
3. **Automatic hardware profiling and placement.** Users still tune `-ngl`, `-ot`, `--n-cpu-moe`, or `--trunk-gb`/`--cache-gb` by hand. prima.cpp's solver points the way.
4. **Speculative decoding that also prefetches experts.** The draft tokens can tell us which experts to load.
5. **Disk layout built for streaming.** GGUF is laid out by tensor. SSD streaming wants per-expert, per-layer contiguous blocks that are aligned for direct I/O — exactly what kimi-k3-in-c's "packed trunk" does, which Strata generalizes to a `.strata` format.

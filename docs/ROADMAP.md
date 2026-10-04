# StrataFlow Roadmap and Status

The single source of truth for what is done and what is open. For the overall
goals and research grounding see [`PLAN.md`](./PLAN.md); for the engine design
see [`ENGINE_CORE_DESIGN.md`](./ENGINE_CORE_DESIGN.md).

Legend: [x] done and merged, [~] partial, [ ] not started.

---

## Mission

Run very large Mixture-of-Experts models (100B up to trillion-scale), **in any
GGUF weight type - quantized or full-precision**, on everyday hardware **without
a GPU**, in a bounded RAM budget, by streaming experts from disk. A GPU, if
present, is an optional accelerator - never a requirement.

Running real pretrained models means running **quantized** weights (Q4_K, Q6_K,
Q8_0, IQ-families, ...), since that is how real models ship; full-precision
(F32/F16) must work too. Quantized-weight support is the current top-priority
gap (see below) - the engine runs F32 only today.

---

## Done (merged to main)

- [x] **Phase 1 / 1b** - fork llama.cpp, vendor ggml, minimal engine, CI on
  Linux/macOS/Windows (PRs #1, #2).
- [x] **Phase 2** - hardware profiler + placement planner driving load params (#3).
- [x] **Phase 3** - tiered weight store, `strata_stream_buft`, SSD profiling,
  bounded SlotPool residency, native direct-I/O, the `.strata` format design and
  feasibility spike (#4-#10).
- [x] **Engine core EC-1..EC-7** - StrataFlow's own forward pass over ggml, no
  longer using `llama_decode` for inference (#11-#19):
  - [x] EC-1 own ggml graph builder (llama-arch MoE), oracle-gated.
  - [x] EC-2 engine-owned KV cache + multi-token decode.
  - [x] EC-3 per-layer segmented execution + top-k expert residency (bounded).
  - [x] EC-4 `.strata` single-file loader through our engine.
  - [x] EC-5 engine is the default and only inference path; llama.cpp out of the
    forward pass (kept only for the tokenizer and as the test oracle).
  - [x] EC-6 predictor-driven expert prefetch (synchronous cache-warming).
  - [x] EC-7 second architecture: dense llama.
- [x] **Bounded-RAM fix** - the `.strata` path holds only the trunk resident;
  the full expert footprint is never allocated (#21). Proven by test: 103 KB
  trunk resident vs 393 KB experts not resident on the tiny MoE.
- [x] Relicense to the Coaade Source-Available License v1.0 (#6, #7).
- [x] Byte-identical correctness gate against the libllama oracle on every
  engine step; CI green on Linux (GCC+Clang), macOS, Windows.

---

## Open (not done)

These are the honest gaps. None are blocked by GPU hardware except where noted;
most are pure CPU/disk work.

### Must-do to actually prove the mission

- [x] **Run quantized models (was TOP PRIORITY, now DONE for the common case).**
  The engine previously forced `GGML_TYPE_F32` for the per-layer staging expert
  tensors, so it could not run quantized models. Fixed: the staging tensors now
  take the SOURCE expert tensor's real ggml type, so quantized experts stream
  and compute through `ggml_mul_mat_id` natively (slot and `.strata` blob sizing
  already came from the real tensor bytes; the packer copies verbatim). Attention
  and embedding/output weights already load at their real type. Validated against
  the libllama oracle on a **Q8_0** llama-arch MoE: identical greedy sequence,
  logit delta within quant-dequant tolerance; the `.strata` streaming path works
  on the quantized model too.
  - [ ] Still to broaden: validate the K-quant families (Q4_K/Q6_K) and the IQ
    families specifically, and confirm `.strata` streaming across all of them.
    Q8_0 is proven; the mechanism is type-agnostic (it uses the source type), so
    the remaining work is validation + any per-family edge cases, not a redesign.
- [ ] **Run a real, large MoE end to end.** Everything so far is verified on a
  2-layer F32 toy fixture. The mission is not proven until an actual multi-GB,
  real (quantized) MoE runs end to end in a bounded RAM budget on a real machine
  with sensible output at a usable speed. Depends on quant support above. Likely
  to surface bugs the toy never did (real tokenizers, large shapes, quant types).
- [ ] **Memory-budget CLI + peak-RSS reporting.** A user cannot yet say "run this
  model in 8 GB" and see the result. Expose `--trunk-gb` / `--cache-gb` /
  expert-slot knobs on the CLI and report peak resident memory, so the bounded
  budget is user-settable and observable.
- [ ] **Benchmark harness with real numbers.** No real tok/s, TTFT, peak RSS, or
  SSD bytes/token measured against the exit criteria in `PLAN.md` and
  `PHASE3_PLAN.md`. (This is the long-promised Phase 0 baseline, now needing the
  engine measured too.)

### Breadth and performance

- [ ] **GPU / multi-backend execution.** CPU + a GPU buffer type via
  `ggml_backend_sched`. The engine already runs on a swappable ggml backend, so
  this is structurally reachable, but it cannot be validated on the CPU-only CI
  or sandbox - it needs real GPU hardware. Optional accelerator, not required for
  the no-GPU mission.
- [ ] **More architectures.** Only llama-arch MoE and dense llama are supported.
  Qwen-MoE (shared expert + per-expert gate), DeepSeek-style (expert groups,
  shared experts, sigmoid gating with group-limited top-k), and others are each
  a new oracle-gated graph builder per the arch-descriptor plan
  (`ENGINE_CORE_DESIGN.md` section 6.4).
- [ ] **Async prefetch overlap.** EC-6 does synchronous cache-warming. True
  background-I/O overlap (load layer N+1's experts on a thread while layer N
  computes) is deferred, along with per-layer graph-build reuse across tokens
  (`ENGINE_CORE_DESIGN.md` section 3.5).

### From the original PLAN.md (Phases 4-7), as written

- [~] **Phase 4 - predictor + pipelined prefetch.** Statistical predictor done
  (EC-6); pipelined/async overlap not done (see above).
- [ ] **Phase 5 - speculative decoding** (EAGLE / draft / n-gram / MTP). Not
  started.
- [ ] **Phase 6 - quantization and kernels.** Mixed-precision `.strata` presets,
  T-MAC/LUT low-bit CPU kernels, quantized/paged KV. Not started.
- [ ] **Phase 7 - advanced / opt-in.** Trained prerouter heads, activation
  sparsity, trellis 2-bit, multi-device home cluster, NPU backends. Not started.

### Release engineering

- [ ] **No GitHub release / version tag.** The repo has no `v0.x` cut yet.
- [ ] **OpenAI-compatible HTTP server** front-end (planned in `PLAN.md` section 3)
  is not wired to the engine yet; only the CLI is.

---

## Notes on sequencing

Recommended order, all CPU/disk work (no GPU needed):

1. **Quantized-weight support.** DONE for the common case (Q8_0 validated against
   the oracle; the path is type-agnostic). Next: validate K-quant and IQ
   families and `.strata` streaming across them.
2. **Memory-budget CLI + peak-RSS reporting.** DONE: `--expert-slots` plus a
   `stats:` line reporting resident model-weight bytes and peak RSS.
3. **Run a real, quantized pretrained MoE** (an actual downloaded model, not a
   generated fixture) through the `.strata` streaming path and record real tok/s
   and peak RAM. This is now unblocked by (1) and (2) and is the key remaining
   proof point.

The mechanism is proven on a generated quantized model; the open work is running
an actual downloaded model and broadening quant-family coverage.

# StrataFlow Roadmap and Status

The single source of truth for what is done and what is open. For the overall
goals and research grounding see [`PLAN.md`](./PLAN.md); for the engine design
see [`ENGINE_CORE_DESIGN.md`](./ENGINE_CORE_DESIGN.md).

Legend: [x] done and merged, [~] partial, [ ] not started.

---

## Mission

Run very large Mixture-of-Experts models (100B up to trillion-scale) on
everyday hardware **without a GPU**, in a bounded RAM budget, by streaming
experts from disk. A GPU, if present, is an optional accelerator - never a
requirement.

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

- [ ] **Run a real, large MoE end to end.** Everything so far is verified on a
  2-layer toy fixture. The mission is not proven until an actual multi-GB MoE
  runs end to end in a bounded RAM budget on a real machine, byte-reasonable
  output at a usable speed. First real exercise of the engine; likely to surface
  bugs the toy never did (real tokenizers, large shapes, quant types beyond F32).
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

The highest-value next work is the top of "Must-do to prove the mission":
expose the memory-budget CLI + RSS reporting, then run a real mid-size MoE
(small enough to fit a modest disk, larger than RAM) through the `.strata`
streaming path and record the numbers. That turns a unit-proven mechanism into a
demonstrated capability. All of it is CPU/disk work and needs no GPU.

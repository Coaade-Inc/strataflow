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
(F32/F16) must work too. Quantized-weight support now runs: Q8_0 and the K-quant
families (Q4_K/Q6_K) are validated against the oracle (see below); the IQ
families are the remaining quant-coverage gap.

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
- [x] **Legacy per-expert Mixtral GGUF support (packer-only).** Real TheBloke
  Mixtral GGUFs name routed-expert FFN weights with the LEGACY, UN-STACKED,
  PER-EXPERT convention `blk.N.ffn_{gate,down,up}.E.weight`, not the stacked
  `blk.N.ffn_{gate,down,up}_exps.weight` 3-D form. Confirmed against vendored
  llama.cpp b11379: the runtime loader does NOT stack legacy tensors (the
  legacy -> stacked regroup happens only in the Python convert script, never at
  GGUF load), so the smallest correct fix is PACKER-ONLY: `strata-pack` now
  recognizes the legacy per-expert names, GROUPS them per (layer, expert) into
  the existing stacked `.strata` expert region (verbatim bytes, no requant), and
  SYNTHESIZES the embedded GGUF metadata to declare stacked `_exps` tensors. The
  engine, `StrataReader`, `strata_format`, and the `mul_mat_id` addressing are
  UNCHANGED (they still see only stacked names). Oracle-gated on generated
  legacy fixtures (incl. a Q3_K variant mirroring the real Mixtral Q3_K_M) with
  a matched stacked-equivalent oracle: the legacy-sourced `.strata` decodes
  byte-identically to the libllama oracle on the stacked model, with bounded
  expert streaming. The real multi-GB Mixtral run stays Colab-only (the real
  GGUF cannot be downloaded offline); the in-sandbox reproduction is these
  fixtures.
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
  - [x] K-quant families (Q4_K/Q6_K) validated against the libllama oracle on
    generated fixtures: the engine greedy token SEQUENCE is byte-identical to the
    oracle for both Q4_K and Q6_K, and K-quant experts stream through the
    `.strata` path (confirmed via `strata-pack` + the CLI with bounded
    `--expert-slots`). Gated by a C++ ggml-based fixture generator
    (`make_kquant_moe_gguf`, using `ggml_quantize_chunk` since the Python `gguf`
    library cannot emit K-quants) and the guarded
    `test_engine_kquant_matches_oracle` sub-test
    (`STRATAFLOW_TEST_KQUANT_MOE_GGUF` / `STRATAFLOW_TEST_Q6K_MOE_GGUF`).
  - [ ] Still to broaden: validate the IQ families specifically and confirm
    `.strata` streaming across them. The mechanism is type-agnostic (it uses the
    source type), so the remaining work is validation + any per-family edge
    cases, not a redesign.
- [ ] **Run a real, large MoE end to end.** Everything so far is verified on a
  2-layer F32 toy fixture. The mission is not proven until an actual multi-GB,
  real (quantized) MoE runs end to end in a bounded RAM budget on a real machine
  with sensible output at a usable speed. Depends on quant support above. Likely
  to surface bugs the toy never did (real tokenizers, large shapes, quant types).
- [x] **Memory-budget CLI + resident/peak-RSS reporting.** DONE (#25): the CLI
  takes `--expert-slots` and prints `stats: resident model weights = X MiB,
  peak RSS = Y MiB` via `sf_session_stats`.
- [x] **Per-layer staging no longer scales with model size (was CRITICAL).**
  Root cause found and fixed: the engine allocated the per-layer staging stacked
  expert tensors at full size for EVERY layer
  (`n_layer x 3 x n_embd x n_ff x n_expert`), which equalled ~the whole model.
  Fix: use ONE staging buffer reused across layers (segments run sequentially),
  bounding staging to a SINGLE layer's expert footprint. Measured: a 785 MiB
  model dropped from 806 MiB -> ~132 MiB peak RSS; a 1177 MiB model also runs in
  ~132 MiB. What this fix bounds, precisely: it removes the per-LAYER growth of
  the staging buffer. It does NOT make whole-process peak RSS move with
  `--expert-slots`. The ~132 MiB is essentially the FIXED llama/ggml backend +
  vocab process floor (reproduced in-sandbox: a 3.9 MiB model AND a `--plan-only`
  run with no decode both measure ~132 MiB), plus a one-layer staging buffer
  sized to the layer's FULL `n_expert` (not to the slot budget) plus the ggml
  compute buffer. So peak RSS is roughly flat across model size for these
  layer-count-similar models and does NOT shrink as `--expert-slots` shrinks.
  The quantities that ARE bounded and DO move with `--expert-slots` are
  StrataFlow's own resident model weights (trunk only) and the SSD bytes
  streamed per token; those are what make "big model, small RAM" true, not the
  peak-RSS figure. (Shrinking staging to the slot budget is NOT safely possible
  without a custom `mul_mat_id` kernel: the stacked staging tensor must carry
  the full `ne[2]` for `i02*nb02` addressing - see `src/tws/stream_buft.h`
  "Model A". So the honest reporting above is the resolution, not a tighter
  peak-RSS bound.)
- [ ] **Misleading placement-plan line.** On the `.strata` streaming path the
  CLI still prints `plan: ... experts fully resident, peak ~N GiB` from the
  planner's estimate, which contradicts the streaming reality. Make the plan
  summary reflect streaming (bounded resident) when experts stream.
- [ ] **`n_ctx_train (0)` warning on generated models.** The tiny/colab
  generators do not write a trained context-length key, so llama warns
  `n_ctx_seq (256) > n_ctx_train (0)`. Harmless but noisy; set a context-length
  in the generators (or handle 0 cleanly) so demo output is clean.
- [~] **Run a real DOWNLOADED pretrained model end to end.** The Colab demo now
  has a `--real-model` flow (`colab/strataflow_colab.py` + `colab/README.md`):
  download a real quantized GGUF from HuggingFace, pack to `.strata`, run bounded
  with `--expert-slots`, and record decoded output + tokens/sec + peak RSS, with
  disk/RAM guards. The default is a small quantized dense-llama model
  (TinyLlama-1.1B-Chat Q4_K_M) because the smallest llama-arch MoE (Mixtral) is
  ~24 GB+; a Mixtral MoE run is exposed via flags for a larger runtime. This is
  DEMONSTRATED on Colab (the sandbox/CI is offline); the next step is capturing
  the measured numbers from a real Colab run. Unblocked by quant support (#26)
  and K-quant validation (K-quant Q4_K/Q6_K oracle gate).
- [~] **Benchmark harness with real numbers.** A harness now exists
  (`colab/strataflow_bench.py` + Cells 9-10 in `colab/README.md`): it runs a
  config MATRIX (model size x `--expert-slots`), measures per run tok/s, TTFT,
  resident weights, peak RSS, on-disk `.strata` size, SSD streamed MiB and
  bytes/token (TTFT + streamed-bytes are the FEAT-002 instrumentation), prints a
  fixed-width table, and writes JSON + CSV artifacts. The GENERATED-F32 MoE path
  is sandbox/CI-verified and demonstrates the Phase 3 criterion 1 "big model,
  small RAM" (peak RSS/resident far below on-disk size and moving with
  `--expert-slots`). Still [~], not [x]: the absolute tok/s ladder in `PLAN.md`
  section 2 is for large REAL models on NVMe, so the generated toy's throughput
  is a mechanism proxy only - real large-model throughput against that ladder
  stays Colab/HW-demonstrated (optional `--real-model` row / a Mixtral quant on
  a larger runtime), not proven in the offline sandbox/CI.

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

1. **Quantized-weight support.** DONE for the common case: Q8_0 and the K-quant
   families (Q4_K/Q6_K) are validated against the oracle (identical greedy token
   sequence) and stream through `.strata`; the path is type-agnostic. Next:
   validate the IQ families and `.strata` streaming across them.
2. **Memory-budget CLI + peak-RSS reporting.** DONE: `--expert-slots` plus a
   `stats:` line reporting resident model-weight bytes and peak RSS.
3. **Run a real, quantized pretrained MoE** (an actual downloaded model, not a
   generated fixture) through the `.strata` streaming path and record real tok/s
   and peak RAM. This is now unblocked by (1) and (2) and is the key remaining
   proof point.

The mechanism is proven on a generated quantized model; the open work is running
an actual downloaded model and broadening quant-family coverage.

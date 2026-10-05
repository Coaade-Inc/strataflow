# Changelog

All notable changes to StrataFlow. This project has not yet cut a versioned
release; everything below is pre-release work on `main`. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what is done and what is open.

The format is loosely based on Keep a Changelog. Dates are omitted; entries are
grouped by the pull request that merged them.

## Unreleased

### Features

- Async/overlapped expert prefetch (hide disk reads behind compute). A dedicated
  background I/O worker thread now loads the next layer's predicted experts off
  disk CONCURRENTLY with the current layer's ggml compute, so read latency hides
  behind compute on the bounded-RAM streaming path. The synchronous
  `ensure_layer_experts_resident` stays the AUTHORITATIVE load: a mispredicted or
  still-in-flight prefetch is a plain miss it corrects inline, so decode is
  byte-identical to the libllama oracle, including under bounded-pool eviction
  (the oracle gate's new async sub-test shows async-vs-sync `|delta| =
  0.000e+00`, identical across repeated runs that perturb prefetch timing). The
  ggml compute path stays single-threaded (only an I/O thread was added); the
  worker never touches the non-thread-safe `SlotPool` (it reads into its own
  staging buffer and the compute thread installs the bytes), so it is
  thread-safe and TSan-clean. A default-ON runtime toggle flows through
  `sf_context_params.async_prefetch` (appended at the END of the struct for ABI
  backward-compatibility; default 1), the CLI `--async-prefetch on|off`, and the
  `STRATAFLOW_ENGINE_ASYNC_PREFETCH` env var; `off` falls back to the
  synchronous warm path. A new overlap metric, `completed-before-use` (of the
  prefetched experts the routing then used, how many the worker had fully read
  AND installed BEFORE the layer's authoritative acquire needed them), is
  surfaced through `sf_runtime_stats` (three fields appended at the END of the
  struct) and printed on the CLI stats line (appended as new `, key = value`
  fields, so the existing parser is unaffected). `colab/strataflow_bench.py`
  gains an `--async-list on,off` sweep and an on-vs-off contrast that confirms
  byte-identical decoded text while reporting `completed-before-use`, streamed
  bytes/token, decode tok/s and TTFT. In-sandbox the MECHANISM is proven on the
  generated MoE (`--layers-list 8 --experts-list 16 --slots-list 4 --async-list
  on,off`): `completed-before-use` goes from 0 (async off) to > 0 (async on) with
  byte-identical output; absolute wall-clock speedup is hardware-dependent and
  stays Colab-only on the real Mixtral (exact `--async-prefetch on`/`off`
  re-verify commands added to `colab/README.md`, auto residency bounded by
  `--cache-gb`). `docs/ROADMAP.md` flips the async-overlap and Phase 4 items to
  done.
- RAM-aware, per-model AUTO expert-residency. StrataFlow now chooses the
  resident expert-bundle pool size automatically per model, from the measured
  free-RAM budget plus the model's own expert layout (`n_layer`, `n_expert`,
  `n_expert_used`, `expert_bytes`), with NO model-specific constants, so it
  adapts to any MoE layout (8x2, 256x8, ...). `expert_slots_resident =
  clamp(ram_for_experts / expert_bytes, lower = max(n_expert_used, a whole layer
  when it fits), upper = n_layer*n_expert)`. When RAM holds the whole working set
  the engine caches it with no per-layer eviction/re-streaming; only when RAM
  genuinely cannot hold it does the pool bound below the full set, and never
  below one layer's top-k. New `--cache-gb` flag caps the expert-RAM budget
  (wired to `sf_context_params.cache_budget`, appended at the end of the struct
  for ABI backward-compatibility). Precedence: explicit `--expert-slots` >
  `--cache-gb` > auto-from-free-RAM. The placement-plan summary is now TRUTHFUL
  on the streaming path: it reports the slot count the engine actually uses, the
  correct state (`experts fully resident` when the whole working set fits, else
  `experts STREAMED (bounded)`), and a realistic peak estimate (trunk + resident
  bundles + one-layer staging), so what RUNS equals what is REPORTED. Oracle
  byte-identity is PRESERVED: an auto-chosen slot count decodes a byte-identical
  greedy sequence/logits to the fully-resident run and to a fixed-slot run
  (oracle-gated tests). Sandbox-proven on generated fixtures via
  `colab/strataflow_bench.py --layers-list 8 --experts-list 16 --slots-list
  2,16,0`: the auto row streams ~8x fewer SSD bytes/token than a
  deliberately-too-small `slots=2` run on the SAME model (8.06 MB/tok vs
  64.39 MB/tok) and decodes faster (47.3 vs 15.3 tok/s steady-state, a mechanism
  proxy on random weights). The real large-MoE (Mixtral) speedup from relying on
  auto residency stays Colab-only (the offline sandbox cannot download it).
  Async-overlap prefetch warming is NOT adapted to the chosen residency in this
  change and is explicitly DEFERRED with the async-overlap item (see
  `docs/ROADMAP.md`): the auto sizing already captures the win, and changing the
  synchronous warm path would risk the oracle guarantee for no measured gain.
- Legacy per-expert Mixtral GGUFs now PACK and STREAM. Real TheBloke Mixtral
  GGUFs name routed-expert FFN weights per expert
  (`blk.N.ffn_{gate,down,up}.E.weight`), not stacked
  (`blk.N.ffn_{gate,down,up}_exps.weight`). Confirmed against vendored llama.cpp
  b11379 that the runtime loader does NOT stack legacy tensors (that regroup is
  a Python-convert-time step), so the fix is PACKER-ONLY: `strata-pack` now
  recognizes the legacy per-expert names, GROUPS them per (layer, expert) into
  the stacked `.strata` expert region with the bytes copied VERBATIM (no
  requant, no reshape), and SYNTHESIZES the embedded GGUF metadata (fresh
  `gguf_init_empty` + all source KV + trunk tensor-infos + one synthesized
  stacked `_exps` 3-D tensor-info per (layer, kind)) so the engine sees only
  stacked names. `parse_expert_tensor_name` in `stream_buft.*`, the engine,
  `StrataReader`, `strata_format`, and the `mul_mat_id` addressing are
  UNCHANGED. Already-stacked inputs keep the verbatim metadata fast-path, so
  their behavior is unchanged. Oracle-gated by `test_legacy_per_expert_pack` on
  generated legacy fixtures (incl. a Q3_K variant mirroring the real Mixtral
  Q3_K_M) with a matched stacked-equivalent oracle: the legacy-sourced `.strata`
  decodes byte-identically (max|logit delta| == 0) to the libllama oracle run on
  the stacked model, with bounded expert streaming below `n_expert`.

### Correctness and honesty fixes

- Packer FAILS LOUDLY on a misclassified MoE instead of writing an OOM-bomb
  `.strata`. If the GGUF metadata says MoE (`llama.expert_count` > 0) but zero
  expert tensors classify, `strata-pack` now aborts with a diagnostic listing
  the expert-tensor names/shapes it found, rather than silently emitting a
  0-expert `.strata` that the engine then tries to hold whole-model-resident and
  gets SIGKILLed at decode. A Mixtral-faithful generated fixture + regression
  test guard it.
- Bounded-RAM reporting is now HONEST about what moves with `--expert-slots`.
  The quantities that are bounded and DO move with the slot budget are
  StrataFlow's own resident model weights (trunk only) and the SSD bytes
  streamed per token. Whole-process peak RSS (`getrusage` `ru_maxrss`) is a
  SEPARATE figure: it carries a FIXED llama/ggml backend + vocab process floor
  (~132 MiB; reproduced in-sandbox by a 3.9 MiB model and by a `--plan-only`
  run with no decode) plus a one-layer expert staging buffer sized to a layer's
  FULL expert count (not the slot budget) plus the ggml compute buffer. So peak
  RSS does NOT move with `--expert-slots` and can exceed on-disk for small
  models. The earlier "785 MiB -> 132 MiB, peak RSS flat vs model size" result
  measured the removal of the per-LAYER staging growth, NOT a slot-driven peak
  RSS bound; the record in `docs/ROADMAP.md` and `colab/README.md` is corrected
  accordingly. The CLI `stats:` line wording is unchanged (Python parser stable).
- Benchmark exit-criteria (`colab/strataflow_bench.py print_exit_criteria`) no
  longer presents a peak-RSS percentage of on-disk as the proof of "big model,
  small RAM" (it previously printed 128% / 113% as if that proved the mission).
  It now reports resident weights + streamed bytes per slot setting as the
  bounded quantities that move with `--expert-slots`, reports peak RSS
  separately with the fixed-floor + one-layer-staging caveat, explicitly flags
  rows where peak RSS >= on-disk, and shows the peak-RSS spread across the slots
  sweep so the slots-vs-peak invariance is visible.
- SSD-probe path fix: the API passed the model FILE path to `profile_hardware`,
  so the profiler tried to write `<file>/.strataflow_ssd_probe.tmp` and emitted
  a cosmetic `profiler: cannot write SSD probe` warning. It now derives the
  containing directory (`sf::containing_dir`, portable across POSIX `/` and
  Windows `\\`) and probes that; bandwidth still measures and the warning is
  gone.

### Benchmarking and instrumentation

- Benchmark harness `colab/strataflow_bench.py`: runs a config MATRIX (model
  size x `--expert-slots`) over the generated-F32 MoE, measures per run both
  end-to-end generate throughput (`gen tok/s` = max-tokens / full wall-clock)
  and steady-state decode throughput (`decode tok/s` = (max-tokens-1) /
  (wall-clock - TTFT), first token removed), plus TTFT, resident model weights,
  peak RSS, on-disk `.strata` size, SSD streamed MiB and an EXACT bytes/token
  (from the raw `streamed_bytes` uint64, not the rounded MiB display value),
  prints a fixed-width results table with an honest exit-criteria report, and
  writes
  machine-readable JSON + CSV artifacts (schema-versioned). The exit-criteria
  report now also contrasts the AUTO row (slots=0) against the smallest fixed
  slot count for the SAME model, stating that auto streams far fewer bytes/token
  (and typically decodes faster) because it caches the whole layer working set
  when RAM allows, kept honest as a mechanism proxy on random weights. The default matrix
  is CPU-only, offline, and free-Colab-tier sized (finishes in a few minutes);
  `--layers-list` / `--experts-list` / `--slots-list` / `--max-tokens` grow it.
  An optional `--real-model` row downloads one real TinyLlama (dense llama)
  through the same path and is cleanly SKIPPED (not a crash) when offline. Wired
  into Colab as Cells 9-10 in `colab/README.md`. The generated-F32 path proves
  the MECHANISM and the Phase 3 "big model, small RAM" bounded-RAM property;
  the absolute tok/s ladder in `PLAN.md` section 2 is for large real models on
  NVMe and the generated toy is a mechanism proxy only (stated, not fabricated).
- Two newly-instrumented metrics surfaced on the CLI `stats:` line and in
  `sf_runtime_stats`: TTFT (time-to-first-token, timed in the CLI around the
  first `on_token` callback) and SSD streamed bytes/token (counted at the
  `StrataReader` disk-read seam). The new `sf_runtime_stats` fields are APPENDED
  at the end of the struct for ABI backward-compatibility, and the existing
  `resident model weights = X MiB, peak RSS = Y MiB` wording is unchanged so the
  Python parser keeps working. The CLI additionally prints the exact
  `streamed_bytes = N` uint64 alongside the human-readable `streamed = W MiB`
  so the harness computes an exact (not display-rounded) bytes/token. The oracle
  correctness gate is preserved (the timing/counting seams add no engine math).

### Engine core (StrataFlow runs its own forward pass over ggml)

- K-quant families (Q4_K/Q6_K) validated against the libllama oracle through the
  engine and the `.strata` streaming path. The engine greedy token sequence is
  byte-identical to the oracle for both Q4_K and Q6_K, and K-quant experts stream
  through `.strata` with bounded `--expert-slots`. Added a C++ ggml-based
  test-fixture generator (`make_kquant_moe_gguf`, built on `ggml_quantize_chunk`
  because the Python `gguf` library cannot emit K-quants) and a guarded
  `test_engine_kquant_matches_oracle` sub-test
  (`STRATAFLOW_TEST_KQUANT_MOE_GGUF` / `STRATAFLOW_TEST_Q6K_MOE_GGUF`).
- Hold only the trunk resident on the `.strata` path; the full expert footprint
  is never allocated in RAM. Proven by test (trunk resident vs experts not
  resident). This is the core bounded-RAM, no-GPU capability. (#21)
- EC-7: second architecture - dense llama (plain gate/up/down FFN), oracle-gated. (#19)
- EC-6: predictor-driven expert prefetch - warm the next layer's likely experts
  into the cache ahead of use; correctness-neutral, measurable hit rate. (#18)
- EC-5: the engine is the default and only inference path; `llama_decode` is no
  longer on the inference path. libllama kept for the tokenizer and the test
  oracle only. (#16)
- EC-4: `.strata` single-file loader through our own loader (no
  `llama_model_load_from_file` on a `.strata`); byte-identical to the source
  GGUF. Closes the Task 5 blocker. (#15)
- EC-3: per-layer segmented execution with top-k expert residency via a bounded
  SlotPool; byte-identical to the fully-resident run. (#14)
- EC-2: engine-owned KV cache + multi-token greedy decode matching the oracle
  sequence. (#13)
- EC-1: StrataFlow's own ggml graph builder for the llama-arch MoE, single-token
  forward, matching the libllama oracle (argmax + logits within tolerance). (#12)
- Engine core design (Option D: own the forward pass over ggml) + feasibility
  spike proving a hand-built ggml graph reproduces llama's math. (#11)

### Tiered weight store and streaming format (Phase 3)

- `.strata` format design + per-expert residency feasibility spike. (#10, #4)
- Phase 3 Tasks 1-4: tiny MoE fixture + forced-streaming proxy + SSD profiling;
  `strata_stream_buft`; bounded SlotPool residency; native direct-I/O backends
  (io_uring / IOCP / F_NOCACHE with portable fallback). (#5, #8, #9)

### Placement and base engine (Phases 1-2)

- Phase 2: hardware profiler + placement planner that drive llama.cpp load
  params (auto `n_gpu_layers` / expert offload). (#3)
- Phase 1b: vendored llama.cpp + GgmlModel real-weights backend. (#2)
- Phase 1: engine scaffold - modules, build system, CLI, tests; cross-platform
  CI on Linux (GCC+Clang), macOS, Windows. (#1)

### Demo / docs

- Colab demo: added a real downloaded-model flow (`--real-model`) to
  `colab/strataflow_colab.py` and `colab/README.md`. It downloads a real
  quantized GGUF from HuggingFace (`hf_hub_download`), packs it to `.strata`,
  runs it bounded with `--expert-slots`, and records the decoded output text,
  measured tokens/sec, and peak RSS, with disk/RAM guards. The default is a
  small quantized dense-llama model (TinyLlama-1.1B-Chat Q4_K_M) because the
  smallest llama-arch MoE (Mixtral) is ~24 GB+; a Mixtral MoE run is exposed via
  flags. The generated-F32 demo stays the default no-download path. Corrected
  the stale "engine is F32-only / a downloaded quantized model will not run yet"
  claims in both files: quantized weights run through the type-agnostic staging
  path (Q8_0 oracle-validated, K-quant Q4_K/Q6_K validated by the
  generated-fixture gate); the remaining honest caveat is architecture coverage
  (llama-arch MoE + dense llama only). Demonstrated on Colab (sandbox offline).

### Project / licensing / docs

- Documentation: ROADMAP + CHANGELOG added; README and PLAN kept current with
  the working engine. (#17, #20, and this change)
- Relicensed to the Coaade Source-Available License, Version 1.0 (free for
  personal, non-commercial use; no reselling/rebranding/competing; no time-based
  conversion). Licensor identified as Coaade Inc., a Delaware C corporation.
  Vendored llama.cpp/ggml remains under its own MIT license. (#6, #7)

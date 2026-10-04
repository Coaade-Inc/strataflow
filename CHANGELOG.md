# Changelog

All notable changes to StrataFlow. This project has not yet cut a versioned
release; everything below is pre-release work on `main`. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what is done and what is open.

The format is loosely based on Keep a Changelog. Dates are omitted; entries are
grouped by the pull request that merged them.

## Unreleased

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

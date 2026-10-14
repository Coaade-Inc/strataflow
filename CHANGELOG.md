# Changelog

All notable changes to StrataFlow. This project has not yet cut a versioned
release; everything below is pre-release work on `main`. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what is done and what is open.

The format is loosely based on Keep a Changelog. Dates are omitted; entries are
grouped by the pull request that merged them.

## Unreleased

### Fixes

- Compress-what-streams: a FUSED gate_up expert bundle is now read and counted
  ONCE on the compressed path, not 3x/2x (`src/tws/strata_file.{h,cpp}`).
  Review follow-up on the compression lever. A fused layout
  (`gate_rel == up_rel == 0`, a two-slice `gate_up|down` bundle - the gpt-oss /
  granite / gemma / phi3-style arches `parse_expert_tensor_name` recognizes) is
  the only layout the compressed per-kind `read_blob` fallback ever served, and
  no in-tree fixture exercised it (the fixtures are legacy/non-fused). Two
  defects, both benchmark-counter / CPU-cost only (NOT a correctness or
  byte-identity break): (1) `read_blob` attributed the compressed size to any
  slice whose `slice_rel == 0`, and a fused bundle has gate AND up both at rel 0,
  so `streamed_bytes()` counted the compressed size TWICE per fused expert;
  (2) `read_bundle` rejected every fused layout, forcing the engine onto three
  per-kind `read_blob` calls that each re-read and re-decompressed the WHOLE blob
  (3x decode work per resident fused expert). THE FIX: `read_blob` now keys the
  compressed-byte increment on the GATE KIND (`kGate`/`kGateUp`) instead of
  `slice_rel == 0`, so it counts exactly once across the gate/up/down calls
  (non-fused is unchanged: `kGate` is the rel-0 slice there too); and
  `read_bundle` now runs its compressed branch BEFORE the verbatim fused
  rejection, serving a compressed fused bundle with ONE whole-blob decode into
  `dst` (the decompressed blob is the whole tightly-packed bundle, and a fused
  `gate_up` slice is gate followed by up contiguously, so `[gate_up | down]` is
  byte-identical to `[gate | up | down]`) - read and decoded once, counted once.
  The verbatim path still rejects fused. Preserves byte-identity (the engine
  sees the same uncompressed bytes), the default/uncompressed byte-for-byte
  identity (SHA256 `strata-pack` == `--compress none`), v1/v2/v3 back-compat, and
  the per-call-local no-shared-state decode (compute thread + async prefetch
  worker stay race-free). A new fused-layout seam subtest in
  `test_compressed_strata` (hand-built two-slice `gate_up|down` v3 `.strata`)
  asserts `read_bundle` serves the compressed fused bundle in ONE read/decode
  and that the three `read_blob` calls count the compressed size EXACTLY ONCE
  (would catch the 2x regression). Full ctest 24/24 green under GCC 11.5 + Clang
  15.0.7 `-Werror` (release + debug); the real-path oracle gate stays
  `max|delta| = 0.000e+00`. No change to `third_party/llama.cpp`.

- MAX now CLAMPS to the largest fitting plan instead of REFUSING on a 12-13
  GiB-class box (`src/plan/planner.cpp`). RamMode::Max (and Auto) are MODES, not
  numeric ceilings: a pure mode run (`target_ram_budget == 0` AND
  `ram_budget == 0`) now ALWAYS converges on the largest plan that FITS physical
  RAM and never refuses while the minimal config fits. The regression: on a
  Mixtral geometry whose full-`n_ctx` KV (8 GiB) + trunk + maximal experts
  overflow physical RAM, MAX kept the FULL KV window (the step-down did not
  engage because full KV + the minimal expert floor fit under MAX's larger
  all-physical-RAM usable), then the MAX>=AUTO monotonicity clamp forced the
  resident expert count up to the reference AUTO plan's (AUTO had windowed its KV
  DOWN and bought MORE experts than MAX can hold at full KV) - overflowing usable
  so the fit invariant set `fits=false`, the C ABI returned `SF_ERR_RUNTIME`, and
  the CLI exited 1. THE FIX (one coherent clamp-not-refuse mechanism): when
  holding at least AUTO's resident experts does not fit at MAX's current (larger)
  KV window, MAX steps its KV window DOWN to AT LEAST AUTO's (bounded by the
  smallest safe window) and adopts at least AUTO's trunk pool, then sizes the
  resident expert count from the remainder (never below AUTO's). Because AUTO
  already fits inside `auto.usable <= max.usable` with its windowed lever set,
  adopting at least AUTO's levers always lands a fitting MAX plan whenever AUTO
  fits. A general MODE-path backstop windows the KV, streams the trunk, and caps
  experts from the remainder for any mode path whose natural plan would overflow,
  so Auto never refuses either. The LOUD refusal (`fits=false` + the honest
  `refusal_reason` naming the floor) is now reachable ONLY for an EXPLICIT
  numeric ceiling (`target_ram_budget` or `ram_budget > 0`) pinned below the
  achievable floor; that wording/floor computation is UNCHANGED. Preserves the
  MAX>=AUTO monotonicity (`max.resident >= auto.resident`, `max KV window >=
  auto`, `max trunk pool >= auto`), the fit invariant (`planned_peak_bytes <=
  usable_ram_bytes` for every returned fitting plan), and byte-identity to the
  libllama oracle (residency/placement/sizing only - never which bytes compute or
  the graph order). On the synthetic 13 GiB repro (full KV 8 GiB, 32 layers x 8
  experts ~72 MiB, 0.5 GiB trunk): BEFORE MAX refused (natural peak 14630 MiB vs
  usable 13312 MiB, floor 2510 MiB) while AUTO fit (peak 10343 MiB, 57 slots,
  `kv_window=16384`); AFTER MAX fits (peak 10518 MiB `<=` 13312 MiB, 57 slots,
  `kv_window=16384`, trunk fully resident), monotonic with AUTO on every lever.
  Full matrix green: GCC 11.5 + Clang 15 `-Werror`, release + debug ctest 22/22
  (incl. the two new regression tests `test_max_clamps_not_refuses_12gb` and
  `test_max_clamps_not_refuses_real_path`), and the gated real-load path
  (`test_target_ram_modes_real_path`) byte-identical across AUTO/TARGET/MAX
  (`max|delta| = 0.000e+00`, oracle `5.179e-02 < 1e-1`). The planner is a pure
  function with no threads, so there is no TSan concern for this change. The real
  ~19 GB Mixtral MAX run on the 12 GiB Colab box is Colab-verified separately;
  the fix and the geometry regression are proven end-to-end in-sandbox. No change
  to `third_party/llama.cpp`; no new feature.

### Features

- Selectable GPU compute backend foundation (CPU default, byte-identical).
  The engine's ggml compute backend is now SELECTABLE instead of a hard-coded
  `ggml_backend_cpu_init()`. A new top-level `STRATAFLOW_GPU=off|cuda|metal|vulkan`
  CMake option (default `off`) wires the vendored ggml `GGML_CUDA`/`GGML_METAL`/
  `GGML_VULKAN` backends from the TOP-LEVEL CMake only (`third_party/llama.cpp`
  is never edited; its options are merely toggled before `add_subdirectory`), and
  defines compile-time guard macros (`STRATAFLOW_GPU_ENABLED` plus
  `STRATAFLOW_GPU_<CUDA|METAL|VULKAN>`) on the single engine TU. At runtime the
  engine selects the first available GPU ggml backend via the backend-agnostic
  ggml-backend DEVICE/REGISTRY API (`ggml_backend_dev_count`/`_get`/`_type`/
  `_init`, the SAME API `enumerate_gpus` uses - not a CUDA-specific call) when a
  GPU backend is compiled in AND a device is present, and ALWAYS falls back to
  the CPU backend otherwise; a missing or failed GPU never fails the load. When
  a GPU backend is selected, the gallocr and every resident buffer (trunk
  `wbuf`, KV `kvbuf`, expert staging `estbuf`) are allocated from the selected
  backend's default buffer type, and the per-token expert "fault bytes into the
  resident staging region" step becomes a host->device upload
  (`ggml_backend_tensor_set`) through a single new host-or-device `stage_write`
  primitive instead of a host `memcpy`. The planner's VRAM budget and the C ABI
  `sf_context_params.preferred_backend` are plumbed through `load_model` ->
  `load_ggml_model`/`load_strata_model` -> `GgmlModel` -> `Engine::load` via a
  backward-compatible defaulted trailing parameter, so existing callers/tests
  compile unchanged. HARD CPU-ONLY INVARIANT (the only part CI/the sandbox can
  validate): on the default `STRATAFLOW_GPU=off` build the macro is 0, every GPU
  branch is compiled out, the engine selects CPU, `ggml_backend_cpu_set_n_threads`
  is still called only for the CPU backend, and decode is BYTE-IDENTICAL to the
  prior engine and the oracle (`max|delta| = 0.000e+00`). The CPU path stays
  single-threaded and TSan-clean (`stage_write` is a pure branch on a bool, no
  new threads). FIRST-INCREMENT SCOPE: when a GPU is selected the WHOLE resident
  set runs in VRAM and the streamed trunk is forced resident
  (streamed-trunk-on-GPU is a documented follow-on); the per-token EXPERT
  upload seam is device-correct regardless, as it is the per-token crux. A new
  `test_backend_selection` unit test proves the CPU-selection invariant and the
  byte-identical-to-oracle decode on a CPU-only build. GPU-accelerated decode is
  NOT built or run-validated in the sandbox (no GPU/toolkit); it is gated behind
  the compile-time guard for hardware validation. Full ctest 25/25 green under
  GCC + Clang `-Werror`. No change to `third_party/llama.cpp`.

- Optional lossless on-disk expert-blob compression (compress what streams).
  `strata-pack` gains a `--compress <lz|none>` flag (default `none`,
  uncompressed): when `lz` is selected, each per-expert blob is compressed at
  pack time with a new self-contained in-tree LZ codec
  (`src/common/lz_codec.{h,cpp}`, no new submodule, no third-party dependency),
  storing the SMALLER of {compressed, verbatim} per blob with a per-blob codec
  id plus compressed/uncompressed lengths in a new `.strata` region. The reader
  (`StrataReader::read_bundle` / `read_blob`) decompresses TRANSPARENTLY at the
  read seam into per-call scratch, so the engine's compute sees byte-identical
  expert bytes while FEWER (compressed) bytes stream from the SSD;
  `streamed_bytes` now reflects the compressed bytes actually read (the win).
  The `.strata` format is bumped to v3, OPTIONAL and BACK-COMPATIBLE: v1 and v2
  files still load and decode byte-identically, and an uncompressed v3 pack is
  byte-for-byte identical to the prior packer output (SHA256-verified).
  `sizeof(StrataSuperblock)` stays 128 and `ExpertIndexEntry` stays 48 bytes
  (v1/v2 readers depend on both). The codec functions are PURE (per-call scratch
  only), so the concurrent compute-thread and prefetch-worker decompress reads
  are race-free; no engine change was needed. ORACLE GATE: the decode on a
  compressed `.strata` is byte-identical to the uncompressed one (per-step logit
  `max|delta| = 0.000e+00`) and matches the libllama oracle within the quant
  tolerance (`~5.2e-2 < 1e-1`) - it is a pure lossless transport optimization,
  not a lossy requant. DECOMPRESS-VS-SSD TRADEOFF (measured, safe): codec
  decompress is ~6.9 GiB/s (gcc release; clang ~6.1 GiB/s) on a 32 MiB mixed
  buffer, well above the ~4 GiB/s SSD per-core bandwidth, so decompress is not
  the new bottleneck. HONEST RATIO: on the already-K-quantized expert fixture
  (`q4_k`) the ratio is ~1.0000 - the already-quantized, high-entropy
  random-weight experts do not shrink and every blob falls back to verbatim; the
  idea is attributed to AirLLM, whose higher-entropy / differently-quantized
  models compressed more, so we do NOT claim their ~3x, and the net wall-clock
  effect scales with whatever ratio the real trained-weight Mixtral achieves
  (Colab-only to measure). The mechanism is proven end to end on a deliberately
  compressible blob (16384-byte blob stored as 224 bytes on disk, `read_bundle`
  returns the full 16384 bytes byte-exact, `streamed_bytes` counts 224). New
  tests `test_lz_codec` (round-trip on empty / small / compressible /
  high-entropy / blob-sized inputs, rejects truncated/garbage with no overrun,
  prints a measured decompress GiB/s) and `test_compressed_strata` (real-path
  pack->StrataReader->engine oracle gate; skips cleanly without the fixture).
  Full ctest 24/24 green under GCC 11.5 + Clang 15.0.7 `-Werror` (release +
  debug). No change to `third_party/llama.cpp`.

- AUTO coherent retune + lossless batched/sequential expert reads +
  reused-expert residency. Fixes the two coupled problems the real-Mixtral Colab
  run exposed, both residency/placement-only so decode stays byte-identical.
  (1) COHERENT AUTO POLICY (`src/plan/planner.cpp`): AUTO now takes a
  comfortable, DOCUMENTED share of total RAM (the conservative `usable_ram`
  good-neighbor budget that leaves headroom for other apps) and co-selects the
  three fit levers (KV window, trunk pool, expert cache) CONSISTENTLY to fill
  that share - more resident means less streaming. The KV-window step-down now
  budgets against the FULL resident trunk, so AUTO windows the KV BEFORE the
  trunk is forced to hard-stream, enforcing the COHERENCE INVARIANT (asserted in
  code and in three tests): AUTO never keeps the full-`n_ctx` KV resident while
  simultaneously hard-streaming the trunk (if `0 < trunk_pool_layers < n_layers`
  then `0 < kv_window < n_ctx`); the levers co-reduce. Before, AUTO on a
  12 GiB-class box was INCOHERENT (it kept full KV while hard-streaming the trunk
  and cached only 2 experts); now, when RAM affords it, it keeps the trunk fully
  resident and caches far more. On the synthetic 12 GiB Mixtral shape: BEFORE
  incoherent (windowed KV AND hard trunk streaming pool=20/32 AND only 4 experts,
  peak ~= usable), with MAX fits=0 (projected peak 14097 MiB `>` usable
  12288 MiB because the clamp-to-Auto raised full KV over budget); AFTER coherent
  (AUTO `kv_window=8192`, trunk fully resident pool=0, 14 experts resident), with
  MAX fits=1 (peak 12249 `<=` 12288, `kv_window=16384`, 18 experts) and MAX stays
  `>= AUTO` on every lever (usable, KV window, resident experts, trunk
  residency). Experts resident went from 2 (old) to 14 (synthetic 12 GiB) /
  104 (mxg real fixture). (2) LOSSLESS BATCHED/SEQUENTIAL EXPERT I/O
  (`src/tws/strata_file.h/.cpp`, `src/model/engine/engine.cpp`): a new
  `StrataReader::read_bundle` issues ONE `read_at` over a whole `(layer, expert)`
  bundle `[blob_offset, blob_offset + blob_length)` and splits it into the
  gate/up/down slices by their known relative offsets, replacing the previous
  THREE per-kind `read_blob` calls. It is byte-identical (the returned bytes
  `memcmp` equal to the three per-kind reads concatenated, the slices are tightly
  packed gate|up|down inside the bundle, and `streamed_bytes()` totals are
  unchanged - the same bytes counted once), and a new `read_calls()` atomic
  counter proves the underlying read-count drops 3:1 per bundle. `read_bundle`
  counts a coalesced read toward `streamed_bytes()` ALL-OR-NOTHING: only a
  FULL-length read (`n == blob_length`) is counted, so a short/partial read -
  which the engine treats as a failure and retries via the three per-kind
  `read_blob` calls (each counting its own slice) - is not double-counted; the
  byte total stays accurate on both the success and the fallback path. The three engine
  expert-read sites (`ensure_layer_experts_resident`, `warm_expert`,
  `pf_read_bundle`) route through `read_bundle` with a per-kind fallback and keep
  loading EXACTLY the router-selected top-k experts, so the resident bytes - and
  the decode - stay byte-identical, and the coalesced read is TSan-safe (const +
  atomic counters) on the compute thread and the async prefetch path. (3) The
  larger coherent AUTO cache (and the predictor's per-layer activation frequency
  keeping reused experts resident) directly cuts re-streaming: on the mxg real
  fixture streamed bytes/token drop from 29,408,625 B (2-slot baseline) to
  3,008,102 B (104-slot AUTO), ~9.8x fewer expert bytes re-streamed per token.
  The predictor-driven residency path is the existing wiring in
  `Engine::forward` (`SlotPool::set_layer_weights(ExpertPredictor::layer_activity())`
  every few tokens, so busier layers keep more of their reused experts resident
  within the mode's budget); it is now proven END TO END by
  `test_predictor_driven_residency` in `tests/test_slot_pool.cpp` (observed
  routing -> `layer_activity()` -> `set_layer_weights` -> the busiest layer wins
  the scarce spare slot and keeps strictly more of its reused experts resident
  across a sweep), with the top-k floor guaranteeing decode stays byte-identical.
  Greedy decode is BIT-IDENTICAL across AUTO/TARGET/MAX (`max|delta| =
  0.000e+00`) and within the quant bound vs the libllama oracle (`5.179e-02 <
  1e-1` on the random-weight q4_k fixture). Full matrix green: GCC 11.5 + Clang
  15 `-Werror`, release + clang ctest 22/22 (+`test_coalesced_expert_read`), the
  fixture-blindness real-load gate (`test_target_ram_modes_real_path`), and TSan
  22/22. CURRENT TOKEN SPEED (the user's question): the real Colab
  Mixtral-8x7B-Instruct Q3_K_M run is ~0.06 tok/s (~16 s/token) on the 12 GiB /
  2-core / ~4 GiB/s SSD / no-GPU box, I/O-bound - ~74-78 GB streamed per 16
  tokens (~4.6 GB/token) because the OLD resident pool held only 2 bundles so
  almost every expert re-streamed each token. This change attacks both the
  bytes/token term (the bigger coherent AUTO cache) and the effective-bandwidth
  term (fewer, larger sequential reads), but the box stays fundamentally
  I/O-bound on 2 cores (no compute/IO overlap; the async prefetch self-backs-off
  at `completed-before-use = 0`), so NO tok/s number is promised - the absolute
  real-Mixtral improvement is a Colab/hardware re-measurement; the sandbox proves
  the MECHANISM (coherent plan geometry, ~9.8x bytes/token reduction on the
  fixture, 3:1 read-count cut, byte-identity). The exact three-mode re-verify
  commands and expected per-mode behavior are in `colab/README.md`;
  `docs/ROADMAP.md` records the lever as done and names the remaining speed work
  (co-activation packing layout, then speculative/draft decoding, technique 6.7
  second half and 6.8). See `docs/RESEARCH_50B_ON_A_LAPTOP.md` section 6.7.
- Target-RAM capability: one dial for the fit-vs-speed tradeoff, with two modes
  {auto, max} plus an explicit numeric target. A single target-RAM concept now
  drives ALL the fit levers together (KV window, expert slots/cache, adaptive
  per-layer trunk streaming, and the VRAM placement the planner already budgets)
  from one decision, on BOTH the CLI and the C ABI. CLI: `--target-ram-gb X`
  (explicit RAM ceiling in GiB), `--ram-mode auto|max`, and `--max-performance`
  (alias for `--ram-mode max`); the existing `--vram-gb` / `--ram-gb` /
  `--cache-gb` / `--expert-slots` and `STRATAFLOW_ENGINE_KV_WINDOW` stay as
  lower-level manual overrides. ABI: two fields appended at the END of
  `sf_context_params` (append-only, backward-compatible, same discipline as
  `expert_slots` / `cache_budget` / `async_prefetch`): `uint64_t
  target_ram_budget` (bytes, 0 = use the mode) and `uint32_t ram_mode` taking a
  new `typedef enum sf_ram_mode { SF_RAM_MODE_AUTO = 0, SF_RAM_MODE_MAX = 1 }`
  (default-zero == auto). Precedence, highest wins: explicit low-level manual
  knobs (`STRATAFLOW_ENGINE_KV_WINDOW` / `--expert-slots` / `--cache-gb`) >
  `--target-ram-gb` (`target_ram_budget`) > `--ram-mode max` > `--ram-mode auto`
  (the DEFAULT, `ram_mode == 0`). An explicit target is used directly as the
  usable budget and drives the KV-window step-down + trunk streaming so the
  projected peak fits even below the trunk-resident floor; a target below the
  achievable floor is refused LOUDLY (`plan.fits=false` + a reason naming the
  floor, surfaced as `SF_ERR_RUNTIME`). Auto is the good-neighbor default: it
  leaves headroom for other apps by taking only the conservative `usable_ram`
  share (`free - max(1 GiB, 12.5%)`, capped at 85% of total). Max dedicates the
  machine: all physical RAM (plus the full VRAM budget), full KV window, maximal
  expert cache, trunk fully resident, minimal streaming. (Max sizes its ceiling
  from `ram_total_bytes`, so on the same machine it holds at least as many
  experts resident as auto and streams no harder - `max >= auto` resident and
  `max.streamed <= auto.streamed` hold monotonically. See the "Target and max
  RAM-mode semantics fixed" entry under Correctness and honesty fixes for the
  inversion this corrected.) GPU honesty: auto/max
  MAY read VRAM and the planner BUDGETS VRAM placement
  (`plan.trunk_layers_in_vram` / `plan.expert_slots_vram` ->
  `derive_llama_placement` -> `n_gpu_layers`), but that only affects the
  non-authoritative plain-GGUF libllama sidecar; the authoritative `.strata`
  engine decode is CPU-only (`n_gpu_layers = 0` everywhere in `engine.cpp`), so
  GPU PLACEMENT is budgeted while full GPU-accelerated COMPUTE is the enterprise
  roadmap track, not implemented. Residency/placement only, so decode stays
  byte-identical across all three modes and to the libllama oracle. This turns
  the user's real Colab three-knob experiment
  (`STRATAFLOW_ENGINE_KV_WINDOW=2048 --expert-slots 8 --cache-gb 1`, which
  brought the real Mixtral-8x7B-Instruct Q3_K_M `.strata` to ~2.4 GiB peak RSS
  byte-identically) into a single `--target-ram-gb 2`. Sandbox-proven through the
  REAL load path on a Mixtral-geometry fixture (`n_ctx` 32768, 32 layers, pack ->
  `.strata` -> `load_model` -> planner -> engine): an explicit ~1.2 GiB target
  fits at projected peak ~1228 MiB (`kv_window=512`, `trunk_pool_layers=4`,
  `slots=5`); auto projects peak ~3647 MiB (usable ~9520 MiB) and max ~3756 MiB
  (usable ~10880 MiB) on the ~12 GiB profile with full KV + trunk resident +
  `slots=256`; the invariants `max.usable > auto.usable` and
  `auto.peak < max.peak` hold, and the greedy decode is byte-identical
  (`max|delta| = 0.000e+00`) across all three modes and within the soft oracle
  bound. Colab-only: the absolute real-Mixtral per-mode peak RSS is the user's
  Colab re-run; the exact three-mode commands and expected plan/stats are in
  `colab/README.md`. `docs/ROADMAP.md` records the capability as done.
- Adaptive per-layer trunk streaming engages BY DEFAULT on the real `.strata`
  load path (FEAT-003 Part B, technique 6.4 from
  `docs/RESEARCH_50B_ON_A_LAPTOP.md`). After the auto KV window, the dominant
  remaining resident cost is the per-layer attention trunk
  (`blk.N.attn_{q,k,v,output}.weight`, ~5.5 GB across 32 layers on real
  Mixtral). Those four projections per layer now stream from the `.strata` v2
  trunk index into a BOUNDED HOT-TRUNK POOL (a ring of `trunk_pool_layers`
  layers) fetched synchronously immediately before each layer's attention
  segment (the same resident-before-op discipline as the edge tensors and the
  experts); the tiny per-layer norms + router stay HOT (resident). The
  aggressiveness is BUDGET-DRIVEN: `plan_placement` derives `trunk_pool_layers`
  from the SAME conservative usable-RAM budget (`ModelShape::attn_trunk_bytes`,
  summed from the trunk index), so a big-RAM box keeps the whole trunk resident
  (pool 0, no per-token trunk traffic) and a tight 12 GB-class box streams it
  hard (pool as low as 1 layer). The decision is a DIRECT `trunk_pool_layers`
  parameter to `engine::Engine::load` (default 0 = whole trunk resident,
  preserving every existing caller), plumbed from `PlacementPlan::trunk_pool_
  layers` through `GgmlModel` - not an env side-channel - so the bounded pool
  applies by default; the plan summary gains a `trunk stream[hot=N layers]`
  segment. `Engine::resident_weight_bytes()` reflects the bounded hot pool and
  the new `Engine::streamed_trunk_bytes()` / `StrataReader::streamed_trunk_
  bytes()` count the per-layer trunk reads DISTINCTLY from the expert
  `streamed_bytes()` (the bench per-token expert number is unchanged). HONEST
  TRADEOFF: these weights are touched every token, so streaming adds per-token
  SSD traffic and is slower on a slow box - accepted because the FIRST goal is
  to FIT small. Residency / byte-provenance only: same bytes, same ops, same
  graph order, single compute thread (the async I/O worker is untouched), so
  decode stays byte-identical to the libllama oracle; the authoritative
  resident-before-op read is the correctness backstop and
  `STRATAFLOW_ENGINE_NO_TRUNK_STREAM=1` forces the whole trunk resident.
  Sandbox-proven on an attention-trunk-dominated Mixtral-mirroring fixture
  through the REAL load path (packer -> `.strata` -> `load_strata_model` ->
  planner -> engine): the plan chose `trunk_pool_layers=5` of 10 layers by
  default under a ~1.27 GiB budget, `resident_weight_bytes` dropped (default CLI
  run on an `n_embd=4096 n_layer=16` fixture at `--ram-gb 1.3`: resident model
  weights 292.9 -> 110.9 MiB, peak RSS 329.5 -> 161.5 MiB), and the streamed
  decode was bit-for-bit identical (`max|delta| = 0.000e+00`) to the
  fully-resident decode on the SAME v2 file and within the oracle quant
  tolerance over a short AND a longer sequence
  (`tests/test_trunk_stream_adaptive.cpp`). Colab-only: the absolute
  real-Mixtral peak-RSS drop is hardware-only (the offline sandbox cannot
  download Mixtral); the re-verify command is in `colab/README.md`.
- Auto KV window ENGAGES BY DEFAULT on the real `.strata` Mixtral load path
  (FEAT-002 Part A, technique 6.5 from `docs/RESEARCH_50B_ON_A_LAPTOP.md`). A
  DEFAULT run (no `STRATAFLOW_ENGINE_KV_WINDOW`, no flags) now reserves a
  windowed KV when the full-`n_ctx` KV would overflow the conservative budget, so
  the engine's measured `resident_kv` drops from the full-`n_ctx` figure
  (8192 MiB on Mixtral) to the windowed size and the plan line gains a
  `kv window=...` segment - fixing the exact Colab symptom (resident_kv at full
  `n_ctx`, no window). Two breaks fixed: (1) the real synthesized-metadata path
  read `context_length`/`head_count`/`head_count_kv` with `gguf_get_val_u32`,
  which aborts or returns 0 when the source stored them under a different GGUF
  integer encoding (the `n_ctx_train(0)` warning is the tell), so the planner saw
  zero geometry and never windowed while the engine reserved the full KV; a new
  robust type-dispatching read (`src/common/gguf_meta.h` `gguf_read_u32`, which
  switches on `gguf_get_kv_type` across u8/i8/u16/i16/u32/i32/u64/i64) is now
  shared by the planner (`src/model/ggml_model.cpp`), the engine
  (`src/model/engine/engine.cpp`), and the packer (`src/tws/strata_pack.cpp`) so
  all three see IDENTICAL geometry on every path. (2) the plan's chosen window
  reached the engine ONLY through the process-global `STRATAFLOW_ENGINE_KV_WINDOW`
  env side-channel; it is now a DIRECT `kv_window` parameter to
  `engine::Engine::load` (default 0 = full/auto, preserving every existing
  caller), plumbed from `PlacementPlan::kv_window` through `GgmlModel`. Engine
  resolution order: explicit `STRATAFLOW_ENGINE_KV_WINDOW` env (user/test) WINS,
  else the plan-provided window, else full `n_ctx`. Residency/window sizing only;
  forward math and graph order unchanged, so decode stays byte-identical.
  Sandbox-proven on a Mixtral-geometry legacy-per-expert fixture (`n_ctx` 32768,
  `head_dim` 128, GQA `n_head_kv` 2, 32 layers, full KV 2048 MiB) packed through
  the REAL `strata-pack` -> `.strata` -> `load_strata_model` -> planner -> engine
  under a simulated 12 GiB budget with NO env override
  (`tests/test_kv_window_real_path.cpp`): geometry reaches the planner non-zero,
  the plan chose `kv_window=16384` (`< n_ctx`), `resident_kv_bytes()` dropped
  2147483648 -> 1073741824, and the plan-window greedy decode is BIT-IDENTICAL to
  the full-resident run (`max|delta| = 0.000e+00`) and matches the libllama
  oracle on the matched stacked fixture. The absolute real-Mixtral peak-RSS drop
  is Colab-only (the offline sandbox cannot download the model); the re-verify
  command and expected plan/stats are in `colab/README.md`.
- Lossless KV paging (technique 6.5 from `docs/RESEARCH_50B_ON_A_LAPTOP.md`, the
  lossless half). The engine-owned per-layer K/V cache no longer reserves the
  full trained `n_ctx` up front: it is bounded to a resident recent WINDOW `W`
  (the per-layer `kvctx` tensors become `[head_dim, n_head_kv, W]`, a ring buffer
  keyed by `pos % W`), and positions that scroll out of the window are tiered to
  a RAM-backed per-layer COLD page store (keyed by absolute position, holding the
  EXACT F32 bytes the ring held). The cold store is NOT pre-allocated at `n_ctx`:
  it is EMPTY at load and GROWS ON DEMAND, bounded to the positions actually
  evicted (never the full `n_ctx`), so a large-`n_ctx` model does not pay a
  multi-GB cold allocation up front (see the real-model empty-output fix below).
  Attention re-assembles the full history
  `0..pos` - cold columns from the store plus the resident ring columns in
  absolute position order (handling ring wrap) - and runs the IDENTICAL
  permute/`mul_mat`/`soft_max_ext`/`mul_mat` op sequence over it, so decode is
  BYTE-IDENTICAL to the full-resident path and the libllama oracle; no past
  position is dropped (this is deliberately NOT lossy StreamingLLM/SnapKV
  eviction). The window `W` defaults to the full `n_ctx` (so the DEFAULT path is
  byte-identical and zero-overhead) and is overridable via the
  `STRATAFLOW_ENGINE_KV_WINDOW` env knob (same `std::getenv` pattern as the other
  `STRATAFLOW_ENGINE_*` knobs) to engage paging. `KvStore` (`src/kv/kv_store.h`)
  owns the paging bookkeeping (window, ring mapping, cold/resident split); the
  ggml-touching copies live in `engine.cpp`; the cold store is populated and read
  on the single compute thread only, so no new cross-thread state is added (TSan
  clean) and it composes with the streamed trunk, the bounded expert `SlotPool`,
  and async prefetch. New `Engine::resident_kv_bytes()` / `full_kv_bytes()`
  accessors, plus a `resident_kv_bytes` field APPENDED (ABI-safe) to
  `sf_runtime_stats` and wired through `sf_session_stats` and the CLI, measure the
  reserved-KV reduction. Sandbox-proven byte-identical and resident-KV-lower on
  the generated fixtures (`tests/test_kv_paging.cpp`): with
  `STRATAFLOW_ENGINE_KV_WINDOW=4` over an 8-token generation (sequence longer than
  the window, so cold KV is paged out and fetched back; cold-fetch counter 462),
  resident KV drops 32,768 -> 2,048 B on the F32 tiny MoE (`n_ctx` 64) and
  1,048,576 -> 16,384 B on the K-quant MoE (`n_ctx` 256), with the paged decode
  bit-for-bit identical to the full-resident engine (`max|delta| = 0.000e+00`)
  and matching the oracle (F32 within 1e-3; Q4_K seq-identical). SSD spill of cold
  pages (option b) and KV quantization (approximate, Stage 4) are deferred.
- Stream the trunk: lossless streaming of the large edge tensors (technique 6.4
  from `docs/RESEARCH_50B_ON_A_LAPTOP.md`). The two biggest trunk tensors, each
  touched exactly once per token at the input/output edges - `token_embd.weight`
  (input embedding) and `output.weight` (lm_head) - now STREAM from the `.strata`
  on demand instead of sitting fully resident in RAM, while the HOT per-layer
  attention/norm/router trunk tensors stay resident (streaming those would be a
  hot-path throughput regression). At 50B scale these vocab x n_embd edge tensors
  dominate the fixed trunk cost, so moving them off the resident floor is the
  dominant FIT lever. The change is LOSSLESS - same bytes, same ops, only byte
  PROVENANCE differs. The `.strata` format is bumped to a BACK-COMPATIBLE v2 that
  adds an optional per-trunk-tensor index (fixed-size records of {tensor name,
  trunk-relative offset, length}, carved from the superblock's reserved pad so
  `sizeof(StrataSuperblock)` stays 128); the trunk block bytes are byte-for-byte
  identical to v1, and a v1 `.strata` still opens with no trunk index and falls
  back to the fully-resident trunk. On a v2 file the two edge tensors are created
  metadata-only and share ONE stable backing buffer, fetched synchronously on the
  compute thread immediately before the segment that reads them (the same
  "resident before the op is built" discipline as the expert path; the async
  prefetch worker and the `SlotPool` are not involved). Sandbox-proven on the
  generated fixtures, byte-identical and resident-bytes-lower: `resident_weight_bytes`
  drops 103,040 -> 69,248 B (-33,792 B) on the F32 tiny MoE and 392,448 ->
  354,432 B (-38,016 B) on the K-quant MoE, with the trunk-streaming decode
  bit-for-bit identical to the fully-resident decode (`max|delta| = 0.000e+00`,
  proven on the SAME v2 file via the test-only `STRATAFLOW_ENGINE_NO_TRUNK_STREAM`
  knob) and to the libllama oracle, on BOTH an F32 and a K-quant fixture; a v1
  back-compat gate proves an old `.strata` (trunk index count 0) still decodes
  byte-identically to the oracle. The real Mixtral/50B peak-RSS reduction (where
  the multi-GB resident embeddings + lm_head previously blocked the fit) is
  hardware-only; see `colab/README.md` for the exact re-verify command.
- Layer-stratified residency + prefetch gating (fewer SSD bytes streamed per
  token; async prefetch no longer hurts). Two composing levers on the
  bounded-RAM streaming path, both oracle byte-identical and adaptive to model
  shape with NO machine-specific constants. LEVER B: the single GLOBAL plain-LRU
  `SlotPool` is replaced by LAYER-STRATIFIED, frequency-aware per-layer
  residency. Each layer gets its own LRU scope and a shape-derived capacity
  (`base = n_slots / n_layer`, floored at a single layer's top-k, spare slots
  distributed by observed per-layer routing frequency from the predictor), so a
  layer's hot experts survive across tokens instead of being evicted by the
  intervening layers' per-token sweep. The TOTAL slot budget the planner chose
  is UNCHANGED (peak RSS stays bounded by `n_slots * slot_bytes`); only how the
  slots are partitioned and evicted changes. The fully-resident fast path
  (`n_slots >= n_layer * n_expert`) is preserved exactly (no eviction). LEVER A:
  async prefetch is GATED so it can never increase bytes streamed - a completed
  speculative read is installed ONLY into a genuinely FREE per-layer slot
  (`SlotPool::has_free_slot_in_layer`), so a prefetch never evicts a
  still-needed authoritative resident; and the worker SELF-BACKS-OFF (quiesces,
  stops issuing speculative reads) once the observed `completed-before-use` rate
  stays ~0 over a shape-derived window. The authoritative
  `ensure_layer_experts_resident` is UNCHANGED (it still acquires exactly the
  router-selected experts before the matmul, so byte provenance / graph order /
  decoded output are identical); the ggml compute path stays single-threaded
  (one background I/O thread that never mutates the pool); the default-ON async
  toggle and the `streamed_bytes` + `prefetch_warmed/used/completed_before_use`
  stats are intact. Sandbox-proven on generated multi-layer fixtures: at the
  shape-derived tight budget (`bench_moe-L8-E16`, slots=32 = `n_layer *
  n_expert_used`, async off) layer-stratified residency streams 327,155,712 vs
  the old global LRU's 522,190,848 bytes (37.3% fewer, 35.8% fewer misses) at
  identical 132.4 MiB peak RSS, byte-identical decode; and at a bounded slots=24,
  async-on streams 46,792,704 bytes/tok vs async-off 61,341,696 (0.76x, async-on
  <= async-off, byte-identical decoded text), reversing the earlier bounded-slot
  regression where async-on streamed ~1.48x MORE than async-off. The
  streamed-bytes/token reduction, byte-identity, and prefetch-no-longer-hurts
  invariant are SANDBOX-PROVEN; the absolute wall-clock tok/s on the real ~19 GB
  Mixtral stays COLAB-ONLY (`completed-before-use` stays ~0 on fast sandbox CPUs,
  mirroring 2-core Colab, so the overlap wall-clock win is hardware-dependent -
  per the project steer, no tok/s target is promised). `colab/strataflow_bench.py`
  surfaces the async on-vs-off `<=` relationship directly, and
  `colab/README.md` carries the exact real-Mixtral before/after re-verify
  commands (RUN A/RUN B auto vs small fixed residency; RUN C/RUN D async on vs
  off; run on current `main` before and this branch after and compare streamed
  MiB / bytes-per-token). `docs/ROADMAP.md` records the lever as done.
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
  done. The Windows buffered fallback of `BlockFile::read_at` is now positional
  and stateless like POSIX `pread`: it opens a buffered Win32 `HANDLE` and reads
  via `ReadFile` with a per-call `OVERLAPPED` offset instead of a shared
  `_fseeki64`+`fread` guarded by a mutex, so concurrent compute and prefetch
  reads no longer serialize through one stateful handle (fixes the Windows-only
  CI hang; reads stay byte-identical). A new unconditional concurrent-read
  stress test in `test_async_io` reads one `BlockFile` handle from multiple
  threads at once and asserts byte-equality, so this positional path is now
  exercised under contention on all four CI platforms (including Windows, where
  the fixture-gated engine/oracle tests skip).
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

- Target and max RAM-mode semantics fixed, and the projected peak recalibrated,
  from three bugs the real-Mixtral Colab field report surfaced (the AUTO run
  printed "est. peak ~10 GiB" against a real ~5 GiB peak; `--target-ram-gb 2`
  REFUSED with a dishonest ~10 GiB floor dominated by full-`n_ctx` KV; and
  `--ram-mode max` held FEWER experts resident (6) + streamed MORE (~78 GiB)
  than AUTO (44 experts, ~67 GiB) - inverted). All three are residency/placement
  only, so decode stays byte-identical to the libllama oracle.
  - (1) Explicit `--target-ram-gb` now SHRINKS the KV window (the power-of-two
    step-down to the smallest safe window) and engages trunk streaming to hit an
    achievable small target, so an achievable target FITS instead of refusing.
    When a target genuinely cannot fit, the refusal names the HONEST minimal-KV
    floor (the smallest safe window + maximal trunk streaming + the top-k
    experts) - for real Mixtral the true floor is ~2.x GiB (the user proved
    2406 MiB decodes at a 2048-column window), NOT the full-`n_ctx` ~10 GiB
    figure the old refusal reported.
  - (2) `--ram-mode max` was INVERTED (fewer residents + more streaming than
    auto) because it sized its usable ceiling from the instantaneous free RAM;
    on a box whose big model had already consumed RAM, free was small so Max
    resolved a tiny ceiling. Max now sizes from ALL PHYSICAL RAM
    (`ram_total_bytes`), falling back to free only when total is unknown, and
    holds the MAXIMUM experts resident + full KV + trunk resident, streaming the
    LEAST. This makes Max MONOTONIC against Auto on the same machine:
    `max.expert_slots_resident >= auto.expert_slots_resident`, Max's KV window is
    at least Auto's, and `max.streamed <= auto.streamed` (Max never has fewer
    residents / more streaming than Auto).
  - (3) `planned_peak_bytes` was ~2x the engine's real resident footprint
    (conservative staging + compute + safety margin stacked on top of a full-KV
    charge), which is WHAT made the explicit-target path refuse achievable
    targets. It now tracks the engine's real windowed resident footprint
    (resident KV at the chosen window + resident experts + one-layer staging +
    trunk + process floor + a trimmed margin): slightly high so the fit
    invariant stays safe, no longer ~2x, and strictly below the same plan sized
    with full-`n_ctx` KV.

  Sandbox-proven through the REAL load path on a Mixtral-geometry fixture
  (`n_ctx` 32768, 32 layers, pack -> `.strata` -> `load_model` -> planner ->
  engine) on a tight 3 GiB-total / 2.25 GiB-free profile (free << total, so auto
  and max DIVERGE): AUTO resolves usable 1280 MiB, steps the KV window down to
  8192 (`< n_ctx`), holds `slots=104`, and streams some experts; MAX resolves
  usable 3072 MiB (= all physical RAM), keeps a FULL KV window (`kv_window=0`),
  holds `slots=256`, and streams nothing - so `max.slots (256) > auto.slots
  (104)`, Max's window >= Auto's, and only Auto streams. An explicit achievable
  ~1.2 GiB target FITS (`fits=true`, `kv_window` in `(0, n_ctx)`,
  `trunk_pool_layers > 0`); a genuinely-too-small target refuses naming a floor
  from the SMALLEST safe window (e.g. "KV window 256", not full KV); byte-identity
  across target/auto/max is exactly `0.0` and the oracle delta is `< 1e-1`. The
  absolute real-Mixtral per-mode peak-RSS numbers are Colab-only (the sandbox
  cannot download the ~19 GB model); the exact three re-verify commands and
  expected per-mode plan/stats are in `colab/README.md`.

- Conservative RAM budget: the auto expert-cache is now sized AFTER subtracting
  ALL other mandatory resident costs, and the planner can no longer plan a
  configuration that OOM-kills the box. This fixes the real symptom where a
  real Mixtral-8x7B Q3_K_M `.strata` (`n_ctx` 32768) on the 12 GiB free Colab
  box loaded the trunk, printed the plan line
  `experts[resident=122 bundles], experts STREAMED (bounded), est. peak ~9 GiB`,
  and was then KILLED (OOM, exit status 137 / SIGKILL) before producing any
  output or stats line - even though the "est. peak ~9 GiB" claimed to fit.
  Adding `STRATAFLOW_ENGINE_KV_WINDOW=512` made the OOM certain; the default run
  only barely survived at ~10.6 GB. Root cause: the planner sized the resident
  expert pool from ALL remaining RAM after only the trunk
  (`ram_for_experts = vram_left + ram`, ~122 bundles on Mixtral) and its
  `planned_peak_bytes` was a dishonest trunk+experts-only figure that omitted
  the KV cache (8 GiB at full `n_ctx`), the one-layer expert staging buffer, the
  ggml compute buffer, the process floor, and any safety margin - and it was
  never used as a hard bound. The fix reworks `plan_placement`
  (`src/plan/planner.cpp`):
  - A CONSERVATIVE usable-RAM ceiling replaces the optimistic MemAvailable
    reading: `usable_RAM = min(ram_free - max(1 GiB, 12.5%), 85% of ram_total)`,
    because MemAvailable on a shared Colab box over-reports what is really free.
  - The resident expert cache is sized from what is LEFT after every other
    mandatory resident cost is subtracted:
    `available_for_experts = usable_RAM - (trunk_in_RAM + resident_KV(chosen
    window) + one-layer staging at FULL n_expert (matching the engine's
    stage_gate/up/down, not n_experts_used) + a conservative ggml compute-buffer
    estimate + a ~160 MiB process floor + a safety margin of max(512 MiB, 8% of
    usable))`; only `available_for_experts / expert_bytes` gives the resident
    bundle count, clamped to `[n_experts_used, full_slots]`. The old
    `ram_for_experts = vram_left + ram` is gone.
  - `planned_peak_bytes` is now an HONEST, conservative upper bound: it is the
    sum of all those components, not the old trunk+experts-only number.
  - HARD FIT INVARIANT: the planner never returns a configuration whose own
    projected peak exceeds `usable_RAM`. It reduces the resident bundle count
    down to the `n_experts_used` floor rather than over-committing, so by
    construction `planned_peak_bytes <= usable_RAM` for any non-refused plan.
  - LOSSLESS auto KV-window default: when full-`n_ctx` KV (8 GiB on Mixtral
    `n_ctx` 32768) would blow the budget, the planner steps the resident KV
    window down to the largest power-of-two (`>= 256`) that fits, recorded in
    `plan.kv_window` and plumbed to the engine via the same
    `STRATAFLOW_ENGINE_KV_WINDOW` path (an explicit user window still wins), so a
    real large-context Mixtral fits by DEFAULT. The freed RAM LOWERS the peak and
    is NOT re-consumed by more experts (the safety margin is held fixed).
  - LOUD refusal: if even the minimal config (`n_experts_used` bundles + the
    smallest safe KV window + trunk) cannot fit, `plan.fits` is set false with a
    clear ASCII reason naming the shortfall, `sf_context_create` returns
    `SF_ERR_RUNTIME`, and the CLI prints the reason and exits non-zero - instead
    of the silent SIGKILL this change replaces.
  Before/after on the synthetic Mixtral-on-12GB box (`n_layer` 32, 8x2,
  `expert_bytes` ~52 MiB, trunk ~5.5 GiB, `n_ctx` 32768, `head_dim` 128,
  `n_head_kv` 8; `ram_total` 12 GiB, `ram_free` 10.6 GiB, no GPU): BEFORE the
  planner picked ~122 resident bundles and printed "est. peak ~9 GiB" while the
  true peak (adding ~8 GiB full KV + staging + compute + floor) OOM-killed the
  box (exit 137). AFTER `slots=19`, auto `kv_window=4096` (lossless; full
  `n_ctx` KV would not fit), honest projected peak 9495 MiB `<=` usable
  9497 MiB, well under 12 GiB with margin. Sandbox-proven by the pure-function
  planner unit test `test_mixtral_12gb_conservative_budget` in
  `tests/test_planner.cpp` (asserts the honest peak `<=` usable `<` ram_total
  AND is strictly larger than the old dishonest trunk+slots*expert_bytes value,
  the slot count is reduced below ~122 and `>= n_experts_used` and `< 256`, a KV
  window lowers the peak, and a pathologically small 2 GiB box triggers the loud
  `fits=false` refusal); the forward math and the libllama oracle byte-identity
  are untouched (residency/window sizing only). The REAL confirmation is the
  user's Colab re-run: with this fix the SAME default run and the
  `STRATAFLOW_ENGINE_KV_WINDOW=512` run from the OOM report should both DECODE
  (not SIGKILL) with identical text and a stats line showing peak RSS safely
  under 12 GB; the exact re-verify commands are in `colab/README.md`.
- Real-hardware empty-output regression in lossless KV paging (technique 6.5)
  fixed, and the cold store is now BOUNDED to used positions. On the real
  Mixtral-8x7B Q3_K_M (`n_ctx` 32768) a Colab run with
  `STRATAFLOW_ENGINE_KV_WINDOW=512` produced NO output at all - no plan line, no
  error, just an empty cell - while the default full-resident run decoded fine.
  Root cause: `Engine::load` PRE-ALLOCATED the per-layer KV cold store at the
  FULL trained `n_ctx` (a `std::vector<float>(n_ctx * floats_per_col)` per layer
  for BOTH K and V) whenever paging was engaged. For Mixtral (`n_ctx` 32768,
  `head_dim` 128, `n_head_kv` 8 => `floats_per_col` 1024, `n_layer` 32) that is
  `32 * 2 * 32768 * 1024 * 4` = 8,589,934,592 bytes (exactly 8 GiB, matching the
  Colab-measured `resident_kv_bytes` byte-for-byte). On the 12 GB box that 8 GiB
  cold allocation - ON TOP OF the ~0.9 GB resident weights + experts - threw
  during LOAD, before the CLI printed its plan line (the CLI prints the plan only
  after `sf_context_create` succeeds), which is why the cell was empty. The
  resident window itself (512 columns, ~128 MiB) was tiny and, with only ~17
  decoded positions, never even overflowed - so the bug was the cold store, not
  the window. The fix: the cold store is no longer pre-allocated at `n_ctx`. It
  is EMPTY at load and GROWS ON DEMAND, sized to the positions actually evicted
  (`(evict_pos + 1) * floats_per_col`, never `n_ctx`); for `W=512` and ~17
  positions it never evicts, so it stays 0 bytes. `resident_kv_bytes()` reports
  only the resident ring (`n_layer * 2 * floats_per_col * 4 * W`, proportional
  `W/n_ctx` of full), and a new `Engine::kv_cold_bytes()` accessor reports the
  current bounded cold-store size (0 when the window never overflowed). A growth
  allocation that fails now logs an ASCII error with the attempted byte count and
  fails the decode cleanly (returned up through `sf_generate` and surfaced by the
  CLI) instead of a silent no-output. The DEFAULT path (no
  `STRATAFLOW_ENGINE_KV_WINDOW`) stays full-resident, byte-identical, zero-
  overhead, and allocates no cold store; the explicit windowed path is the fit
  lever this change repairs. An AUTO KV-window default (sizing `W` from the RAM
  budget + `n_ctx`, like the auto expert-residency policy) is DEFERRED as a
  low-risk follow-on to keep this change minimal and the default byte-identical.
  Sandbox-proven on a new large-`n_ctx` fixture
  (`tools/testdata/make_tiny_moe_bigctx_gguf.py`, `n_ctx` 4096) via the new
  `tests/test_kv_paging.cpp::test_kv_paging_bigctx_small_window` gate at `W=64`
  (`W/n_ctx` 1/64), run with `STRATAFLOW_TEST_BIGCTX_MOE_GGUF`: the SHORT regime
  (no overflow, gen=8) reports `residentKV 32768 B`, `fullKV 2097152 B`,
  `coldBytes 0 B`, `cold_fetches 0`; the LONG regime (overflow, gen=80) reports
  `residentKV 32768 B`, `fullKV 2097152 B`, `coldBytes 12800 B`,
  `cold_fetches 650`. Both regimes are byte-identical to the full-resident engine
  (`max|delta| = 0.000e+00`) and within 1e-3 of the oracle; `residentKV ==
  fullKV / 64` proves the `W/n_ctx` reservation and `coldBytes 12800`
  (proportional to the 25 evicted positions, far below the full-`n_ctx` cold size
  of 2,097,152 B) proves the cold store is bounded. The existing `W=4` fixtures
  (`n_ctx` 64/256) still pass bit-identical. Colab re-verify of the real Mixtral
  (sandbox cannot download it): `STRATAFLOW_ENGINE_KV_WINDOW=512` should now
  DECODE text byte-identical to the default run and report `resident_kv` ~128 MiB
  (`resident_kv_bytes` ~134217728, i.e. `512/32768 * 8192 MiB`) instead of
  8192 MiB, cutting peak RSS by ~8 GB (from ~10.6 GB toward ~2.5 GB).
- `.strata` MoE shape pre-pass regression fixed (OOM on legacy Mixtral .strata).
  `load_strata_model` built `ModelShape` from a vocab-only `llama_model`, which
  reported `expert_bytes=0`/`trunk_bytes=0`/`total_bytes=0`. The superblock
  patch filled `is_moe`/`n_experts`/`n_layers` but NOT the byte fields or the
  top-k (`n_experts_used`), so `plan_placement` (gated on `expert_bytes > 0`)
  fell to the dense branch, computed 0 expert slots, streaming never engaged,
  and a real legacy-Mixtral `.strata` loaded fully resident and OOM-killed a
  12 GB box (`rc=-9`) at the default `--expert-slots 0`. The fix populates
  `expert_bytes` (from the first expert index entry's `blob_length`),
  `trunk_bytes`/`total_bytes` (from the superblock + index), and
  `n_experts_used` (from the embedded GGUF metadata's
  `<arch>.expert_used_count`, fallback 2) BEFORE planning. The plan now
  correctly sees MoE + bounded streaming for any `.strata` with experts. A
  regression test (`test_strata_shape_prepass`) packs the legacy fixture to
  `.strata`, loads it with a modest budget, and asserts the plan is the MoE path
  (NOT dense). A Mixtral-scale planner test (`test_mixtral_scale_12gb_auto`)
  guards the auto policy for the real shape + 12 GB budget.
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

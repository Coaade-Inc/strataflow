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
- [x] **Automatic hardware tuning - per-model expert residency (auto-sized).**
  The engine no longer needs a hand-tuned `--expert-slots`: by default it
  auto-sizes the resident expert-bundle pool PER MODEL from the measured free-RAM
  budget plus the model's own expert layout (`n_layer`, `n_expert`,
  `n_expert_used`, `expert_bytes`), with NO model-specific constants, so it
  adapts to any layout (8x2, 256x8, ...). `expert_slots_resident =
  clamp(ram_for_experts / expert_bytes, lower = max(n_expert_used, a whole layer
  when it fits), upper = n_layer*n_expert)`. When RAM holds the whole working set
  it caches it (no per-layer eviction/re-streaming - the Mixtral-on-Colab win);
  only when RAM genuinely cannot hold it does the pool bound below the full set,
  and never below one layer's top-k. Precedence: explicit `--expert-slots` >
  `--cache-gb` (expert-RAM cap) > auto-from-free-RAM. The number the engine
  actually uses now equals the number the plan reports, with the correct
  stream/resident state and a realistic peak estimate, and oracle byte-identity
  is preserved (an auto-chosen slot count decodes a byte-identical greedy
  sequence/logits to the fully-resident and to a fixed-slot run). Sandbox-proven
  on generated fixtures: the bench harness
  (`colab/strataflow_bench.py --layers-list 8 --experts-list 16 --slots-list
  2,16,0`) shows the auto row streaming ~8x fewer SSD bytes/token than a
  deliberately-too-small `slots=2` run on the SAME model (8.06 MB/tok vs
  64.39 MB/tok) and faster steady-state decode (47.3 vs 15.3 tok/s, mechanism
  proxy on random weights). The real large-MoE (Mixtral) speedup from relying on
  auto residency stays Colab-only (offline sandbox cannot download it).
  Regression fix (OOM on a legacy-Mixtral `.strata`): `load_strata_model` built
  the `ModelShape` from a vocab-only `llama_model`, so `expert_bytes`/
  `trunk_bytes`/`total_bytes` were 0 and the superblock patch did not set the
  byte fields or the top-k. `plan_placement`'s MoE branch is gated on
  `expert_bytes > 0`, so the model was misclassified as dense: 0 expert slots,
  no streaming, whole model planned resident, OOM-killing a 12 GB box. Fixed by
  populating `expert_bytes` (expert index `blob_length`), `trunk_bytes`/
  `total_bytes` (superblock + index), and `n_experts_used` (embedded GGUF
  metadata) BEFORE planning, so a real MoE `.strata` is planned as MoE +
  bounded streaming. Guarded by `test_strata_shape_prepass` (packs the legacy
  fixture and asserts the plan is NOT dense) and `test_mixtral_scale_12gb_auto`
  (auto policy on the real Mixtral shape + 12 GB budget).
- [x] **Layer-stratified residency + prefetch gating (bytes-streamed/token
  reduction, async no longer hurts).** Two composing levers on the bounded-RAM
  streaming path, both oracle byte-identical and adaptive to model shape with no
  machine constants. LEVER B: the single GLOBAL plain-LRU SlotPool is replaced
  by LAYER-STRATIFIED, frequency-aware per-layer residency - each layer gets its
  own LRU scope and a shape-derived capacity (`base = n_slots / n_layer`, floored
  at a single layer's top-k, spare slots distributed by observed per-layer
  routing frequency), so a layer's hot experts survive across tokens instead of
  being evicted by the intervening layers' sweeps. The TOTAL slot budget (and
  thus peak RSS) is unchanged; only how slots are partitioned/evicted changes.
  Sandbox-proven on `bench_moe-L8-E16` at the shape-derived tight budget
  (slots=32 = `n_layer * n_expert_used`): streamed bytes drop from 522,190,848
  to 327,155,712 (37.3% fewer, 35.8% fewer misses) at identical peak RSS, decode
  byte-identical. LEVER A: async prefetch no longer increases bytes streamed - a
  completed speculative read is only installed into a genuinely FREE per-layer
  slot (so it can never evict a still-needed authoritative resident), and the
  worker SELF-BACKS-OFF (stops issuing speculative reads) once the observed
  `completed-before-use` rate stays ~0 over a shape-derived window. This
  reverses the earlier async-on regression: on `bench_moe-L8-E16` at the bounded
  slots=24, async-on now streams 46,792,704 bytes/tok vs async-off 61,341,696
  (0.76x, async-on <= async-off), decoded text byte-identical at the same
  132.4 MiB peak RSS. The authoritative `ensure_layer_experts_resident` is
  unchanged (byte provenance / graph order identical), the compute path stays
  single-threaded (one extra I/O thread that never mutates the pool), and the
  default-ON async toggle + `completed-before-use`/`streamed_bytes` stats are
  intact. Sandbox-proven: the streamed-bytes/token reduction, byte-identity, and
  prefetch-no-longer-hurts invariant. Colab-only: the absolute wall-clock tok/s
  on the real ~19 GB Mixtral (`completed-before-use` stays ~0 on fast sandbox
  CPUs, mirroring 2-core Colab, so the overlap wall-clock win is hardware-
  dependent; exact before/after re-verify commands in `colab/README.md`).
- [x] **Stream the trunk (lossless edge-tensor streaming, technique 6.4).**
  Per `RESEARCH_50B_ON_A_LAPTOP.md` section 6.4, the two LARGE trunk tensors
  touched exactly once per token at the input/output edges - `token_embd.weight`
  (input embedding) and `output.weight` (lm_head) - now STREAM from the `.strata`
  like experts instead of sitting fully resident, while the HOT per-layer
  attention/norm/router trunk tensors stay resident (streaming them would be a
  hot-path throughput regression). At 50B scale these vocab x n_embd edge tensors
  dominate the fixed trunk cost, so this is the dominant FIT lever. The change is
  LOSSLESS (pure residency / byte provenance): the `.strata` format is bumped to
  a BACK-COMPATIBLE v2 that adds an optional per-trunk-tensor index (name, trunk-
  relative offset, length); the trunk block bytes are byte-for-byte identical to
  v1 and a v1 file still opens with no trunk index (full-resident fallback). On a
  v2 file the two edge tensors are created metadata-only and share ONE stable
  backing buffer, fetched synchronously on the compute thread immediately before
  the segment that reads them (same "resident before the op" discipline as the
  expert path; the async worker and SlotPool are untouched). Sandbox-proven,
  byte-identical and resident-bytes-lower on generated fixtures: on the F32 tiny
  MoE `resident_weight_bytes` drops 103,040 -> 69,248 B (-33,792 B, one edge
  tensor) and on the K-quant MoE 392,448 -> 354,432 B (-38,016 B), with the
  trunk-streaming decode bit-for-bit identical to the fully-resident decode
  (`max|delta| = 0.000e+00`, proven on the SAME v2 file via the
  `STRATAFLOW_ENGINE_NO_TRUNK_STREAM` test knob) and to the libllama oracle, on
  BOTH an F32 and a K-quant fixture; the v1 back-compat path decodes byte-
  identically to the oracle too. Colab-only: the absolute peak-RSS reduction on
  the real Mixtral/50B case (where the multi-GB resident embeddings + lm_head
  previously blocked the fit) is hardware-only; the exact re-verify command and
  expected effect are in `colab/README.md`.
- [x] **Adaptive per-layer trunk streaming (technique 6.4, FEAT-003 Part B).**
  Generalizes the edge-tensor streaming above to the DOMINANT remaining resident
  cost after the KV window: the four large per-layer attention projections
  (`blk.N.attn_{q,k,v,output}.weight`), ~5.5 GB across 32 layers on real
  Mixtral. They now live metadata-only in a per-layer context (`pltctx`) and are
  fetched into a BOUNDED HOT-TRUNK POOL (a ring of `trunk_pool_layers` layers)
  immediately before each layer's attention segment, reusing the SAME v2 trunk
  index (`read_trunk_tensor`) and resident-before-op discipline as the edge
  tensors; the tiny per-layer norms + router stay HOT (resident). BUDGET-DRIVEN:
  the planner derives `trunk_pool_layers` from the SAME conservative usable-RAM
  budget (`ModelShape::attn_trunk_bytes` summed from the trunk index), so a
  big-RAM box keeps the whole trunk resident (pool 0, zero per-token trunk
  traffic) and a tight 12 GB-class box streams it hard (pool as low as 1). The
  decision is plumbed to `Engine::load` as a load parameter via `GgmlModel`
  (exactly like `plan.kv_window`), so it applies BY DEFAULT with no env
  side-channel; the plan line gains a `trunk stream[hot=N layers]` segment.
  HONEST TRADEOFF: these weights are touched every token, so streaming adds
  per-token SSD traffic and is slower on a slow box - accepted because the FIRST
  goal is to FIT small, and the policy streams only as much as the budget
  forces. `resident_weight_bytes()` reflects the bounded hot pool;
  `StrataReader::streamed_trunk_bytes()` counts the per-layer trunk reads
  DISTINCTLY from the expert `streamed_bytes()` (the bench per-token expert
  number is unchanged). LOSSLESS (pure residency / byte provenance; same bytes,
  ops, graph order), single compute thread (no new TSan surface), with the
  `STRATAFLOW_ENGINE_NO_TRUNK_STREAM=1` knob forcing the whole trunk resident.
  Sandbox-proven on an attention-trunk-dominated Mixtral-mirroring fixture
  through the REAL load path (packer -> `.strata` -> `load_strata_model` ->
  planner -> engine): the plan chose `trunk_pool_layers=5` of 10 layers by
  default under a ~1.27 GiB budget, `resident_weight_bytes` dropped (e.g. the
  default CLI run on an `n_embd=4096 n_layer=16` fixture at `--ram-gb 1.3`:
  resident model weights 292.9 -> 110.9 MiB, peak RSS 329.5 -> 161.5 MiB), and
  the streamed decode was bit-for-bit identical (`max|delta| = 0.000e+00`) to the
  fully-resident decode on the SAME file and within the oracle quant tolerance,
  over a short AND a longer sequence (`tests/test_trunk_stream_adaptive.cpp`).
  Colab-only: the absolute real-Mixtral peak-RSS drop (the ~5.5 GB resident
  attention trunk heading toward a bounded hot set) is hardware-only; the
  re-verify command and expected effect are in `colab/README.md`.
- [x] **Lossless KV paging (technique 6.5, lossless half).** Per
  `RESEARCH_50B_ON_A_LAPTOP.md` section 6.5, the engine-owned per-layer K/V no
  longer reserve the full trained `n_ctx` up front: they are sized to a resident
  recent WINDOW `W` (a ring buffer keyed by `pos % W`, so the resident
  reservation is `W/n_ctx` of the full cost), and positions that scroll out of
  the window are tiered to a RAM-backed per-layer COLD page store (keyed by
  absolute position, the EXACT F32 bytes the ring held). The cold store is
  BOUNDED - it is empty at load and GROWS ON DEMAND, sized only to the positions
  actually evicted (never the full `n_ctx`). Attention
  re-assembles the full history `0..pos` (cold columns + resident ring columns
  in absolute order, handling wrap) and runs the IDENTICAL op sequence over it,
  so decode stays BYTE-IDENTICAL to the full-resident path and the oracle - no
  past position is dropped (this is NOT lossy StreamingLLM/SnapKV eviction). `W`
  defaults to `n_ctx` (so the DEFAULT path is byte-identical and zero-overhead)
  and is overridable via `STRATAFLOW_ENGINE_KV_WINDOW` to engage paging.
  `KvStore` (`src/kv/kv_store.h`) owns the paging bookkeeping; the ggml-touching
  copies live in `engine.cpp`; everything is on the single compute thread (no new
  background thread; TSan clean). A new `resident_kv_bytes()` / `full_kv_bytes()`
  accessor pair (and a `resident_kv_bytes` field appended ABI-safely to
  `sf_runtime_stats` + `sf_session_stats`, surfaced by the CLI) measures the win.
  Sandbox-proven byte-identical and resident-KV-lower on generated fixtures
  (`tests/test_kv_paging.cpp`): with `STRATAFLOW_ENGINE_KV_WINDOW=4` over an
  8-token generation (sequence longer than the window, so cold KV is paged out
  and fetched back - cold-fetch counter 462), resident KV drops 32,768 -> 2,048 B
  on the F32 tiny MoE (`n_ctx` 64) and 1,048,576 -> 16,384 B on the K-quant MoE
  (`n_ctx` 256), with the paged decode BIT-identical to the full-resident engine
  (`max|delta| = 0.000e+00`) and matching the libllama oracle (F32 within 1e-3;
  Q4_K seq-identical). A real-hardware fix followed: the cold store was first
  pre-allocated at full `n_ctx` (8 GiB on Mixtral: `n_layer` 32 x 2 x `n_ctx`
  32768 x `floats_per_col` 1024 x 4 B, matching the Colab-measured
  `resident_kv_bytes` byte-for-byte), which OOMed at LOAD on a 12 GB box and
  produced NO output at all (empty cell, before any plan line) when
  `STRATAFLOW_ENGINE_KV_WINDOW=512` was set; grow-on-demand is the fix, a genuine
  allocation failure now logs an ASCII error and fails the decode cleanly, a new
  `kv_cold_bytes()` accessor reports the bounded cold-store size, and the regime
  the tiny `W=4` fixtures missed (large `n_ctx`, window well below it, BOTH a
  short no-overflow and a long overflow run) is now a dedicated gate
  (`test_kv_paging_bigctx_small_window` over the `n_ctx` 4096 fixture
  `make_tiny_moe_bigctx_gguf.py`, `W=64`): SHORT (gen=8) residentKV 32768 B,
  fullKV 2097152 B, coldBytes 0 B, cold_fetches 0; LONG (gen=80) residentKV
  32768 B, fullKV 2097152 B, coldBytes 12800 B, cold_fetches 650 - both
  byte-identical (`max|delta| = 0.000e+00`) and within 1e-3 of the oracle
  (`residentKV == fullKV/64` proves the `W/n_ctx` reservation; coldBytes
  proportional to the 25 evicted positions, far below the full-`n_ctx` cold size
  2097152 B, proves the cold store is bounded). When the full-`n_ctx` KV fits the
  budget the DEFAULT stays full-resident (lossless, zero-overhead); when it would
  overflow, the auto KV-window default now ENGAGES (see the FEAT-002 Part A item
  below). Colab-only: the absolute peak-RSS reduction at real long context on
  Mixtral/50B is hardware-only - the re-verify (RUN 2,
  `STRATAFLOW_ENGINE_KV_WINDOW=512`) should now decode byte-identically with
  `resident_kv` ~128 MiB instead of 8192 MiB (~8 GB lower peak RSS); the exact
  command and expected numbers are in `colab/README.md`. Deferred follow-ons:
  SSD spill of cold pages (option b) and KV quantization (approximate, Stage 4).
- [x] **Conservative RAM budget - the planner guarantees a projected peak under
  the budget ("runs on everyday hardware").** The auto expert-cache is now sized
  AFTER subtracting ALL other mandatory resident costs, so the planner can no
  longer plan a configuration that OOM-kills the box. This repairs the real
  symptom: a real Mixtral-8x7B Q3_K_M `.strata` (`n_ctx` 32768) on the 12 GiB
  free Colab box loaded the trunk, printed
  `experts[resident=122 bundles], experts STREAMED (bounded), est. peak ~9 GiB`,
  then was KILLED (OOM, exit status 137 / SIGKILL) before any output - the
  "est. peak ~9 GiB" was dishonest. Root cause: the planner sized the resident
  pool from ALL remaining RAM after only the trunk (~122 bundles) and its
  `planned_peak_bytes` omitted KV (8 GiB at full `n_ctx`), one-layer staging, the
  compute buffer, the process floor, and any margin, and was never a hard bound.
  Fix (`src/plan/planner.cpp`): (1) a CONSERVATIVE usable-RAM ceiling
  `usable_RAM = min(ram_free - max(1 GiB, 12.5%), 85% of ram_total)` replaces the
  optimistic MemAvailable reading; (2) the expert cache is sized from what is
  LEFT after `trunk + resident_KV(chosen window) + one-layer staging at FULL
  n_expert (matching the engine's stage_gate/up/down) + a conservative ggml
  compute estimate + a ~160 MiB process floor + a safety margin of
  max(512 MiB, 8% of usable)`; (3) `planned_peak_bytes` is now an HONEST
  conservative upper bound (the sum of those components); (4) a HARD FIT
  INVARIANT - the planner never returns a plan whose projected peak exceeds
  `usable_RAM`, reducing the resident bundle count down to the `n_experts_used`
  floor rather than over-committing; (5) a LOSSLESS auto KV-window default - when
  full-`n_ctx` KV would blow the budget, the resident KV window steps down to the
  largest power-of-two (`>= 256`) that fits (recorded in `plan.kv_window` and
  plumbed via `STRATAFLOW_ENGINE_KV_WINDOW`, an explicit user window still wins),
  so a real large-context Mixtral fits by DEFAULT, with the freed RAM LOWERING
  the peak (not re-consumed by more experts - the margin is held); (6) a LOUD
  refusal - if even the minimal config cannot fit, `plan.fits=false` with a clear
  ASCII reason, `sf_context_create` returns `SF_ERR_RUNTIME`, and the CLI prints
  the reason and exits non-zero, instead of a silent SIGKILL. Before/after on the
  synthetic Mixtral-on-12GB shape (trunk ~5.5 GiB, `expert_bytes` ~52 MiB,
  `n_ctx` 32768, `head_dim` 128, `n_head_kv` 8; `ram_total` 12 GiB, `ram_free`
  10.6 GiB): BEFORE ~122 bundles / "est. peak ~9 GiB" that OOM-killed (exit 137);
  AFTER `slots=19`, auto `kv_window=4096`, honest projected peak 9495 MiB `<=`
  usable 9497 MiB, well under 12 GiB with margin. Sandbox-proven by the
  pure-function planner unit test `test_mixtral_12gb_conservative_budget`
  (`tests/test_planner.cpp`), which reproduces the over-commit deterministically
  with synthetic shapes (no model) and asserts the honest peak `<=` usable `<`
  ram_total and strictly above the old dishonest value, the reduced slot count,
  the KV window lowering the peak, and the pathological-budget loud refusal.
  Oracle byte-identity is preserved (residency/window sizing only; forward math
  unchanged). Colab-only: the SAME default and `STRATAFLOW_ENGINE_KV_WINDOW=512`
  runs from the OOM report are EXPECTED to DECODE (not OOM) with identical text
  and peak RSS safely under 12 GB, pending the user's Colab re-run; the exact
  re-verify commands and the projected safe peak are in `colab/README.md`.
- [x] **Auto KV window ENGAGES BY DEFAULT on the real `.strata` load path
  (FEAT-002 Part A - "fit small now").** The conservative budget + auto KV
  window above chose a window correctly in the pure-function planner test, but on
  the REAL Mixtral `.strata` the DEFAULT run (no `STRATAFLOW_ENGINE_KV_WINDOW`,
  no flags) still reserved the full-`n_ctx` KV (8192 MiB on Mixtral) and the plan
  line had NO `kv window=` segment - the exact Colab symptom. Two breaks, both
  fixed: (1) the real synthesized-metadata path read
  `context_length`/`head_count`/`head_count_kv` with `gguf_get_val_u32`, which
  aborts or returns 0 when the source stored them under a different GGUF integer
  encoding (the `n_ctx_train(0)` warning is the tell), so the planner saw zero
  geometry and its window step-down (gated on non-zero geometry) never fired
  while the engine independently read `n_ctx` and reserved the full KV; fixed by
  a robust type-dispatching integer read (`src/common/gguf_meta.h`
  `gguf_read_u32`) shared by the planner, the engine, and the packer so all three
  see IDENTICAL geometry on every path. (2) the plan's chosen window reached the
  engine ONLY through the process-global `STRATAFLOW_ENGINE_KV_WINDOW` env
  side-channel; now `PlacementPlan::kv_window` is a DIRECT `Engine::load`
  parameter plumbed through `GgmlModel`, applied by DEFAULT. Resolution order in
  the engine: explicit `STRATAFLOW_ENGINE_KV_WINDOW` env (user/test) WINS, else
  the plan-provided window, else full `n_ctx`. Sandbox-proven on a
  Mixtral-geometry legacy-per-expert fixture (`n_ctx` 32768, `head_dim` 128, GQA
  `n_head_kv` 2, 32 layers, full KV 2048 MiB) packed through the REAL
  `strata-pack` -> `.strata` -> `load_strata_model` -> planner -> engine under a
  simulated 12 GiB budget with NO env override (`tests/test_kv_window_real_path.cpp`):
  the loaded `ModelShape` reports geometry non-zero, the plan chose
  `kv_window=16384` (`< n_ctx`), and `resident_kv_bytes()` dropped
  2147483648 -> 1073741824 with the plan-window greedy decode BIT-IDENTICAL to the
  full-resident run (`max|delta| = 0.000e+00`) and matching the libllama oracle
  on the matched stacked fixture. Residency/window sizing only; forward math and
  graph order unchanged, so decode stays byte-identical. Colab-only: the absolute
  real-Mixtral peak-RSS drop is hardware-only; the re-verify command and expected
  plan/stats are in `colab/README.md`.
- [x] **Target-RAM capability: one dial for fit vs speed (modes {auto, max} +
  explicit numeric target).** A single target-RAM concept now drives ALL the fit
  levers together (KV window, expert slots/cache, adaptive per-layer trunk
  streaming, and the VRAM placement the planner already budgets) from one
  decision, exposed on BOTH the CLI (`--target-ram-gb X`, `--ram-mode auto|max`,
  and `--max-performance` as an alias for `--ram-mode max`) and the C ABI (two
  fields appended at the END of `sf_context_params`: `uint64_t
  target_ram_budget` in bytes, 0 = use the mode; `uint32_t ram_mode` taking
  `sf_ram_mode` { `SF_RAM_MODE_AUTO = 0`, `SF_RAM_MODE_MAX = 1` }, default-zero
  == auto). Precedence, highest wins: explicit low-level manual knobs
  (`STRATAFLOW_ENGINE_KV_WINDOW` / `--expert-slots` / `--cache-gb`) >
  `--target-ram-gb` (`target_ram_budget`) > `--ram-mode max` > `--ram-mode auto`
  (the DEFAULT, `ram_mode == 0`). An explicit target is used directly as the
  usable budget and drives the KV-window step-down + trunk streaming so the
  projected peak fits the target even below the trunk-resident floor; if even the
  minimal configuration cannot fit the target the planner refuses LOUDLY
  (`plan.fits=false` + a reason naming the achievable floor), where the floor is
  now the HONEST minimal-KV config (the smallest safe window + maximal trunk
  streaming + the top-k experts, ~2.x GiB on real Mixtral), NOT the dishonest
  full-`n_ctx` ~10 GiB figure the pre-fix refusal reported, surfaced as
  `SF_ERR_RUNTIME`. **Auto is the good-neighbor default**: it leaves headroom for
  other apps on a shared laptop by building on the conservative `usable_ram`
  share (`free - max(1 GiB, 12.5%)`, capped at 85% of total), so the model does
  not take the whole box. **Max dedicates the machine**: it sizes its usable
  ceiling from ALL PHYSICAL RAM (`ram_total_bytes`), NOT the instantaneous free
  RAM, then holds the MAXIMUM experts resident with a full KV window and the
  trunk fully resident (minimal streaming, maximal tok/s). Sizing from total
  physical RAM makes Max MONOTONIC against Auto on the same machine:
  `max.expert_slots_resident >= auto.expert_slots_resident`, Max's KV window is
  at least Auto's, and Max streams the trunk no harder than Auto
  (`max.streamed <= auto.streamed`) - Max is never inverted (never fewer
  residents / more streaming than Auto). This fixes the earlier inversion where
  Max sized from the instantaneous free RAM: on a box whose big model had already
  consumed RAM, free was small, so Max resolved a tiny ceiling and held FEWER
  experts + streamed MORE than Auto. GPU honesty, stated
  plainly: auto/max MAY read VRAM from the profiler and the planner BUDGETS VRAM
  placement (`plan.trunk_layers_in_vram` / `plan.expert_slots_vram`, mapped by
  `derive_llama_placement` to `n_gpu_layers`), but that only affects the
  NON-authoritative plain-GGUF libllama sidecar; the authoritative `.strata`
  engine decode runs CPU-only (every `llama_model_params` in `engine.cpp` sets
  `n_gpu_layers = 0`), so GPU PLACEMENT is budgeted while full GPU-accelerated
  COMPUTE is the enterprise roadmap track below, NOT implemented. The RAM side is
  the proven, measured mechanism. Residency/placement only, so decode stays
  byte-identical across all three modes and to the libllama oracle (the levers
  change which bytes are resident, never which bytes compute or the graph order).
  The projected peak is a RECALIBRATED, honest estimate: it previously ran ~2x
  the engine's real resident footprint (printing "~10 GiB" for an auto run that
  really peaked ~5 GiB and "~11 GiB" for a max run that really peaked ~2.2 GiB,
  because conservative staging + compute + margin stacked on top of a full-KV
  charge), which is WHAT made the explicit-target path refuse achievable targets;
  it now tracks the engine's windowed resident footprint (slightly high so the
  fit invariant stays safe, no longer ~2x). Sandbox-proven through the REAL load
  path on a Mixtral-geometry fixture (`n_ctx` 32768, 32 layers, pack ->
  `.strata` -> `load_model` -> planner -> engine) on a tight 3 GiB-total /
  2.25 GiB-free profile (free << total, so auto and max DIVERGE): an explicit
  `--target-ram-gb` ~1.2 GiB fits with `kv_window` in `(0, n_ctx)`,
  `trunk_pool_layers > 0` (trunk streaming engaged); AUTO resolves usable
  1280 MiB, steps the KV window down to 8192 (`< n_ctx`), holds `slots=104`, and
  streams some experts; MAX resolves usable 3072 MiB (= all physical RAM, the
  fix), keeps a FULL KV window (`kv_window=0`), holds `slots=256`, and streams
  nothing; the monotonicity invariants hold (`max.expert_slots_resident
  (256) >= auto (104)`, Max's window >= Auto's, `max.streamed <= auto.streamed`),
  a genuinely-too-small target refuses with a floor from the SMALLEST safe window
  (not full KV), and the decode is byte-identical across all three modes
  (`|delta| = 0.0`) within the soft oracle bound. Colab-only: the real
  Mixtral-8x7B-Instruct Q3_K_M run that this dial maps to (next bullet) is the
  absolute-number re-verify; the sandbox proves the mechanism, the mode
  differences, the honest refusal floor, estimate sanity, and byte-identity.
- [x] **AUTO coherent retune + batched/sequential expert I/O (technique 6.7,
  first half).** Fixes the two coupled problems the real-Mixtral Colab run
  exposed, both residency/placement-only so decode stays byte-identical.
  (1) COHERENT AUTO POLICY (`src/plan/planner.cpp`): AUTO now takes a
  comfortable, DOCUMENTED share of total RAM (the conservative `usable_ram`
  good-neighbor budget) and co-selects the three fit levers (KV window, trunk
  pool, expert cache) consistently to FILL that share - more resident means less
  streaming. The KV-window step-down now budgets against the FULL resident trunk,
  so AUTO windows the KV BEFORE the trunk is forced to hard-stream, enforcing the
  COHERENCE INVARIANT (asserted in code and in three tests): AUTO never keeps the
  full-`n_ctx` KV resident while simultaneously hard-streaming the trunk (if
  `0 < trunk_pool_layers < n_layers` then `0 < kv_window < n_ctx`). Before, AUTO
  on a 12 GiB-class box was INCOHERENT (full KV while hard-streaming the trunk,
  caching only 2 experts); now, when RAM affords it, it keeps the trunk resident
  and caches far more. On the synthetic 12 GiB Mixtral shape: BEFORE incoherent
  (windowed KV AND hard trunk streaming pool=20/32 AND only 4 experts, peak ~=
  usable), MAX fits=0 (projected peak 14097 MiB `>` usable 12288 MiB); AFTER
  coherent (AUTO `kv_window=8192`, trunk fully resident pool=0, 14 experts), MAX
  fits=1 (peak 12249 `<=` 12288, `kv_window=16384`, 18 experts) and stays
  `>= AUTO` on every lever. Experts resident went 2 (old) -> 14 (synthetic
  12 GiB) / 104 (mxg real fixture). (2) LOSSLESS BATCHED/SEQUENTIAL EXPERT I/O
  (`src/tws/strata_file.*`, `src/model/engine/engine.cpp`):
  `StrataReader::read_bundle` issues ONE `read_at` over a whole `(layer, expert)`
  bundle and splits it into gate/up/down by the known relative offsets, replacing
  the three separate `read_blob` calls - byte-identical (the bytes `memcmp` equal
  to the three per-kind reads, slices tightly packed gate|up|down,
  `streamed_bytes()` totals unchanged), with a `read_calls()` counter proving the
  3:1 read-count cut, through the REAL `StrataReader` in
  `tests/test_coalesced_expert_read.cpp` (`read_bundle` counts toward
  `streamed_bytes()` all-or-nothing - only a full-length read counts - so the
  per-kind fallback on a short read never double-counts). PREDICTOR-DRIVEN
  RESIDENCY: within the mode's budget, busier layers keep more of their reused
  experts resident via `SlotPool::set_layer_weights(ExpertPredictor::layer_activity())`
  (refined every few tokens in `Engine::forward`), proven end to end by
  `test_predictor_driven_residency` in `tests/test_slot_pool.cpp`; the top-k
  floor keeps decode byte-identical. The larger coherent AUTO cache cuts
  bytes/token: on the mxg real fixture streamed bytes/token drop from 29,408,625
  (2-slot baseline) to 3,008,102 (104-slot AUTO), ~9.8x fewer. Greedy decode is
  BIT-IDENTICAL across AUTO/TARGET/MAX (`|delta| = 0.000e+00`) and within the
  quant bound vs the libllama oracle (`5.179e-02 < 1e-1`). Full matrix green
  (GCC + Clang `-Werror`, release + clang ctest 22/22, the fixture-blindness
  real-load gate, and TSan 22/22). Colab-only: the current real speed is
  ~0.06 tok/s (~16 s/token) on the 12 GiB / 2-core / ~4 GiB/s SSD box, I/O-bound
  (~74-78 GB streamed per 16 tokens, ~4.6 GB/token); this lever attacks both the
  bytes/token term (bigger coherent AUTO cache) and the effective-bandwidth term
  (fewer, larger sequential reads), but the box stays fundamentally I/O-bound on
  2 cores (no compute/IO overlap; the async prefetch self-backs-off at
  `completed-before-use = 0`), so NO tok/s number is promised - the absolute
  real-Mixtral improvement is a Colab/hardware re-measurement. REMAINING SPEED
  WORK toward the continuation: the adjacent-expert multi-bundle coalescing and
  the co-activation packing LAYOUT (`PLAN.md` 3.10, technique 6.7 second half),
  then speculative / draft decoding (technique 6.8, Phase 5 below) as the biggest
  effective tok/s multiplier.
- [x] **Compress what streams - optional lossless on-disk expert-blob
  compression (technique 6.7b).** The AirLLM "compress what you stream" idea
  applied to the expert path: `strata-pack --compress lz` compresses each
  per-expert blob at pack time with a small self-contained in-tree LZ codec
  (`src/common/lz_codec.{h,cpp}`, no new submodule, no third-party dependency),
  storing the SMALLER of {compressed, verbatim} per blob, and `StrataReader`
  decompresses transparently at the `read_bundle` / `read_blob` seam into
  per-call scratch, so the engine's compute sees byte-identical expert bytes
  while FEWER (compressed) bytes stream from the SSD and `streamed_bytes()`
  counts the compressed size. The `.strata` format bumped to v3, OPTIONAL and
  back-compatible: v1 and v2 files still load and decode byte-identically, and an
  uncompressed v3 pack is byte-for-byte identical to the prior packer output
  (SHA256-verified); default stays uncompressed. LOSSLESS: the decode on a
  compressed `.strata` is byte-identical to the uncompressed one (per-step logit
  `max|delta| = 0.000e+00`) and matches the libllama oracle within the quant
  tolerance (`~5.2e-2 < 1e-1`). The decompress-vs-SSD tradeoff is measured and
  safe: codec decompress is ~6.9 GiB/s (gcc release; clang ~6.1 GiB/s) on a
  32 MiB mixed buffer, well above the ~4 GiB/s SSD per-core bandwidth, so
  decompress is NOT the new bottleneck and the lever does not backfire. HONEST
  RATIO: on the already-K-quantized expert fixture (`q4_k`) the ratio is
  ~1.0000 - the already-quantized, high-entropy random-weight experts do not
  shrink and every blob falls back to verbatim; AirLLM reported larger wins on
  higher-entropy / differently-quantized models, so we do NOT claim their ~3x.
  The net wall-clock effect scales with whatever ratio the real trained-weight
  Mixtral achieves, a Colab-only measurement. The MECHANISM is proven end to end
  on a deliberately compressible blob: a 16384-byte blob stored as 224 bytes on
  disk, `read_bundle` returns the full 16384 bytes byte-exact, and
  `streamed_bytes()` counts 224 (the compressed size), not 16384. The codec is
  pure (per-call scratch only), race-free for the concurrent compute-thread and
  prefetch-worker reads. Covered by `test_lz_codec` (round-trip + decompress
  GiB/s) and `test_compressed_strata` (real-path pack->StrataReader->engine
  oracle gate, skips cleanly without the fixture); full ctest 24/24 green under
  GCC + Clang `-Werror`. REVIEW FOLLOW-UP (fused + compressed read path): a
  FUSED gate_up bundle (`gate_rel == up_rel == 0`, a two-slice `gate_up|down`
  layout - gpt-oss / granite / gemma / phi3-style) is now read and counted ONCE
  on the compressed path, not 3x/2x. The compressed per-kind `read_blob`
  fallback used to attribute the compressed size to any slice with
  `slice_rel == 0` (so a fused bundle, whose gate AND up are both at rel 0,
  counted it TWICE in `streamed_bytes()`), and `read_bundle` rejected every
  fused layout so the engine fell back to three per-kind calls that each
  re-decompressed the WHOLE blob. Fixed by keying the `read_blob` increment on
  the gate KIND (`kGate`/`kGateUp`) rather than `slice_rel == 0`, and by running
  `read_bundle`'s compressed branch BEFORE its verbatim fused rejection so a
  compressed fused bundle decodes once into `dst` (`[gate_up | down]` is
  byte-identical to `[gate | up | down]`). Benchmark-counter / CPU-cost only, NOT
  a correctness or byte-identity break; byte-identity, default byte-for-byte
  identity, v1/v2/v3 back-compat, and per-call-local decode are all preserved. A
  new fused-layout seam subtest in `test_compressed_strata` pins both fixes (the
  in-tree fixtures are legacy/non-fused, so this is the first test to exercise
  fused + compressed). FOLLOW-ONS (out of scope here): trunk and KV
  compression, and a separate opt-in LOSSY re-quant lever (needs a quality gate).
- [x] **MAX clamps to the largest fitting plan, never refuses (12-13 GiB-class
  regression).** RamMode::Max (and Auto) are MODES, not numeric ceilings: a pure
  mode run (`target_ram_budget == 0` AND `ram_budget == 0`) now ALWAYS converges
  on the largest plan that FITS physical RAM and never refuses while the minimal
  config fits. The regression (`src/plan/planner.cpp`): when the full-`n_ctx` KV
  (8 GiB on real Mixtral) + the minimal expert floor fit under MAX's larger
  all-physical-RAM usable, the KV step-down did NOT engage, so MAX kept the full
  KV window; the MAX>=AUTO monotonicity clamp then forced the resident expert
  count up to the reference AUTO plan's (AUTO had windowed its KV DOWN and bought
  MORE experts than MAX can hold at full KV), overflowing usable so the fit
  invariant set `fits=false` and the CLI exited 1. THE FIX (one coherent
  clamp-not-refuse mechanism): when holding at least AUTO's resident experts does
  not fit at MAX's current (larger) KV window, MAX steps its KV window DOWN to AT
  LEAST AUTO's (bounded by the smallest safe window) and adopts at least AUTO's
  trunk pool, then sizes the resident expert count from the remainder (never
  below AUTO's); a general MODE-path backstop windows the KV, streams the trunk,
  and caps experts from the remainder for any mode path whose natural plan would
  overflow, so Auto never refuses either. Because AUTO already fits inside
  `auto.usable <= max.usable` with its windowed lever set, adopting at least
  AUTO's levers always lands a fitting MAX plan whenever AUTO fits. The LOUD
  refusal (`fits=false` + the honest `refusal_reason` naming the floor) is now
  reachable ONLY for an EXPLICIT numeric ceiling (`target_ram_budget` or
  `ram_budget > 0`) pinned below the achievable floor; that wording/floor
  computation is UNCHANGED. Preserves the MAX>=AUTO monotonicity, the fit
  invariant (`planned_peak_bytes <= usable_ram_bytes` for every returned fitting
  plan), and byte-identity (residency/placement/sizing only). On the synthetic
  13 GiB repro (full KV 8 GiB, 32 layers x 8 experts ~72 MiB, 0.5 GiB trunk):
  BEFORE MAX refused (natural peak 14630 MiB vs usable 13312 MiB, floor 2510 MiB)
  while AUTO fit (peak 10343 MiB, 57 slots, `kv_window=16384`); AFTER MAX fits
  (peak 10518 MiB `<=` 13312 MiB, 57 slots, `kv_window=16384`), monotonic with
  AUTO. Full matrix green: GCC 11.5 + Clang 15 `-Werror`, release + debug ctest
  22/22 (incl. the two new regressions `test_max_clamps_not_refuses_12gb` and
  `test_max_clamps_not_refuses_real_path`), and the gated real-load path
  byte-identical across AUTO/TARGET/MAX (`max|delta| = 0.000e+00`, oracle
  `5.179e-02 < 1e-1`). The planner is a pure function with no threads (no TSan
  concern for this change). Colab-only: the real ~19 GB Mixtral MAX run on the
  12 GiB box is the absolute re-verify; the fix and the geometry regression are
  proven end-to-end in-sandbox. No change to `third_party/llama.cpp`.
- [x] Relicense to the Coaade Source-Available License v1.0 (#6, #7).
- [x] Byte-identical correctness gate against the libllama oracle on every
  engine step; CI green on Linux (GCC+Clang), macOS, Windows.

---

## Peak RSS over time (reconciliation)

A note to set the record straight, because the history is easy to misread. The
recollection that "early on, when we first ran and tested the engine, peak RSS
came out to a couple of GB or less" is CORRECT. Peak RSS did not quietly
regress; it jumped exactly once, when a much larger real model was run at a very
large context, and the KV-paging work has since brought it back down. Here is
the honest timeline, grounded only in this repo's own merged measurements (see
the "Per-layer staging no longer scales with model size" and "Lossless KV
paging" entries above, `CHANGELOG.md`, and
[`RESEARCH_50B_ON_A_LAPTOP.md`](./RESEARCH_50B_ON_A_LAPTOP.md) sections 1-2):

- **Early / small and dense models genuinely peaked low.** The generated-MoE
  demos measured ~132 MiB peak RSS. The bounded-staging fix (#27) recorded a
  785 MiB model at ~132 MiB and a 1177 MiB model also at ~132 MiB. The real
  downloaded dense model (TinyLlama-1.1B-Chat Q4_K_M, ~0.67 GB on disk) ran end
  to end on the free Colab box with coherent output at a low peak well under a
  couple of GB. So "early tests peaked around a couple GB or less" is accurate.
  As the staging entry explains, the ~132 MiB figure is largely the FIXED
  llama/ggml backend + vocab process floor; the quantities that actually move
  with model and budget are StrataFlow's own resident weights, the resident KV,
  and the bytes streamed per token.

- **The jump to ~10.6 GB peak RSS came ONLY with the real large Mixtral-8x7B
  Q3_K_M at its FULL trained context `n_ctx` 32768.** This was not an engine
  regression and not the trunk or the bounded experts. It was measured
  byte-for-byte: the KV cache was reserved for the full 32768-token context in
  F32, which is `n_layer` 32 x 2 (K and V) x `n_ctx` 32768 x `floats_per_col`
  1024 x 4 B = 8,589,934,592 bytes, i.e. 8192 MiB (8 GiB) resident KV - the
  overwhelming majority of the ~10.6 GB peak. The trunk stayed small (~939 MiB
  resident) and the experts stayed bounded. In other words, a far larger model
  AT A VERY LARGE CONTEXT was run; the resident floor did not creep up on the
  small models.

- **KV paging (#42) plus the real-scale cold-store fix (#43) address exactly
  this.** With a bounded KV window (`STRATAFLOW_ENGINE_KV_WINDOW`, e.g. 512) the
  resident KV drops proportionally (`W/n_ctx`): for Mixtral at `W` 512 that is
  ~128 MiB instead of 8192 MiB, about 8 GB lower, with decode byte-identical to
  the full-resident path. That restores the "~2 GB" regime for a real large
  model: the projected post-fix peak RSS on the real Mixtral run is ~2.5 GB.
  Keep this HONEST: ~2.5 GB is the EXPECTED / projected number pending the
  user's Colab re-run (RUN 2 in `colab/README.md`); it is Colab-measured
  territory, not a sandbox measurement, so it is a projection to confirm, not a
  confirmed sandbox figure. The in-sandbox proof is the byte-identity and the
  `W/n_ctx` resident-KV reduction on the fixtures, not the absolute GB on real
  hardware.

Bottom line: the early couple-of-GB-or-less recollection is right; the one-time
spike to ~10.6 GB was KV reserved at full 32768 context on a real large model;
KV paging brings a real large model back toward ~2.5 GB (projected,
Colab-pending), byte-identical.

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
  The conservative-RAM-budget fit invariant (see the done item above) removes the
  silent-OOM blocker on this: on the 12 GiB Colab box the planner now guarantees
  a projected peak under the budget, auto-sizes the resident experts and the KV
  window to fit, and refuses loudly rather than SIGKILL, so the real Mixtral
  default and `STRATAFLOW_ENGINE_KV_WINDOW=512` runs are EXPECTED to decode
  without OOM - pending the user's Colab re-run (sandbox-proven via the
  synthetic-Mixtral planner unit test; the real end-to-end decode is Colab-only).
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
- [x] **Misleading placement-plan line (FIXED).** On the `.strata` streaming
  path the CLI used to print `plan: ... experts fully resident, peak ~N GiB` from
  the planner's estimate, which contradicted the streaming reality. The plan
  summary now reports the actual resident bundle count the engine uses, the
  correct state (`experts fully resident` when the whole working set fits, else
  `experts STREAMED (bounded)`), and a realistic peak (trunk + resident bundles +
  one-layer staging). What RUNS equals what is REPORTED.
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
  a larger runtime), not proven in the offline sandbox/CI. The harness also
  contrasts the auto row (slots=0) against a deliberately-too-small fixed slot
  count for the SAME model and reports that auto streams far fewer bytes/token
  (and decodes faster), demonstrating the per-model auto-residency policy. With
  `--async-list on,off` it now also reports, per `(model, slots)`, that async-on
  streams `<=` as many bytes/token as async-off at a bounded slot count (the
  prefetch-no-longer-hurts invariant from the gating lever above), alongside the
  `completed-before-use` overlap metric; the streamed-bytes/token reduction from
  layer-stratified residency vs the old global LRU is likewise sandbox-measured
  (see "Layer-stratified residency + prefetch gating"). What stays Colab-only is
  the absolute wall-clock tok/s on real large models - the mechanism (fewer
  bytes streamed/token, async no longer increasing streamed bytes, byte-
  identity) is proven in-sandbox; the throughput ladder is hardware-dependent.

### Breadth and performance

- [ ] **Enterprise / multi-GPU execution (future work, nothing implemented
  yet).** The no-GPU laptop path stays the core mission and the proven
  differentiator; this is an ADDITIONAL enterprise tier that uses GPUs when they
  are present. A GPU remains an OPTIONAL accelerator, never a requirement, and
  none of the items below are built or validated yet. The whole track can only
  be validated on real GPU hardware - it cannot be tested on the CPU-only CI or
  sandbox - mirroring the honest caveat the single GPU bullet always carried.
  It is a large, hardware-dependent track, orthogonal to the current CPU/disk
  fit-and-speed levers in
  [`RESEARCH_50B_ON_A_LAPTOP.md`](./RESEARCH_50B_ON_A_LAPTOP.md); sequence it
  accordingly (after the lossless CPU/disk levers that make a 50B-class model
  fit and run on the no-GPU box).

  - [ ] **Single-GPU acceleration first.** Route the hot experts, the hot trunk
    and the KV cache into VRAM via a GPU ggml backend using
    `ggml_backend_sched` (CPU + a GPU buffer type). This is NOT a separate code
    path: VRAM is already the TOP tier in the engine's design - the tiered
    weight store spans VRAM, RAM and SSD (see
    [`ENGINE_CORE_DESIGN.md`](./ENGINE_CORE_DESIGN.md) sections 2.4 and 3.5, the
    `ggml_backend_sched` plan and the SSD->RAM / RAM->VRAM transfer nodes, and
    the strata/flow VRAM-RAM-SSD diagram). So the GPU becomes simply the FASTEST
    tier of the same streaming engine: the same residency and byte-provenance
    discipline, just with VRAM as the destination. Structurally reachable
    because the engine already runs on a swappable ggml backend; still needs
    real GPU hardware to validate.

    FOUNDATION ALREADY IN PLACE (CPU-tested, not GPU-validated). The
    VRAM-placement DECISION this track consumes is built and under test, so the
    GPU compute step plugs into an existing, exercised seam rather than a blank
    one:
      - GPU DISCOVERY: `gpu_discovery` enumerates every ggml
        `GGML_BACKEND_DEVICE_TYPE_GPU` device (name, `vram_bytes`, backend) into
        `HardwareProfile::gpus`; a no-op that leaves the list empty on a CPU-only
        box (so the whole path is a no-op without a GPU).
      - VRAM-FIRST PLACEMENT: the planner's Rule 1/Rule 2
        (`src/plan/planner.cpp`) place the always-active trunk, then the routed
        experts, on the FASTEST device first - VRAM, then RAM - producing
        `plan.trunk_layers_in_vram` and `plan.expert_slots_vram` (the
        KTransformers / llama.cpp-offload insight, VRAM as the top tier).
      - PURE TRANSLATION: `derive_llama_placement`
        (`src/plan/llama_placement.{h,cpp}`) maps that plan to the concrete
        knobs (`n_gpu_layers`, and the routed-expert CPU-offload override whose
        regex already covers the FUSED `gate_up_exps` tensors), as a pure
        function over plain structs with no ggml/llama calls - unit-testable with
        no GPU. Covered by `test_llama_placement` (no-GPU is CPU-identical,
        small-GPU + big-MoE offloads experts, whole-working-set-in-VRAM keeps
        experts resident, dense never offloads, `n_gpu_layers` clamps to the
        layer count) and the planner's VRAM-pressure cases in `test_planner`.
    SELECTABLE-BACKEND SEAM NOW IN PLACE (CPU-validated, not GPU-validated).
    The engine's ggml compute backend is SELECTABLE: a `STRATAFLOW_GPU=off|cuda|
    metal|vulkan` CMake option (default `off`) wires the vendored ggml backends
    from the top-level CMake and defines compile-time guards on the engine TU,
    and at runtime the engine picks the first available GPU ggml backend via the
    backend-agnostic ggml-backend DEVICE/REGISTRY API (falling back to CPU when
    none is compiled in or present). When a GPU is selected the gallocr and
    every resident buffer (trunk `wbuf`, KV `kvbuf`, expert staging `estbuf`) are
    allocated from the selected backend's default (device) buffer type, and the
    per-token expert "fault bytes into the resident region" step becomes a
    host->device upload (`ggml_backend_tensor_set`) through a single
    host-or-device `stage_write` primitive. The planner VRAM budget and the C
    ABI `preferred_backend` are plumbed to `Engine::load`. For this first
    increment, when a GPU is selected the whole resident set runs in VRAM and
    the streamed trunk is forced resident (streamed-trunk-on-GPU is a follow-on);
    the per-token EXPERT upload seam is device-correct regardless.
    HONEST SCOPE: on the default CPU-only build the whole GPU path compiles out,
    the engine selects CPU, and decode is byte-identical to before and the
    oracle (`max|delta| = 0.000e+00`), proven by `test_backend_selection` (CPU
    selection + byte-identity) under GCC + Clang `-Werror`. The remaining,
    hardware-gated work is the actual GPU-COMPUTE validation (and
    streamed-trunk-on-GPU / multi-tier `ggml_backend_sched` splitting), which
    cannot be built or run-validated in the CPU-only sandbox/CI and must be
    exercised on real GPU hardware.
  - [ ] **Multi-GPU parallelism.** Expert / tensor / pipeline parallelism across
    devices; a device-placement policy deciding which layers and experts live on
    which GPU; NVLink / PCIe topology-aware placement so transfers follow the
    fast links; overlapping GPU compute with the SSD->RAM->VRAM prefetch and
    streaming the engine already does; and multi-GPU KV placement so the paged
    KV cache can be split across devices. All future work.
  - [ ] **Enterprise serving concerns (named as goals, not built).** High
    throughput with batching and concurrent requests; the planned
    OpenAI-compatible HTTP server front-end as the serving entry point (already
    an open item below - this track connects to it); observability (metrics for
    residency, streamed bytes, VRAM/RAM occupancy, per-request latency); and
    running the largest models (100B-1T) FAST when the hardware is present. None
    of these exist yet; they define the enterprise target, not current state.

  This track does NOT change the no-GPU mission. It is an additional tier for
  enterprise deployment on GPU hardware, layered on the same engine, and
  validated only where real GPUs exist.
- [ ] **More architectures.** Only llama-arch MoE and dense llama are supported.
  Qwen-MoE (shared expert + per-expert gate), DeepSeek-style (expert groups,
  shared experts, sigmoid gating with group-limited top-k), and others are each
  a new oracle-gated graph builder per the arch-descriptor plan
  (`ENGINE_CORE_DESIGN.md` section 6.4).
- [x] **Async prefetch overlap (DONE).** A dedicated background I/O worker
  thread now overlaps the next layer's predicted-expert disk reads with the
  current layer's ggml compute, so read latency hides behind compute on the
  bounded-RAM streaming path. The synchronous `ensure_layer_experts_resident`
  stays the AUTHORITATIVE load: a mispredicted or still-in-flight prefetch is a
  plain miss it corrects inline, so decode is byte-identical to the libllama
  oracle including under bounded-pool eviction (the oracle gate's async sub-test
  shows async-vs-sync `|delta| = 0.000e+00`, identical across repeated runs that
  perturb prefetch timing). The ggml compute path stays single-threaded (only an
  I/O thread was added); the worker never touches the non-thread-safe
  `SlotPool`, reading into its own staging buffer and handing completed bytes to
  the compute thread to install, which keeps it thread-safe/TSan-clean. A
  default-ON runtime toggle (`sf_context_params.async_prefetch`, the CLI
  `--async-prefetch on|off`, env `STRATAFLOW_ENGINE_ASYNC_PREFETCH`) lets a
  misbehaving platform fall back to the synchronous warm path and lets the bench
  contrast on vs off. An overlap metric (`completed-before-use`: prefetches the
  worker finished and installed before the layer needed them) is surfaced
  through `sf_runtime_stats` and the CLI stats line. In-sandbox the MECHANISM is
  proven (`completed-before-use` goes from 0 with async off to > 0 with async
  on, byte-identical output); absolute wall-clock speedup is hardware-dependent
  and re-verifiable on Colab/real-Mixtral (see `colab/README.md`). Per-layer
  graph-build reuse across tokens (`ENGINE_CORE_DESIGN.md` section 3.5) remains
  a separate deferred item. FOLLOW-UP (see "Layer-stratified residency +
  prefetch gating" above): async prefetch is now GATED so it never increases
  bytes streamed - a completed prefetch installs only into a free per-layer slot
  and the worker self-backs-off when overlap stays ~0 - which reverses the
  earlier bounded-slot regression where async-on streamed MORE than async-off.

### From the original PLAN.md (Phases 4-7), as written

- [x] **Phase 4 - predictor + pipelined prefetch.** Statistical predictor done
  (EC-6); background-I/O async overlap now DONE too (see "Async prefetch overlap"
  above): a background thread loads layer N+1's predicted experts while layer N
  computes, byte-identical to the oracle, with a default-ON toggle and an
  overlap metric. Per-layer graph-build reuse across tokens remains deferred.
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
   `stats:` line reporting resident model-weight bytes and peak RSS. Expert
   residency is now AUTO-sized per model from the free-RAM budget + the model's
   expert layout by default (`--cache-gb` caps it, `--expert-slots` overrides);
   no hand-tuning needed.
3. **Run a real, quantized pretrained MoE** (an actual downloaded model, not a
   generated fixture) through the `.strata` streaming path and record real tok/s
   and peak RAM. This is now unblocked by (1) and (2) and is the key remaining
   proof point.

The mechanism is proven on a generated quantized model; the open work is running
an actual downloaded model and broadening quant-family coverage.

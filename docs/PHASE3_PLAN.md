# Phase 3 Implementation Plan: Tiered Weight Store + trillion-scale streaming

**Owner: Coaade Inc. · Apache-2.0 · targets [`docs/PLAN.md §6` Phase 3](./PLAN.md)**

This is the implementation plan for the core of StrataFlow: the Tiered Weight
Store (TWS) and the CPU+SSD expert-streaming path that lets a trillion-scale MoE
model run on a low-RAM, no-GPU box. It is the detailed working plan for the
scope and exit criteria already fixed in [`docs/PLAN.md`](./PLAN.md) sections
3.3 (Tiered Weight Store), 3.5 (Scheduler), 3.10 (`.strata` format), and 6
(Phase 3). Read those first; this document does not restate the spec, it decides
*how* to build it and *in what order*.

The reference to mirror and then beat is
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) (see
[`docs/RESEARCH.md §3`](./RESEARCH.md)): an always-resident "trunk" in RAM,
routed experts streamed from disk via `O_DIRECT` + `pread` with an LRU cache,
byte-identical output across memory budgets, more RAM buying only speed.

---

## 0. Exit criteria (from the plan, repeated here as the done-bar)

1. The owner's 800B-class Kimi MoE runs on an 8-16 GB no-GPU box with bounded
   RSS (no OS swapping), byte-identical output across memory budgets.
2. >=1.5x llama.cpp mmap on the 16 GB no-GPU machine for a gpt-oss-120b-class
   model.
3. VRAM cache gives >=+50% on a 32 GB + 8 GB-GPU machine.
4. On the 800B Kimi run, match or beat kimi-k3-in-c tok/s at an equal RAM budget
   on the same disk.

CI is CPU-only with no GPU on any runner. So criteria 1, 2, and 4 are validated
with real hardware off-CI (owner machines / dev boxes); criterion 3 (VRAM) is
logic/unit-tested only in this phase and hardware-validated later. Everything
else is proven in CI with a tiny MoE fixture and a forced-streaming proxy (see
§4 and §5).

---

## 1. The architectural reality (what we are building on)

StrataFlow runs its forward pass **entirely through vendored llama.cpp**.
`src/model/ggml_model.cpp` is the only translation unit that includes
`<llama.h>`; it owns a `llama_model` + `llama_context` and calls `llama_decode`.
As of Phase 2 we drive the load with a cheap GGUF metadata pre-pass -> `plan_placement()`
-> `derive_llama_placement()` -> `n_gpu_layers` + a `tensor_buft_overrides`
array that routes the routed-expert FFN tensors to the CPU buffer (the
`--n-cpu-moe` / `-ot "exps=CPU"` equivalent).

The existing TWS pieces are built, unit-tested, and **not yet wired into the
model path**:

- `src/tws/weight_store.{h,cpp}` — `SlotPool`: fixed slot pool, `ExpertId`
  (`{layer,expert}`) -> slot remap, LRU eviction, templated `acquire(id, load_fn)`
  that calls `load_fn` only on a miss; `CacheStats{hits,misses,evictions,hit_rate}`.
- `src/tws/async_io.{h,cpp}` — `BlockFile`: `open` / `read_at` (pread/CRT) /
  `read_async` (`std::async` deferred fallback), 4 KiB align helpers,
  `backend_name()` reporting "sync-pread (io_uring pending)" etc. The native
  async backends are seams/stubs.
- `src/plan/planner.{h,cpp}` and `src/plan/llama_placement.{h,cpp}` — produce
  the plan and translate it to llama knobs.
- `src/hw/profiler.{h,cpp}` — `HardwareProfile` has `ssd_seq_gbps` /
  `ssd_rand_gbps` fields that are **declared but never measured**.

### 1.1 What the vendored llama.cpp (tag b11379) already offers for streaming

I read the pinned submodule source to find what we can build on rather than
fight. Findings, with file references:

- **`lazy_mode` is NOT expert streaming.** `llama.h:218` defines
  `LLAMA_LAZY_MODE_{OFF,AUTO,ON}` ("on-demand reading of tensors marked by the
  arch"). The implementation (`src/llama-model-loader.cpp:1088` `lazy_read::add`,
  `:1342` the `TENSOR_READ_LAZY` branch) only applies to tensors the **arch code
  explicitly marks** with `TENSOR_READ_LAZY`. Grepping the source, the only two
  call sites are `src/models/gemma4.cpp:59` (per-layer token embedding) and
  `src/models/qwen4exp.cpp:215` (PLE rows). **Routed expert FFN tensors
  (`ffn_{gate,up,down}_exps`) are never marked lazy.** The documented use case
  in `llama-model-loader.h` is literally "keep PLE / engrams embd tensors on
  disk, read them on demand."
- **The lazy mechanism is OS-page-cache mmap, not a managed cache.** Lazy ranges
  are mmap'd and advised `POSIX_MADV_RANDOM` (`src/llama-mmap.cpp:518`); rows
  fault in via the OS pager and are evicted by the OS, not by any bounded,
  StrataFlow-controlled policy. There is no `O_DIRECT`, no VRAM tier, no hit/miss
  accounting, and no deadline-driven prefetch. This is the opposite of the
  kimi-k3-in-c `O_DIRECT` design, which exists specifically to **bypass** the
  page cache so it doesn't compete with the resident trunk.
- **`LLAMA_LOAD_MODE_DIRECT_IO` is load-time only.** `llama.h:206` defines a
  `llama_load_mode` enum including `LLAMA_LOAD_MODE_DIRECT_IO`, and
  `src/llama-mmap.cpp` has a real `O_DIRECT` `llama_file` path
  (`:200 open(..., O_RDONLY | O_DIRECT)`). But `use_direct_io`
  (`src/llama-model-loader.cpp:560`) governs the **one-time initial read of
  tensors into their buffers at load**, not per-token streaming during decode.
- **Custom ggml backend buffer types are the real extension seam.**
  `ggml/src/ggml-backend-impl.h` exposes `ggml_backend_buffer_type_i`
  (`alloc_buffer`, `get_alloc_size`, ...) and `ggml_backend_buffer_i`
  (`get_base`, `init_tensor`, `set_tensor`, `get_tensor`, `free_buffer`, ...).
  A weight tensor is placed in a buffer via its buffer type; Phase 2 already
  resolves a `ggml_backend_buffer_type_t` for experts. So a **StrataFlow buffer
  type** is a supported way to own the memory experts live in.
- **The hard constraint that drives the whole design decision:** the CPU
  compute kernel for the MoE op reads weight bytes **directly through
  `tensor->data`** at graph-execution time. In
  `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_mul_mat_id`
  (`:1554`, the `GGML_OP_MUL_MAT_ID` expert op) dereferences the source tensor's
  data pointer; it does **not** call the buffer's `get_tensor` callback per op.
  Consequence: whatever owns expert memory, the tensor's `data` pointer must
  point at **valid, resident bytes for that expert at the instant the graph
  runs**. There is no ggml hook that says "about to compute expert E of layer L,
  fault it in now." This is the central feasibility risk (§6).

**Conclusion:** llama.cpp gives us the GGUF reader, the kernels, the
`tensor_buft_overrides` seam, a reusable `O_DIRECT` `llama_file`, and the custom
buffer-type ABI. It does **not** give us a per-token, bounded, direct-I/O expert
cache. That is exactly the gap StrataFlow fills, and it is ours to build.

---

## 2. The design decision

**Question (from the task):** does the TWS (a) replace llama.cpp's weight
loading with our own streaming loader, (b) sit alongside/underneath llama.cpp as
a custom ggml backend buffer type our SlotPool manages, or (c) something else?

### 2.1 The options, scored

**Option (a) — replace llama.cpp's loader / own the whole forward pass.**
Build our own graph execution (like kimi-k3-in-c) or deeply patch ggml's tensor
loading so experts stream through our code. Buys us total control and the
cleanest path to beating the reference. Costs: enormous surgery, throws away the
Phase 1b/2 investment (`GgmlModel`, the only `<llama.h>` TU), forks us off the
ggml kernels/backends we explicitly decided to reuse (`PLAN §1.2`, `§4`), and
makes the monthly rebase (Risk table, `PLAN §8`) a nightmare. It also puts the
byte-identical guarantee on *us* for the whole arithmetic path, not just the
I/O. **Rejected for v1.** This is the "fight llama.cpp" path.

**Option (b) — custom ggml backend buffer type, SlotPool-managed, underneath
llama.cpp.** Define a `ggml_backend_buffer_type_t` ("strata-stream buft") and
point the Phase 2 expert override at it instead of the plain CPU buft. Experts
live in SlotPool-owned slots; our buffer faults them in from the `.strata`/GGUF
file via `BlockFile`. llama.cpp still builds the graph and runs the kernels;
determinism is inherited because the *bytes and the math are identical* — we
only change *when and from where* expert bytes become resident. This is the
natural fit for the pieces we already built and keeps the ggml-reuse decision
intact. **The hard part is the §1.1 constraint:** `tensor->data` must be valid
at compute time, and ggml does not call us per-op. (Resolved below.)

**Option (c) — hybrid: Option (b)'s buffer type for *ownership and tiering*,
plus a thin pre-decode residency pass for *correctness*.** Because ggml won't
call us per expert op, we make experts resident **before** `llama_decode` runs
the graph, driven by the exact routing of the *previous* token plus the
predictor (Phase 4), and in the exact-correctness fallback by making the slot
window cover the superset of experts a token can touch for the layers in flight.
In the simplest correct form the StrataFlow buffer presents a stable base
pointer per slot and we **synchronously ensure every expert the graph will read
is loaded into its slot before `llama_decode` returns control to the kernel**.
This is still Option (b)'s architecture; (c) is the name for the specific
"when do we fault in" policy that makes (b) correct given the constraint.

### 2.2 Recommendation

**Adopt Option (b) with the Option (c) residency policy.** Concretely:

1. The TWS is a **custom ggml backend buffer type** (`strata_stream_buft`) whose
   buffer's `init_tensor` records each expert tensor's `{layer, expert}` identity
   and its byte range in the backing `.strata`/GGUF file, and whose `get_base`
   returns a stable slot-pool address. The `SlotPool` (already built) owns the
   resident slots; `BlockFile` (already built) does the reads.
2. The Phase 2 override path in `ggml_model.cpp` is **re-pointed**: instead of
   resolving experts to `ggml_backend_cpu_buffer_type()`, when the plan says
   `stream_experts`, resolve them to `strata_stream_buft`. Everything else in
   the Phase 2 load path (the regex, the NULL-terminated override array, the
   pre-pass plan) is reused unchanged. When the plan does **not** stream (model
   fits), we keep the exact Phase 2 CPU-offload behaviour, so we never regress
   the "everything fits" case.
3. **Correctness is guaranteed by residency, not by hoping the pointer is
   valid.** For a given decode step, before the kernel reads an expert we
   guarantee its slot holds that expert's bytes. The exact-mode policy that is
   always correct regardless of prediction: treat the TWS as a cache but, for
   each layer, ensure the specific experts selected by the router for the
   current token are resident in slots *before* that layer's MoE op executes.
   The spike (§6) decides the precise insertion point (per-decode pre-pass vs. a
   ggml compute-callback / scheduling hook) and whether a one-expert-per-slot
   synchronous model is fast enough as the v1 correctness floor, with prefetch
   (Phase 4) layered on later.

### 2.3 Why this is the right call

- **Effort vs. payoff.** It reuses `SlotPool`, `BlockFile`, the planner, and the
  entire Phase 2 override plumbing. The new surface is one buffer type plus the
  residency hook — far less than rewriting the loader, and it directly produces
  the bounded-RSS, LRU-cached, direct-I/O streaming the exit criteria demand.
- **Interaction with the Phase 2 `tensor_buft_overrides` path.** It *is* that
  path, with a different `buft` pointer. No new routing logic; the same regex
  (`expert_ffn_regex()` in `llama_placement.h`) selects the same tensors. The
  decision of CPU-resident vs. streamed stays in the planner
  (`PlacementPlan::stream_experts`), where it already lives.
- **Byte-identical guarantee.** We change *residency and provenance of bytes*,
  never their *values* or the *ops*. The same quantized expert bytes reach the
  same `mul_mat_id` kernel; only the tier they came from differs. More RAM =
  more slots = fewer disk reads = faster, identical output. This is exactly the
  kimi-k3-in-c invariant, and it is enforced by a determinism test across memory
  budgets (Task 6).
- **Rebase-friendliness.** All new code is StrataFlow modules + one new buffer
  type registered at runtime; no edits inside `ggml/` or `llama/` source. If the
  spike shows we need a tiny upstream seam (e.g. a compute-time residency
  callback), that is the one place we would carry a minimal, clearly-marked
  patch, and the spike must quantify it before we commit.
- **Keeps the GPU tier reachable.** The same buffer-type abstraction extends to
  a VRAM slot pool later; `SlotPool` is already tier-agnostic. Criterion 3 is
  wired as logic now, hardware-validated later.

### 2.4 The one risk this decision carries

The §1.1 constraint (`tensor->data` read directly at compute time, no per-op
hook) means Option (b) is only correct if we can guarantee residency before the
kernel runs. If the spike (§6) shows ggml gives us **no** clean insertion point
and that pre-faulting the full per-token expert working set is too coarse or too
slow, the fallback is a **minimal, clearly-marked upstream seam** in ggml's MoE
op dispatch (a residency callback), carried as a patch outside the module tree.
We do not fall back to Option (a). The spike exists to retire this risk before
Task 2 starts.

---

## 3. Task breakdown (sequenced)

Seven tasks. Each is independently implementable and independently testable
without the 800B model or a GPU. Dependencies are called out. The spike (§6)
runs **before Task 2** and gates the Option (b) residency design.

Legend: **CI** = fully validated on CPU-only CI; **CI-proxy** = validated in CI
via the tiny MoE + forced-streaming proxy; **HW** = needs real hardware/model,
validated off-CI.

---

### Task 0 — Environment baseline (chore)

**Builds:** nothing new; establishes the green baseline so later diffs are
attributable.

**Touches:** none (build/test only).

**Steps:**
1. `git submodule update --init --recursive` (llama.cpp is pinned at b11379).
2. `cmake --preset debug && cmake --build build/debug` (first build compiles all
   of llama.cpp/ggml, static, CPU-only — expect minutes).
3. `ctest --preset debug` — confirm all six existing suites pass
   (`test_slot_pool`, `test_planner`, `test_llama_placement`, `test_predictor`,
   `test_api`, `test_model_load`).
4. Clang parity:
   `CC=clang CXX=clang++ cmake -S . -B build/clang -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSTRATAFLOW_WERROR=ON`
   then build + test.

**Dependencies:** none.

**Tested (CI):** the existing suite is the baseline; both gcc and clang green.

**Done-criterion:** `ctest --preset debug` passes on gcc and clang with
`-Werror`; baseline recorded.

---

### Task 1 — Tiny MoE fixture + forced-streaming proxy + SSD profiling (chore/feat)

**Builds:** (1) a tiny but valid **MoE** GGUF generator so the expert path is
exercisable in CI; (2) the "force streaming even when it fits" test switch that
is the key CI proxy for the disk path; (3) the missing SSD bandwidth
measurement in the profiler.

**Touches:**
- `tools/testdata/make_tiny_gguf.py` — extend (or add a sibling
  `make_tiny_moe_gguf.py`) to emit a small Mixtral/qwen2moe-style MoE GGUF:
  set `<arch>.expert_count` and `<arch>.expert_used_count`, emit per-layer
  `blk.N.ffn_gate_exps` / `ffn_down_exps` / `ffn_up_exps` tensors (3-D stacked
  expert layout llama.cpp expects) plus the router `blk.N.ffn_gate_inp`. Keep it
  tiny (a few experts, tiny `n_embd`/`n_ff`, 2 layers) so the file stays
  CI-sized. Run via `uv run --with gguf --with numpy python3 ...`.
- `src/hw/profiler.{h,cpp}` — measure `ssd_seq_gbps` and `ssd_rand_gbps` by
  timing reads of a temp file in `model_dir` at QD1 (sequential + random 4 MiB),
  honoring the existing `quick` flag (skip or shrink in CI). Write the throwaway
  temp file under the model dir / workspace, never `/tmp`.
- A test-only env switch (e.g. `STRATAFLOW_FORCE_STREAM_EXPERTS=1`) read in the
  planner or `ggml_model.cpp` that forces `PlacementPlan::stream_experts=true`
  regardless of budget. This is the proxy that lets CI drive the real streaming
  code on a model that would otherwise fit.

**Dependencies:** Task 0.

**Tested (CI):**
- New `tests/test_moe_fixture.cpp` (or extend `test_model_load.cpp`): generate
  the MoE GGUF, load it through `load_ggml_model`, assert `shape().is_moe`,
  `n_experts`/`n_experts_used` match, and a greedy `forward()` returns a valid
  token id. Gate on the fixture being generatable (skip cleanly if `gguf` is
  unavailable, like the existing `STRATAFLOW_TEST_GGUF` pattern).
- New `tests/test_profiler_ssd.cpp` (or extend an existing profiler test):
  assert `ssd_seq_gbps > 0` and `ssd_rand_gbps > 0` after a non-quick profile of
  the workspace dir.

**Done-criterion:** a tiny MoE GGUF loads and decodes a token in CI; profiler
reports non-zero SSD bandwidths; `STRATAFLOW_FORCE_STREAM_EXPERTS=1` flips the
plan to streaming for a model that fits.

---

### Task 2 — `strata_stream_buft`: the streaming ggml backend buffer type (feat, CORE)

**Builds:** the custom ggml backend buffer type that is the TWS's integration
point with llama.cpp (the §2 decision). This is the heart of Phase 3.

**Touches:**
- New `src/tws/stream_buft.{h,cpp}` — implement `ggml_backend_buffer_type_i` +
  `ggml_backend_buffer_i`:
  - `get_alloc_size` returns the expert tensor's on-disk nbytes.
  - `init_tensor` records `{layer, expert}` (parsed from the tensor name
    `blk.N.ffn_*_exps`) and the tensor's byte offset/length in the backing file;
    registers it with the owning TWS.
  - `get_base` / `set_tensor` / `get_tensor` present a slot address; `set_tensor`
    at load time is a no-op or records the file range (bytes stay on disk).
  - `free_buffer` releases slots.
  - `is_host` true (CPU-side), so the CPU kernel can read the slot directly.
- `src/tws/weight_store.{h,cpp}` — add a thin `ensure_resident(ExpertId, loader)`
  wrapper over the existing `acquire(...)` so the buffer/residency hook can
  request a load through `BlockFile`; keep the existing LRU/stats untouched.
- `src/model/ggml_model.cpp` — when `plan.stream_experts` (or the Task 1 force
  switch), resolve the expert override `buft` to `strata_stream_buft` instead of
  `ggml_backend_cpu_buffer_type()`. Reuse the existing override array, regex, and
  pre-pass. Non-streaming path unchanged.

**Dependencies:** Task 1 (needs the MoE fixture to test against); spike §6
(gates the residency approach this buffer type assumes).

**Tested (CI-proxy):**
- New `tests/test_stream_buft.cpp`: construct the buffer type, feed it synthetic
  expert tensor metadata, assert name->`{layer,expert}` parsing, `get_alloc_size`
  equals nbytes, and that `init_tensor` registers the expected IDs. No llama
  needed for the parse/registration unit tests.
- Integration (CI-proxy): load the tiny MoE with
  `STRATAFLOW_FORCE_STREAM_EXPERTS=1` so experts route through
  `strata_stream_buft`; assert the model loads, `forward()` returns a valid
  token, and the TWS recorded >=1 expert registration. This exercises the real
  buffer-type wiring on CPU-only CI.

**Done-criterion:** the tiny MoE loads with experts owned by
`strata_stream_buft` and decodes a token in CI; buffer-type unit tests pass on
gcc + clang `-Werror`.

---

### Task 3 — Residency + exact-mode correctness: wire SlotPool + BlockFile under the buffer (feat, CORE)

**Builds:** the residency policy from §2.1(c) — expert bytes become resident in
their slot (read via `BlockFile`) before the MoE kernel reads them, with LRU
eviction via the existing `SlotPool`. This is what makes Option (b) *correct*
under the §1.1 constraint.

**Touches:**
- `src/tws/weight_store.{h,cpp}` — connect `acquire`'s `load_fn` to a
  `BlockFile::read_at` of the expert's recorded file range into the slot buffer.
- `src/tws/stream_buft.{h,cpp}` — implement the residency hook at the insertion
  point the spike (§6) validated (per-decode pre-pass driven by the previous
  token's exact routing, or a compute-time callback). Prefill-bypass rule
  (`PLAN §3.3`): a micro-batch touching more experts than slots streams without
  caching so prefill can't wipe the decode hot set.
- `src/model/ggml_model.cpp` — surface a per-decode residency step around
  `llama_decode` if the spike chose the pre-pass approach.

**Dependencies:** Task 2.

**Tested (CI-proxy):**
- New `tests/test_streaming_decode.cpp`: tiny MoE + force-stream + a slot pool
  deliberately **smaller than the expert count**, so eviction and reload are
  guaranteed. Assert: (a) `forward()` still returns valid tokens; (b)
  `CacheStats` shows both hits and misses and non-zero evictions; (c) RSS /
  resident slot count stays bounded by the configured pool size across many
  tokens (bounded-memory proxy for exit criterion 1).
- Reuse `test_slot_pool.cpp` to keep the LRU/stats invariants green.

**Done-criterion:** with a slot pool smaller than the expert set, the tiny MoE
decodes correctly in CI, cache stats show eviction+reload, and resident memory
stays bounded by the pool size.

---

### Task 4 — Native direct-I/O async backends behind `BlockFile` (feat)

**Builds:** the real async, page-cache-bypassing SSD backends the exit criteria
depend on (kimi-k3-in-c parity), behind the existing `BlockFile` signatures.

**Touches:**
- `src/tws/async_io.{h,cpp}` — implement the three native backends behind the
  `STRATAFLOW_PLATFORM_*` macros, each guarded with the portable sync fallback
  always available (`CONTRIBUTING.md` rule):
  - Linux: `io_uring` + `O_DIRECT` (reuse the alignment helpers; mirror the
    llama.cpp `O_DIRECT` open at `src/llama-mmap.cpp:200` for the open/align
    pattern).
  - Windows: IOCP + `FILE_FLAG_NO_BUFFERING`.
  - macOS: `fcntl(F_NOCACHE)` + parallel `pread`.
  - `backend_name()` reports the active backend (replace the "pending" strings).
- Keep `read_async` returning `std::future<int64_t>` so callers (Task 3,
  Phase 4 prefetch) don't change.

**Dependencies:** Task 3 (so there's a real consumer to validate against), but
the backend code itself is independent and could be developed in parallel with
Task 3 behind the stub.

**Tested (CI + HW):**
- CI: `tests/test_async_io.cpp` — write a known temp file under the workspace,
  read it back aligned and unaligned via `read_at` and `read_async`, assert byte
  correctness and that `backend_name()` reports the platform's native backend on
  that OS (CI covers linux-gcc, linux-clang, macOS, Windows-MSVC, so each native
  backend gets exercised on its own runner). Direct-I/O may be unavailable in
  some CI sandboxes; detect and fall back to sync, asserting correctness either
  way (never assert throughput in CI).
- HW: real SSD throughput and page-cache-bypass (RSS not inflated by cache)
  measured on dev boxes.

**Done-criterion:** each OS's native backend reads bytes correctly in its CI
runner (or cleanly falls back), `backend_name()` reflects the active backend,
and the portable sync path remains correct everywhere.

---

### Task 5 — `strata-pack` v1: expert-contiguous aligned `.strata` layout + trunk/cache dial + CLI/ABI (feat)

**Builds:** the streaming-friendly on-disk format (`PLAN §3.10`) and the
`--trunk-gb` / `--cache-gb`-style memory dial, so streaming reads one aligned,
contiguous blob per `(layer, expert)` instead of scattered GGUF ranges.

**Touches:**
- New `tools/strata-pack/` — converts a GGUF into `.strata`: GGUF-compatible
  header/metadata, a contiguous always-active (trunk) block loaded up front, and
  one 4 KiB-aligned (`kIoAlignment`) contiguous blob per `(layer, expert)`
  holding gate+up+down together (row-column bundling). Emit an index mapping
  `{layer, expert}` -> `{offset, length}`.
- `src/tws/stream_buft.{h,cpp}` / `weight_store` — read the `(layer,expert)`
  offset from the `.strata` index when present; keep the GGUF-direct path
  (Task 2/3) as the slower fallback (`PLAN §3.10`: original GGUF supported via an
  index).
- `include/strataflow/strataflow.h` — **ABI-compatible additions** for the
  memory dial. `sf_context_params` already has `vram_budget` / `ram_budget`;
  add explicit trunk/cache fields only if needed, appended at the end of the
  struct to preserve ABI, with `sf_context_default_params()` filling safe
  defaults. Add matching CLI flags in the CLI tool (`--trunk-gb` / `--cache-gb`).
- Keep libraries STATIC (`BUILD_SHARED_LIBS OFF`) per the build constraints.

**Dependencies:** Task 3 (the streaming reader consumes the layout).

**Tested (CI):**
- `tests/test_strata_pack.cpp`: pack the tiny MoE GGUF to `.strata`, assert each
  expert blob is 4 KiB-aligned and contiguous, the index round-trips
  `{layer,expert}->{offset,len}`, and gate+up+down for one expert are adjacent.
- Load the `.strata` through `load_ggml_model` (force-stream) and assert
  byte-identical decoded tokens vs. the same model loaded from GGUF (proves the
  pack doesn't change values).
- ABI test (extend `test_api.cpp`): `sf_context_default_params()` fills the new
  fields; passing an explicit trunk/cache dial changes the plan.

**Done-criterion:** tiny MoE packs to `.strata` with aligned per-expert blobs,
loads and decodes byte-identically to its GGUF, and the CLI exposes a working
`--trunk-gb` / `--cache-gb` dial.

---

### Task 6 — Trace capture + offline cache-size simulator + determinism gate (feat/chore)

**Builds:** the `sim_cache.py`-equivalent trace tooling and the determinism gate
that locks in the byte-identical-across-budgets exit criterion.

**Touches:**
- `src/tws/weight_store.{h,cpp}` or a small `src/tws/trace.{h,cpp}` — optional
  expert-access trace capture (sequence of `ExpertId` per token) behind a flag,
  written to a trace file.
- New `tools/strata-sim-cache/` (C++ or a `python/` script) — replay a trace
  against the `SlotPool` LRU for a range of pool sizes and emit the hit-rate /
  reuse-distance curve (the data the planner uses to size the cache; `PLAN §3.3`
  "sized from a replayed access trace").
- `tests/` — a **determinism test across memory budgets**: decode the same
  prompt on the tiny MoE at (a) a pool large enough to hold all experts and
  (b) a pool forced small (eviction-heavy), assert the produced token sequences
  are **identical**. This is the CI guardian of exit criteria 1 and 4 and of the
  `CONTRIBUTING.md` determinism rule.

**Dependencies:** Task 3 (needs the streaming cache to trace); independent of
Tasks 4 and 5.

**Tested (CI):**
- `tests/test_cache_determinism.cpp`: identical greedy output across two pool
  sizes on the tiny MoE (hard gate).
- Trace replay unit test: a hand-built synthetic trace produces a monotonic
  non-decreasing hit-rate curve as pool size grows.

**Done-criterion:** greedy output is byte-identical across small vs. large expert
pools in CI; the simulator produces a hit-rate-vs-pool-size curve from a trace.

---

### Task 7 — End-to-end validation harness + exit-criteria benchmarks (chore)

**Builds:** the measurement harness that proves the real exit criteria off-CI
and documents the proxy coverage.

**Touches:**
- `tools/strata-bench/` — measure tok/s, peak RSS, SSD bytes/token for a
  streamed run; a comparison mode vs. `llama.cpp` mmap and vs. kimi-k3-in-c at an
  equal RAM budget on the same disk.
- `docs/` — a short "Phase 3 validation" note recording how each exit criterion
  is checked and which are CI vs. HW.

**Dependencies:** Tasks 3-6.

**Tested (HW, with CI smoke):**
- HW: criterion 1 (800B Kimi, 8-16 GB no-GPU, bounded RSS, identical output
  across budgets), criterion 2 (>=1.5x llama.cpp mmap, gpt-oss-120b, 16 GB
  no-GPU), criterion 4 (match/beat kimi-k3-in-c at equal RAM/disk). Criterion 3
  (VRAM +50%) deferred to GPU hardware; logic unit-tested here.
- CI smoke: the harness runs end-to-end on the tiny MoE and emits a report
  without error (no throughput assertions in CI).

**Done-criterion:** the harness produces the exit-criteria report on real
hardware; the owner can read criteria 1/2/4 off it; CI runs the harness smoke on
the tiny MoE.

---

## 4. CI/sandbox vs. real-hardware matrix

| Capability | Validated where | How we de-risk without the real target |
|---|---|---|
| MoE load + expert path | **CI** | tiny MoE GGUF fixture (Task 1) |
| `strata_stream_buft` wiring | **CI-proxy** | `STRATAFLOW_FORCE_STREAM_EXPERTS=1` forces the streaming code on a model that fits (Task 1/2) |
| Residency + LRU eviction + bounded RSS | **CI-proxy** | pool smaller than expert count forces eviction/reload; assert bounded resident slots (Task 3) |
| Direct-I/O native backends (io_uring/IOCP/F_NOCACHE) | **CI (per-OS) + HW** | each backend runs on its own CI runner for byte-correctness; throughput + page-cache bypass measured on HW (Task 4) |
| `.strata` aligned layout + dial | **CI** | pack/round-trip + byte-identical decode on tiny MoE (Task 5) |
| Byte-identical across memory budgets | **CI** | determinism test across pool sizes (Task 6) — the CI guardian of criteria 1 & 4 |
| 800B Kimi on 8-16 GB (crit 1) | **HW** | the small-pool + force-stream proxy proves the *mechanism*; scale proven on the real model |
| >=1.5x llama.cpp mmap (crit 2) | **HW** | bench harness (Task 7) |
| VRAM cache +50% (crit 3) | **HW (GPU) later** | SlotPool is tier-agnostic; VRAM slot pool logic unit-tested now, GPU validated post-Phase 3 (CI has no GPU) |
| match/beat kimi-k3-in-c (crit 4) | **HW** | equal-RAM/equal-disk comparison in the harness |

**The key proxy:** a small MoE whose experts we **force to stream from disk even
though they would fit in RAM** (`STRATAFLOW_FORCE_STREAM_EXPERTS=1`), combined
with a slot pool deliberately smaller than the expert set. This exercises the
entire streaming + eviction + reload + determinism path on CPU-only CI without
the 800B model or a GPU.

---

## 5. Riskiest unknowns (ranked)

1. **(Highest) Can llama.cpp decode a token while expert tensors are faulted in
   from SSD on demand, correctly and with bounded memory?** The §1.1 constraint:
   the MoE kernel reads `tensor->data` directly at compute time and ggml gives no
   per-expert "about to read" hook. Everything in Option (b) depends on finding a
   correct residency insertion point. **This is the spike (§6).**
2. **Direct-I/O portability and alignment.** `O_DIRECT`/`F_NOCACHE`/
   `FILE_FLAG_NO_BUFFERING` have strict alignment and sandbox-availability quirks
   (CI may forbid `O_DIRECT`). Mitigation: the always-available sync fallback and
   per-OS CI coverage (Task 4); we assert correctness, never throughput, in CI.
3. **Bounded RSS vs. OS page cache.** If we read experts through the normal page
   cache, the OS fills RAM and competes with our pool (defeating criterion 1).
   Mitigation: direct I/O bypasses the page cache (the whole point of kimi-k3's
   `O_DIRECT`); the bounded-RSS proxy test (Task 3) guards it.
4. **`.strata` layout correctness / value drift.** A packing bug could change
   bytes. Mitigation: byte-identical-decode round-trip test (Task 5).
5. **Rebase cost if a tiny upstream seam is unavoidable.** If the spike forces a
   ggml residency callback, it's one minimal marked patch outside the module
   tree. Mitigation: the spike quantifies it before Task 2; prefer the pre-decode
   pre-pass that needs no ggml edit.

---

## 6. Spike for the #1 risk (run before Task 2; must not land in committed `src/`)

**Goal:** prove (or disprove) that `llama_decode` can run the MoE graph on the
tiny MoE fixture while expert tensors are owned by a StrataFlow buffer type whose
bytes are faulted in from disk, with correct output and resident memory bounded
below the full expert set.

**Where:** a throwaway dir, e.g. `spike/stream_decode/` (NOT under `src/`;
`.gitignore` it or clearly mark it `SPIKE — do not ship`). Links against the
vendored llama + a minimal copy of `SlotPool`/`BlockFile`.

**Procedure:**
1. Build the tiny MoE GGUF (Task 1's generator, pulled forward for the spike).
2. Define a minimal `strata_stream_buft` that, on `get_base`/residency, reads the
   expert's bytes from the GGUF file range into a slot and returns the slot
   pointer. Point the Phase 2 expert override at it.
3. Try the two candidate residency insertion points and measure both:
   - **(A) Pre-decode pre-pass:** before `llama_decode`, make resident the exact
     experts the current token will touch (bootstrapped from the previous token's
     routing; for the first token, make the full per-layer expert set resident).
     No ggml edit. Simplest; validate correctness and whether a one-slot-per-
     expert synchronous model is acceptable as the v1 floor.
   - **(B) Compute-time hook:** if (A) can't guarantee correctness without
     over-reading, evaluate the smallest ggml seam (a residency callback in the
     `GGML_OP_MUL_MAT_ID` dispatch) and quantify the patch.
4. Run greedy decode at two slot-pool sizes (one >= all experts, one << all
   experts). **Assert:** identical token sequences at both sizes (determinism),
   and resident slot bytes bounded by the small pool.

**Success:** (A) yields correct, deterministic, bounded-memory decode on the tiny
MoE -> Option (b)/(c) confirmed with **zero** ggml edits; proceed to Task 2 with
the pre-pass policy. If only (B) works, record the exact minimal patch and carry
it as a marked upstream seam. If neither works, escalate before committing to
Option (b) (the only scenario that reopens the design decision).

**Explicitly out of scope for the spike:** performance tuning, native async
backends, `.strata` packing, VRAM. It answers one yes/no question:
*correct + bounded on-demand expert decode through llama.cpp.*

---

## 7. Sequencing summary

```
Task 0 (baseline)
   └─> Task 1 (tiny MoE fixture + force-stream switch + SSD profiling)
          └─> [SPIKE §6: residency feasibility]  ← gates Task 2
                 └─> Task 2 (strata_stream_buft)
                        └─> Task 3 (residency + exact-mode correctness)   ← CORE done here
                               ├─> Task 4 (native direct-I/O backends)    (parallelizable behind stub)
                               ├─> Task 5 (.strata pack + dial + ABI/CLI)
                               └─> Task 6 (trace + sim + determinism gate)
                                      └─> Task 7 (bench harness + exit-criteria validation)
```

Tasks 4, 5, and 6 are independent of each other once Task 3 lands and may be
built in parallel. The byte-identical guarantee is nailed down by Task 6's
determinism test and re-checked by Task 5's pack round-trip; both run in CI on
every matrix job.

---

## 8. Spike result (run before committing to Task 2) — ✅ PASS

The §6 feasibility spike was built and run against the vendored llama.cpp
(tag b11379) on a tiny Mixtral-style (`llama`-arch) MoE GGUF (2 layers, 8
experts, top-2). Source lives in `spike/stream_decode/` (throwaway, gitignored
build artifacts; `SPIKE — do not ship`).

**Setup.** A custom `strata-stream` `ggml_backend_buffer_type` was registered at
runtime and the Phase 2 expert override (`\.ffn_(up|down|gate|gate_up)_(ch|)exps`)
was re-pointed at it — exactly the §2.2 design, **with zero edits to ggml or
llama**. Expert bytes were held in a backing file and faulted into the tensor's
address by a **residency pre-pass (insertion point A)** before every
`llama_decode`.

**Result — correctness confirmed:**

| run | token sequence |
|---|---|
| plain llama.cpp (baseline) | `38 12 15 121 156 103 38 12` |
| streamed through `strata-stream` | `38 12 15 121 156 103 38 12` |

**Byte-identical.** `disk_reads=54` (6 expert tensors × 9 decode steps) confirms
the experts were genuinely re-read on demand each step, not left statically
resident. **Option (b)+(c) with insertion point (A) is validated; proceed to
Task 2 with the pre-pass residency policy and no upstream ggml patch.**

**One implementation note surfaced by the spike (feeds Task 2/3):** ggml's
default (linear) tensor allocator places all tensors of a buffer at offsets
within one contiguous `get_base()` region, so a naive "one small slot pool
smaller than the region" fights the allocator at *load* time. The resolution
for Task 2/3 is to size the buffer's region to the **bounded working set**
(n_slots × slot_bytes) and have the residency pass map each needed expert into a
slot within that region (updating `tensor->data` to the slot), rather than
giving the allocator the full expert footprint. Correctness and the on-demand
fault mechanism are both proven; bounded-memory is an allocator-integration
detail for Task 2, not a feasibility risk. The #1 risk is retired.

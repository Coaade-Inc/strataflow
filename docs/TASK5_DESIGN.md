<!--
Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
-->

# Phase 3 Task 5 Design: the `.strata` packed format + trunk/cache memory dial

**Status:** design + feasibility, implementation-ready. **Branch:** `phase-3/task-5-design` (not pushed).
**Scope baseline:** [`docs/PHASE3_PLAN.md` Task 5](./PHASE3_PLAN.md) and [`docs/PLAN.md` §3.10](./PLAN.md). Read those first; this document decides the exact format, the reader integration, the ABI/CLI dial, the test plan, and resolves the one pivotal question that governs what Task 5 is allowed to claim.

This is a design document. It does not implement the task. The build spec in §7 is what a coder implements next.

---

## 1. The pivotal question, resolved

> Does `.strata` merely reorganize expert bytes on disk (aligned, contiguous per-`(layer,expert)` blobs = better/aligned streaming I/O, but at load time we still reassemble the FULL stacked expert tensor into the ggml buffer, so RAM is bounded per-LAYER as in Task 3) — OR can it enable TRUE per-expert sub-working-set residency (only the top-k active experts' bytes resident per token)?

**Answer, stated honestly:**

- The *kernel* reads **only the routed top-k experts' bytes** of each stacked expert tensor per token. This is now proven, not assumed (§1.1–§1.3). So the headline win is reachable **as an I/O property**: Task 5 can bound **bytes-streamed-from-disk per token** to the top-k active experts, not the whole layer.
- The *address-space / RAM-residency* win is a **separate, harder property** blocked by a timing constraint, not by the kernel: for **exact** correctness, every expert the current token's router *could* select must already be resident at its correct `cur_a*nb02` offset **before** `llama_decode` runs, because the router runs **inside** the graph and the active set is unknown until the graph executes (§1.4). Within one decode there is no per-op seam to fault an expert in on demand.

Therefore **Task 5 delivers**:

1. The `.strata` on-disk format: aligned, contiguous per-`(layer,expert)` blobs with gate+up+down bundled, a trunk block, and an index — giving **sequential, 4 KiB-aligned, direct-I/O-friendly reads** and the data layout that makes per-expert streaming *possible*.
2. **Bounded bytes-streamed-per-token to the top-k active experts** via a per-op residency seam that is *safe to add* (§1.5) — this is the real SSD-bottleneck win and the honest "trillion-params in a tiny I/O budget" claim.
3. The **trunk/cache memory dial** (`--trunk-gb` / `--cache-gb`), ABI-compatible.

**Task 5 defers**: *exact* per-expert **RAM-resident-set** bounding **inside a single decode without any seam**. That is impossible given the in-graph router (§1.4). The two routes to it are (a) a Phase 4 **predictor** (previous-token / hidden-state routing) that makes per-expert residency *approximate* and is only exact when the prediction is right, or (b) the **per-op residency seam** this design specifies (§1.5), which makes per-expert *byte-streaming* exact but still needs the full-footprint virtual region to exist. True per-expert **RAM** bounding (not just I/O bounding) additionally needs a demand-paged virtual region (§1.6) and is marked a Phase 4 follow-up.

The distinction the rest of this section nails down: **bounding RESIDENT RAM** vs **bounding BYTES-STREAMED-PER-TOKEN**. Task 5 delivers the second exactly and the first per-layer (Task 3's model) with a documented path to per-expert.

### 1.1 The kernel reads only routed experts — plain path (source)

`ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_mul_mat_id` (the `GGML_OP_MUL_MAT_ID` op):

```c
for (int cur_a = 0; cur_a < n_as; ++cur_a) {          // n_as == n_expert
    const int64_t cne1 = matrix_row_counts[cur_a];    // rows routed to expert cur_a
    if (cne1 == 0) {
        continue;                                     // UNROUTED expert: skipped
    }
    // tiled takes over if profitable for this expert
    if (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, ...)) {
        continue;
    }
    const char * src0_cur = (const char *) src0->data + cur_a * nb02;  // only here
    ...
}
```

`matrix_row_counts` is built from the router's `ids` tensor earlier in the same op. An unrouted expert (`cne1 == 0`) is `continue`d **before** `src0->data + cur_a*nb02` is ever formed. The weight bytes of unrouted experts are never dereferenced. (`ggml-cpu.c` ~lines 1664–1730; the offset read is line 1683.)

### 1.2 The kernel reads only routed experts — tiled path (source)

The tiled path is **not** a separate full-tensor scan. It is called **inside** the same `for cur_a` loop, **after** the `cne1 == 0` skip, once per *routed* expert. In `ggml/src/ggml-cpu/tiled/tiled.cpp`:

- `ggml_compute_forward_mul_mat_id_tiled(...)` (entry ~line 1227) returns `false` on `use_ref`, on unsupported `src0/src1` types, or when `ggml_tiled_min_batch(cne1)` says the routed row count is below the tiling threshold — in which case the plain path (§1.1) runs for that expert. It is dispatched **per expert**, never over all `n_as`.
- `ggml_compute_forward_mul_mat_id_tiled_one_expert<...>` (~line 838) computes `const char * src0_cur = (const char *) src0->data + cur_a * nb02;` — the **same** per-expert offset as the plain path (line 859). It touches only `src0_cur`'s `nb02`-sized slice.

So the tiled path **also** skips unrouted experts and reads the same single routed-expert slice. There is no code path in `MUL_MAT_ID` that scans all `n_expert` weight regions.

### 1.3 Empirical confirmation (spike, this task)

Source reading is necessary but a page-fault probe is decisive. A throwaway spike (`spike/strata_residency/spike_touch.cpp`, `SPIKE — do not ship`, binary gitignored) owns the expert tensors with a custom ggml buffer type backed by a page-aligned `mmap` region, fills it with the real weights, then before each single-token decode `mprotect(PROT_NONE)`s **every** per-`(layer,expert)` slice and installs a `SIGSEGV` handler that records which slice first faults and re-enables it. Run on the tiny MoE fixture (2 layers × 8 experts, top-2, F32), single-threaded:

```
expert stacked tensors: 48 slices (per-expert) across all stacked tensors   # 2 layers x 3 kinds x 8 experts
token=38  distinct expert-slices touched=12 : blk.0.ffn_{gate,down,up}_exps[e0,e5] blk.1.ffn_{gate,down,up}_exps[e2,e3]
token=12  distinct expert-slices touched=12 : blk.0.ffn_{gate,down,up}_exps[e1,e7] blk.1.ffn_{gate,down,up}_exps[e4,e5]
token=15  distinct expert-slices touched=12 : blk.0.ffn_{gate,down,up}_exps[e2,e5] blk.1.ffn_{gate,down,up}_exps[e0,e5]
token=121 distinct expert-slices touched=12 : blk.0.ffn_{gate,down,up}_exps[e3,e7] blk.1.ffn_{gate,down,up}_exps[e4,e5]
```

Of 48 per-expert slices, **exactly 12** fault per token = 2 layers × 3 kinds × **top-2** experts. The remaining 36 slices stay `PROT_NONE` for the whole decode and the decode completes correctly, so **nothing else in a single `llama_decode` touches inactive-expert bytes** — not load (prompt decode ran before arming), not graph build, not the KV path, not any prefetch/validation. The decoded tokens (`38 12 15 121 …`) match the §8 baseline sequence in `PHASE3_PLAN.md`, so poisoning the inactive experts did not perturb output.

**Caveat (honest):** the fixture is F32, so at runtime it exercised the **plain** path (§1.1). The tiled path is confirmed for the K-quant types by source reading (§1.2); the design does not depend on it behaving differently, and the test plan (§6) adds a quantized-fixture variant so the tiled path is exercised in CI once Task 5 lands.

### 1.4 The CPU_REPACK load-time question — resolved (does not touch all bytes for us)

The load log `"… cannot be used with preferred buffer type CPU_REPACK, using CPU instead"` raised the worry that llama repacks/validates expert weights at load (which would touch all expert bytes and defeat lazy residency). It does **not**, for our experts:

- `ggml/src/ggml-cpu/repack.cpp`: `ggml_backend_cpu_repack_buffer_set_tensor` runs `tensor_traits->repack(tensor, data, size)` with `GGML_ASSERT(size == ggml_nbytes(tensor))` — i.e. CPU_REPACK **does** rewrite the *whole* tensor at load. If experts lived on CPU_REPACK, lazy residency would be defeated.
- But experts **do not** live on CPU_REPACK. In `src/llama-model-loader.cpp` (~1235–1286), when a `tensor_buft_overrides` pattern matches and `overrides->buft != ggml_backend_cpu_buffer_type()`, the loader takes `buft = overrides->buft` **directly** and skips `select_weight_buft` (which is what picks CPU_REPACK). Our `strata_stream_buft` is not the plain CPU buft, so expert tensors are placed on **our** buffer type, whose `set_tensor` does a plain `memcpy` (and, in production, records the file range) — **no repack, no all-bytes touch**. The `n_tensors_moved` log fires precisely *because* experts were moved off the layer-default CPU_REPACK onto our buft; it is the signal that the override won, not a warning that bytes were rewritten.

This is why the §8 feasibility spike already decoded byte-identically: the override path keeps expert bytes out of the repacker.

### 1.5 Why exact per-expert streaming needs a per-op seam (and why it is safe)

The kernel reads only top-k (§1.1–§1.3), but the router (`ffn_gate_inp`) that *chooses* top-k runs **inside** the graph during `llama_decode`. One `llama_decode` builds and runs the **full** graph for all layers in a single call; the `ggml_backend_sched` eval callback batches node ranges and **cannot relocate a weight leaf's `data` pointer mid-graph** (established in Task 3). So before `llama_decode` we do **not** know which experts token *T* will touch.

Two consequences:

- **For exact correctness with a pre-decode pre-pass only (Task 3's model):** every expert a decode *could* read must be resident. For `MUL_MAT_ID` that is the whole stacked tensor per `(layer, kind)` (so `cur_a*nb02` lands on valid bytes for whichever experts route). Resident RAM is bounded **per layer**, exactly, deterministically. This is what Task 2/3 ship and what Task 5 keeps as the always-correct fallback.
- **For exact per-expert byte-streaming (the Task 5 I/O win):** add a **per-op residency seam** — a residency callback invoked at `MUL_MAT_ID` dispatch, *after* `matrix_row_counts` is known but *before* `src0_cur` is read, that ensures only the routed experts' `nb02` slices are present. Because the kernel skips `cne1 == 0` experts, the seam only needs to stream the top-k slices from the `.strata` per-expert blobs. This bounds **bytes-streamed-per-token** to top-k exactly, with no prediction and no correctness risk.

The seam is the one place this design may carry a **minimal, clearly-marked upstream patch** (per `PHASE3_PLAN.md` §2.4/§5 risk 5): a single callback hook in the `MUL_MAT_ID` branch of `ggml_compute_forward`. It is behind a null-check (no-op when StrataFlow is not driving the buffer), lives at exactly one site, and is additive. **Task 5 specifies the seam but makes it optional** (see §7 scope): the format and dial land first and are correct with the pre-pass alone (per-layer bounding, I/O still improved by `.strata` locality); the seam upgrades I/O-per-token from per-layer to top-k and is gated behind a config flag so a seam-free build stays byte-identical.

### 1.6 RESIDENT-RAM vs BYTES-STREAMED-PER-TOKEN — which exit criteria each satisfies

| Property | What bounds it | Exact? | Exit criterion served | Task 5? |
|---|---|---|---|---|
| **Bytes-streamed-per-token** | `.strata` top-k streaming (per-op seam, §1.5) or full-layer streaming (pre-pass) | Exact (seam: top-k; pre-pass: per-layer) | #1 bounded SSD traffic, #2 ≥1.5× mmap, #4 beat kimi-k3 at equal RAM/disk | **Yes** (seam = top-k; without seam = per-layer, still aligned/contiguous) |
| **Resident RAM (cache tier)** | `SlotPool` size (`--cache-gb`) + trunk (`--trunk-gb`) | Exact, bounded by the dial | #1 bounded RSS, no OS swapping | **Yes** (the dial) |
| **Resident RAM at *per-expert* granularity within one decode** | demand-paged full-footprint virtual region (`PROT_NONE` reserve, populate top-k per decode) + per-op seam | Exact only with the seam; address space stays full-size | the aspirational "trillion params, tiny RAM" headline *as RAM* | **Deferred** (Phase 4): the virtual region must still span `n_expert*nb02`, so this bounds *physical* pages, not address space; needs the seam **and** careful munmap/madvise(DONTNEED) eviction |

The honest headline for Task 5: **bounded, aligned, top-k bytes-streamed-per-token + a bounded resident-RAM dial**, exact and deterministic. True per-expert *resident-RAM* shrinkage inside one decode is a Phase 4 build on top of the seam.

---

## 2. The `.strata` on-disk format (v1)

Design goals: GGUF-compatible metadata so existing tooling reads the header; a contiguous trunk block loaded once; one 4 KiB-aligned contiguous blob per `(layer, expert)` holding gate+up+down together (row-column bundling); an index for `{layer,expert} -> {offset,length}`; original GGUF still loadable through an index (slower). Little-endian throughout, matching GGUF.

### 2.1 File layout

```
+--------------------------------------------------------------+ offset 0
| STRATA SUPERBLOCK (fixed 128 bytes, see 2.2)                 |
+--------------------------------------------------------------+
| GGUF METADATA BLOB (verbatim GGUF header+KV+tensor-info,     |
|   no_alloc form; so gguf_* and read_shape_from_gguf parse it)|
+--------------------------------------------------------------+
| EXPERT INDEX  (packed array of ExpertIndexEntry, 2.4)        |
+--------------------------------------------------------------+ pad -> 4096
| TRUNK BLOCK   (all non-expert tensors, contiguous, 2.5)      |
|   4 KiB-aligned start; loaded up front into normal buffers   |
+--------------------------------------------------------------+ pad -> 4096
| EXPERT REGION                                                |
|   per (layer,expert) blob, each 4 KiB-aligned:               |
|     [ gate slice | up slice | down slice ]  (bundled)        |
|   co-activated experts placed adjacently (Phase 4 reorders;  |
|   v1 writes natural {layer-major, expert-minor} order)       |
+--------------------------------------------------------------+ EOF
```

Every blob start and the trunk/expert region starts are aligned to `kIoAlignment` (4096, from `src/tws/async_io.h`) so `O_DIRECT`/`F_NOCACHE`/`FILE_FLAG_NO_BUFFERING` reads need no bounce buffer.

### 2.2 Superblock (fixed 128 bytes at offset 0)

| field | type | notes |
|---|---|---|
| `magic` | `char[8]` | `"STRATA01"` |
| `version` | `u32` | format version = 1 |
| `flags` | `u32` | bit0 = experts bundled gate+up+down; bit1 = index sorted by `{layer,expert}` |
| `gguf_meta_offset` | `u64` | start of the embedded GGUF metadata blob |
| `gguf_meta_size` | `u64` | bytes |
| `index_offset` | `u64` | start of the expert index |
| `index_count` | `u64` | number of `ExpertIndexEntry` |
| `trunk_offset` | `u64` | 4 KiB-aligned start of trunk block |
| `trunk_size` | `u64` | bytes |
| `expert_region_offset` | `u64` | 4 KiB-aligned start of expert region |
| `expert_region_size` | `u64` | bytes |
| `n_layers` | `u32` | redundant with GGUF; convenience |
| `n_experts` | `u32` | redundant with GGUF; convenience |
| `align` | `u32` | the alignment used (4096); readers assert `== kIoAlignment` |
| `header_crc32` | `u32` | CRC32 of bytes `[0, header_crc32 field)` |
| `reserved` | `u8[…]` | pad to 128 |

Readers validate `magic`, `version`, `align == kIoAlignment`, and `header_crc32` before trusting any offset.

### 2.3 Embedded GGUF metadata

The packer copies the source GGUF's header + KV + tensor-info section verbatim (the `no_alloc` metadata region: everything before the tensor data) into the metadata blob, so `read_shape_from_gguf` (in `src/model/ggml_model.cpp`, which uses `gguf_init_from_file` with `no_alloc=true`) can run **unchanged** against a `.strata` file by pointing it at `gguf_meta_offset`. This preserves arch, `expert_count`, `expert_used_count`, tensor shapes/types, and names — the packer never reinterprets weights.

### 2.4 Expert index

```c
struct ExpertIndexEntry {   // 48 bytes, 8-byte aligned
    uint32_t layer;         // blk.N
    uint32_t expert;        // 0..n_experts-1
    uint64_t blob_offset;   // absolute file offset, 4 KiB-aligned
    uint64_t blob_length;   // total bytes of the (gate+up+down) bundle, incl. internal pad
    uint64_t gate_rel;      // byte offset of the gate slice within the blob
    uint64_t up_rel;        // byte offset of the up slice within the blob
    uint64_t down_rel;      // byte offset of the down slice within the blob
};
```

`gate_rel`/`up_rel`/`down_rel` are the per-kind sub-ranges inside the bundle so the reader can map a `{layer,expert,kind}` to an exact aligned read without re-deriving sizes. Slice lengths are `nb02` for the corresponding stacked tensor (the per-expert stride), taken from the GGUF tensor-info. The index is written sorted by `{layer,expert}` (flags bit1) for binary search; v1 readers may also build an in-memory hash.

### 2.5 Trunk block

All non-expert tensors (`token_embd`, `output`, per-layer `attn_*`, `ffn_norm`, `ffn_gate_inp` router, norms) are concatenated contiguously in GGUF tensor order, each at its natural ggml alignment, the whole block 4 KiB-aligned at `trunk_offset`. This is the always-resident set the planner keeps in RAM/VRAM. The router `ffn_gate_inp` is deliberately in the trunk (it must be resident every token to compute routing).

### 2.6 What v1 does *not* do

- No per-expert reordering by co-activation (writes natural order; Phase 4 adds a reordered variant using calibration tables, format already supports it via arbitrary `blob_offset`).
- No sidecars (calibration stats, prerouter/EAGLE heads, recovery LoRA) — reserved via `flags` + a future trailer, out of Task 5 scope.
- No requantization (bytes are copied verbatim; byte-identical decode is the test gate, §6).

---

## 3. Reader integration + GGUF-direct fallback

### 3.1 Reader seam

`.strata` reading plugs in under the existing streaming path with **no change to the kernel contract**:

- New `src/tws/strata_file.{h,cpp}` (`StrataReader`): opens a `.strata` via `BlockFile`, validates the superblock, parses the index into an `{layer,expert,kind} -> {offset,length}` map, and exposes `const GgufMeta& metadata()` (a pointer/length into the embedded GGUF blob) plus `read_blob(layer, expert, kind, dst)`.
- `src/model/ggml_model.cpp`: `read_shape_from_gguf` gains a `.strata`-aware entry (detect `STRATA01` magic; if present, parse the embedded GGUF blob at `gguf_meta_offset`, else treat as plain GGUF). The rest of `load_ggml_model` (plan → override array → `strata_stream_buft`) is unchanged.
- `src/tws/stream_buft.{h,cpp}` + `src/tws/weight_store.{h,cpp}`: the backing store gains a file-range source. Today `stream_buft` has a `store` (in-memory) plus a documented "optional BlockFile seam" and the residency pass's `load_slot` lambda. Task 5 wires `load_slot` to:
  1. If a `StrataReader` is configured for the live buffer: `read_blob` the exact `{layer,expert,kind}` aligned range from the `.strata` expert region.
  2. Else (GGUF-direct fallback): read the stacked-tensor byte range recorded at load from the source GGUF (what Task 2/3 already do via `set_tensor` into the in-memory `store`; the fallback keeps that path).

The `strata_stream_buft` **buffer contract does not change**: `is_host == true`, the committed region holds valid bytes at compute time. Only the *provenance* of the bytes changes (aligned `.strata` blob vs scattered GGUF range vs in-memory store). This preserves the byte-identical guarantee by construction (same bytes, same math).

### 3.2 How the dial drives streaming granularity

- **Pre-pass mode (default, seam-free, exact per-layer):** `stream_buft_ensure_decode_residency(prefill)` streams whole stacked tensors per `(layer, kind)` through the `SlotPool`, now sourced from `.strata` blobs. The `.strata` layout makes each such read sequential and aligned (gate+up+down of a layer's experts are adjacent), which is the I/O-locality win even without the seam.
- **Per-op seam mode (opt-in, exact top-k):** when the residency seam (§1.5) is enabled, `load_slot` streams only the routed `{layer,expert,kind}` blobs the `MUL_MAT_ID` dispatch asks for, bounding bytes-streamed-per-token to top-k. Falls back to pre-pass mode when the seam is unavailable (seam-free build), with identical output.

### 3.3 Co-residency of gate+up+down

The three kinds for one `(layer,expert)` are bundled in one blob (one aligned, contiguous read), matching `PLAN §3.10` row-column bundling. The reader issues one `read_at` per `(layer,expert)` and splits by `gate_rel`/`up_rel`/`down_rel` into the three stacked-tensor slots. This is the single biggest SSD win over scattered GGUF ranges (one seek+read instead of three).

---

## 4. ABI-compatible `sf_context_params` additions + CLI dial

### 4.1 ABI additions (append-only)

`include/strataflow/strataflow.h` — append at the **end** of `sf_context_params` (preserving ABI; `vram_budget`/`ram_budget` already exist and stay):

```c
typedef struct sf_context_params {
    /* ... existing fields unchanged ... */
    sf_backend preferred_backend;

    /* ---- Task 5 memory dial (appended; ABI-compatible) ---------------- */
    /* Explicit trunk/cache split in BYTES. 0 = derive from ram_budget/
       vram_budget + the plan (back-compatible: old callers that memset or use
       sf_context_default_params get 0 => existing behavior). */
    uint64_t trunk_bytes;        /* always-resident trunk cap (RAM tier)      */
    uint64_t cache_bytes;        /* streamed-expert cache cap (RAM tier)      */
    uint64_t vram_cache_bytes;   /* hot-expert cache cap (VRAM tier); 0=auto  */
} sf_context_params;
```

`sf_context_default_params()` fills the new fields with `0` (= "derive from budgets/plan"), so every existing caller and binary keeps its current behavior. `--trunk-gb`/`--cache-gb` are multiplied to bytes by the CLI before filling these fields.

### 4.2 Planner wiring

`src/plan/planner.{h,cpp}`: `PlacementPlan` already carries `trunk_layers_in_vram/ram`, `expert_slots_vram/ram`, `planned_peak_bytes`. Task 5 adds a path where an explicit `trunk_bytes`/`cache_bytes` **overrides** the auto-derived split: `cache_bytes / slot_bytes` → `expert_slots_ram` (the `SlotPool` size, wired through `set_stream_buft_slots`); `trunk_bytes` caps the trunk the plan keeps resident. When both are 0, the existing auto plan is unchanged. This keeps the dial additive and the auto path a strict superset.

### 4.3 CLI flags

The CLI tool (`tools/strata-cli`) gains `--trunk-gb <float>` and `--cache-gb <float>` (GiB → bytes). They set `trunk_bytes`/`cache_bytes`. `--strata <path>` / plain positional model path auto-detects `.strata` vs `.gguf` by magic. Flags are documented in `--help`; absent flags → `0` → auto plan.

---

## 5. Packer tool

New `tools/strata-pack/` (C++, STATIC, our C++20): `strata-pack <in.gguf> <out.strata>`.

1. Open the source GGUF via `gguf_*` (`no_alloc=true`) to get metadata + tensor-info (reuse the pattern in `read_shape_from_gguf`).
2. Write the superblock (placeholder offsets), copy the GGUF metadata blob verbatim.
3. Classify tensors: expert (`parse_expert_tensor_name` returns valid) vs trunk (everything else). `parse_expert_tensor_name` already recognizes `blk.N.ffn_{gate,down,up,gate_up}_exps`.
4. Write the trunk block (4 KiB-aligned), recording nothing in the index (loaded wholesale).
5. For each `(layer, expert)`: write a 4 KiB-aligned blob = gate slice ⊕ up slice ⊕ down slice (each slice = the per-expert `nb02`-stride sub-range sliced out of the stacked GGUF tensor), record an `ExpertIndexEntry`. For a fused `ffn_gate_up_exps` arch, bundle gate_up ⊕ down.
6. Write the index, backfill superblock offsets + `header_crc32`.
7. v1 writes experts in `{layer-major, expert-minor}` order (co-activation reordering is Phase 4; the format already allows arbitrary order).

The packer never requantizes or reshapes; it copies bytes. This is what makes the byte-identical test (§6) a true guard.

---

## 6. Test plan

All CI, CPU-only, on the tiny MoE fixture (`tools/testdata/make_tiny_moe_gguf.py`), gated to skip cleanly when the fixture/python is unavailable (existing `STRATAFLOW_TEST_GGUF` pattern). New `tests/test_strata_pack.cpp` + `tests/test_strata_file.cpp`, registered in `tests/CMakeLists.txt`.

1. **Pack round-trip / index integrity** (`test_strata_file.cpp`): pack the tiny MoE → `.strata`; reopen with `StrataReader`; assert superblock magic/version/`align==4096`/`header_crc32`; assert `index_count == n_layers * n_experts`; assert every `{layer,expert}` round-trips to a `{offset,length}` and `read_blob` returns the same bytes as the corresponding GGUF stacked-tensor slice.
2. **4 KiB alignment** : assert every `blob_offset % 4096 == 0` and `trunk_offset % 4096 == 0` and `expert_region_offset % 4096 == 0`.
3. **gate+up+down adjacency** : for a chosen `(layer,expert)`, assert `gate_rel < up_rel < down_rel` within one blob and that the three slices are contiguous (modulo internal alignment pad) in one `blob_length`, i.e. one `read_at` yields all three kinds.
4. **Byte-identical decode `.strata` vs GGUF** (the headline guard): load the tiny MoE from GGUF and from its `.strata` (both with `STRATAFLOW_FORCE_STREAM_EXPERTS=1`), greedy-decode the same prompt, assert **identical token sequences**. Proves the pack changed layout, not values.
5. **Determinism across the dial** : decode with `cache_bytes` large (holds all experts) vs forced small (eviction-heavy via `set_stream_buft_slots`); assert identical token sequences and that `StreamBuftStats` shows hits+misses+evictions in the small case (reuses the Task 3 proxy; extends it to the `.strata` source).
6. **ABI defaults** (`test_api.cpp`): `sf_context_default_params()` sets `trunk_bytes==0 && cache_bytes==0 && vram_cache_bytes==0`; a struct `memset` to zero still yields valid behavior; passing explicit `trunk_bytes`/`cache_bytes` changes `PlacementPlan.expert_slots_ram` (observable via `sf_describe_plan`). Guards the append-only ABI.
7. **Quantized fixture (tiled-path coverage)** : add a Q4_K (or similar K-quant) variant of the fixture so the `MUL_MAT_ID` **tiled** path is exercised; re-run test 4 (byte-identical GGUF vs `.strata`) on it. Closes the §1.3 F32-only caveat in CI.
8. **Seam parity (if the per-op seam is implemented)** : with the seam enabled vs disabled, assert identical greedy output and that seam-mode `StreamBuftStats` streams ≤ top-k × kinds × layers blobs per token (bytes-streamed-per-token bound). If the seam is deferred, this test is written `GTEST_SKIP()`-style with a documented reason.

Build/verify: `cmake --preset debug && cmake --build build/debug && ctest --preset debug`, plus clang `-Werror` parity. No throughput assertions in CI.

---

## 7. Scope line for the coder

**Implement in Task 5 (format + dial, exact + deterministic, no required ggml edit):**

1. `tools/strata-pack/` packer per §5 (verbatim byte copy; trunk block; 4 KiB-aligned per-`(layer,expert)` gate+up+down bundles; GGUF metadata embedded; index).
2. `src/tws/strata_file.{h,cpp}` `StrataReader` (superblock + index parse via `BlockFile`; `read_blob`).
3. Wire `src/tws/stream_buft.cpp` `load_slot` + `src/tws/weight_store` to source expert bytes from `StrataReader` when a `.strata` is loaded, else keep the GGUF-direct/in-memory fallback (Task 2/3) unchanged. **Do not change** the buffer contract (`is_host`, committed region valid at compute time) or the pre-pass residency model — that stays Task 3's exact per-layer model.
4. `src/model/ggml_model.cpp`: `.strata` magic detection in `read_shape_from_gguf`; parse embedded GGUF; everything downstream unchanged.
5. ABI: append `trunk_bytes`, `cache_bytes`, `vram_cache_bytes` to `sf_context_params`; default them to `0` in `sf_context_default_params()`; wire the override into the planner (`expert_slots_ram`/trunk cap) only when non-zero.
6. CLI: `--trunk-gb` / `--cache-gb`, `.strata` auto-detect.
7. Tests §6.1–§6.7 (§6.8 only if the seam is implemented). New files carry the Coaade SPDX header. STATIC libs. C++20 our-targets-only. No edit to `ggml/`/`llama/` source for this slice.

**Optional stretch within Task 5 (the exact top-k I/O win), gated and additive:**

8. The per-op residency seam (§1.5): a single null-checked callback at the `GGML_OP_MUL_MAT_ID` dispatch site that lets `load_slot` stream only routed `{layer,expert,kind}` blobs after `matrix_row_counts` is known. If implemented, it is behind a config flag, a seam-free build stays byte-identical, and test §6.8 proves the per-token byte bound. If skipped, record it as the first Phase 4 item.

**Explicitly deferred (do NOT attempt in Task 5):**

- True per-expert **resident-RAM** shrinkage inside one decode (needs §1.5 seam **and** a demand-paged full-footprint virtual region with `madvise(DONTNEED)` eviction — §1.6).
- Co-activation reordering of expert blobs (Phase 4 calibration).
- Sidecars (calibration/prerouter/EAGLE/LoRA), requantization presets (Phase 6).
- Native async direct-I/O backends (Task 4) — `.strata` reads go through the existing `BlockFile` whose backend Task 4 upgrades independently.

**What Task 5 delivers vs defers (one line):** Task 5 delivers the aligned, contiguous, GGUF-compatible `.strata` format with gate+up+down bundling, a byte-identical streaming reader with a GGUF-direct fallback, and an ABI-compatible `--trunk-gb`/`--cache-gb` memory dial — bounding **bytes-streamed-per-token** (per-layer exactly without the seam; top-k exactly with the optional seam) and **resident RAM** (via the dial), both exact and deterministic. It **defers** true per-expert *resident-RAM* bounding inside a single decode, which the in-graph router makes impossible without a per-op seam plus demand-paged address space (Phase 4).

<!--
Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
-->

# Engine Core Design: own the forward pass over ggml (Option D)

**Status:** design + one de-risking spike (DONE), implementation-ready. **Branch:** `engine-core/design` (not pushed).
**Scope baseline:** [`docs/TASK5_BLOCKER.md`](./TASK5_BLOCKER.md) (Option D), [`docs/PLAN.md` section 3.5](./PLAN.md) (per-token DAG scheduler), [`docs/PHASE3_PLAN.md` section 2](./PHASE3_PLAN.md) (the a/b/c decision and the section 1.1 compute-time constraint), and [`docs/TASK5_DESIGN.md` sections 1.4-1.6](./TASK5_DESIGN.md) (why exact per-expert residency needs us to own the compute loop). Read those first; this document decides *how* we take ownership of execution and *in what order*.

This is a design document plus the evidence from one throwaway spike (`spike/engine_core/`, FEAT-002, binary gitignored). It does not implement the engine. The task breakdown in section 8 is what a coder implements next. The spike's measured agreement with the llama oracle (section 7, section 10) is the gate that decides whether we proceed.

The headline decision: **build and run StrataFlow's own transformer forward pass directly on ggml**, replacing libllama's execution path (`llama_decode` / llama graph build / llama KV cache). We **keep ggml as the kernel and math library and keep ggml's GGUF reader** (we do not reimplement matmul, attention, softmax, rope, or quant kernels), and we **keep libllama only as a correctness oracle and for the tokenizer/vocab**. This unlocks per-token top-k expert streaming and the `.strata` single-file format, both of which libllama's monolithic decode made impossible.

---

## 1. Motivation and decision

### 1.1 Where we are

StrataFlow runs its forward pass **entirely through vendored libllama**. `src/model/ggml_model.cpp` is the only translation unit that includes `<llama.h>`; it owns a `llama_model` + `llama_context`, loads via `llama_model_load_from_file`, and `GgmlModel::forward` runs one `llama_decode` per token and takes the greedy argmax, carrying `n_past_`. Shape is read cheaply up front by `read_shape_from_gguf` (`gguf_init_from_file`, `no_alloc=true`). This is the Phase 1b/2 architecture described in [`docs/PHASE3_PLAN.md` section 1](./PHASE3_PLAN.md).

Two walls follow from running inference through `llama_decode`:

1. **The `.strata` single-file format is blocked.** [`docs/TASK5_BLOCKER.md`](./TASK5_BLOCKER.md) records the failure: `load_ggml_model()` reads the `.strata` embedded metadata correctly, but then still hands the `.strata` path to `llama_model_load_from_file`, and llama's GGUF loader rejects it:

   ```
   gguf_init_from_reader: invalid magic characters: 'STRA', expected 'GGUF'
   ```

   The whole load fails and falls back to the dry-run model. llama's loader needs a file it recognizes as GGUF with the trunk tensor DATA at `data_offset`; the `.strata` packer deliberately splits trunk and expert data into separate aligned regions, which llama's loader cannot consume.

2. **Exact per-expert residency inside one decode is impossible under libllama.** [`docs/PHASE3_PLAN.md` section 1.1](./PHASE3_PLAN.md) and [`docs/TASK5_DESIGN.md` sections 1.4-1.5](./TASK5_DESIGN.md) establish the constraint: the MoE kernel (`ggml_compute_forward_mul_mat_id` in `third_party/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`) reads weight bytes directly through `tensor->data` at compute time, and the router (`ffn_gate_inp`) that chooses the top-k experts runs *inside* the graph. One `llama_decode` builds and runs the full graph for all layers in a single call, so the active expert set is unknown until the graph has already run, and ggml exposes no per-op residency hook that would let us fault experts in on demand mid-decode.

### 1.2 The owner's rejection of the sidecar workaround

[`docs/TASK5_BLOCKER.md`](./TASK5_BLOCKER.md) proposed Option A: keep loading the original `.gguf` through libllama (trunk loads normally, experts route to `strata_stream_buft`), and use the `.strata` only as the aligned per-expert byte source. The owner rejected Option A: it leaves libllama driving inference, keeps the two-file story instead of the single-file story, and does not improve speed because the expert working set per decode is still governed by libllama's whole-graph execution. Option A trades the single-file goal for low risk and buys nothing on the critical path.

### 1.3 The decision: Option D

**Own the forward pass.** Build the transformer graph ourselves from ggml ops, allocate it ourselves, and execute it ourselves on the ggml backends. Specifically:

- **KEEP** ggml as the kernel and math library: `ggml_mul_mat`, `ggml_mul_mat_id`, `ggml_rms_norm`, `ggml_rope_ext`, `ggml_soft_max_ext`, `ggml_swiglu_split`, and the CPU/CUDA/Metal/Vulkan backends. We do NOT reimplement matmul, attention, softmax, rope, swiglu, or any quant kernel.
- **KEEP** ggml's GGUF reader (`gguf_init_from_file`, `gguf_get_tensor_*`) for pulling trunk tensors into our own ggml tensors.
- **STOP** using libllama's inference path: no `llama_decode`, no llama graph build, no llama KV cache for the forward pass.
- **KEEP** libllama linked and available only as (a) a correctness ORACLE in tests and (b) the tokenizer/vocab (`llama_tokenize`, `llama_token_to_piece`, `llama_vocab_*`), justified in section 4.

This is "Option (a)/own-the-forward-pass" from [`docs/PHASE3_PLAN.md` section 2.1](./PHASE3_PLAN.md), which that plan rejected for Phase 3 v1 on effort and rebase-risk grounds. The ground has shifted: Phase 3's buffer-type + pre-pass model (Option b/c) is correct and shipped, but it bounds expert residency only per-layer, and the `.strata` single-file format is blocked at the loader. The spike in section 7 retires the central feasibility risk (can our own graph reproduce llama's math), so Option D is now the chosen path for the engine core, built incrementally and gated at every step against the oracle.

---

## 2. Building and executing a forward pass directly on ggml

Everything below cites functions verified present in the vendored tree (`third_party/llama.cpp`, tag b11379). Line numbers are approximate anchors into the public headers; the signatures are exact. The reference graph being replicated is `third_party/llama.cpp/src/models/llama.cpp` (the llama-arch builder) and `build_moe_ffn` in `third_party/llama.cpp/src/llama-graph.cpp` (~line 2011). The spike (`spike/engine_core/spike_engine.cpp`) is the working, measured instance of this section.

### 2.1 Graph construction

- **Build context.** `ggml_init` (`ggml.h`) with a `no_alloc=true` context: tensor metadata (shapes, ops, the DAG) lives here, but no tensor data is backed yet; the allocator (section 2.3) assigns data later. `mem_size` is sized from `ggml_tensor_overhead() * N_NODES + ggml_graph_overhead()`.
- **Graph object.** `ggml_new_graph(ctx)` (`ggml.h` ~line 2889) allocates a `ggml_cgraph`; `ggml_build_forward_expand(gf, tensor)` (`ggml.h` ~line 2873) walks the DAG from the output tensor and records the topological node order.
- **Inputs and outputs.** `ggml_set_input` / `ggml_set_output` (`ggml.h` ~lines 893-894) mark the token-id and position tensors as inputs (so the allocator keeps them addressable for `ggml_backend_tensor_set` after allocation) and the logits as output. The spike creates `inp_tok` (I32, 1 element) and `inp_pos` (I32, 1 element).

### 2.2 The op set (grounded in ggml.h)

The forward pass for the llama-arch MoE uses exactly these ops; each is a real ggml function:

- **Embedding lookup:** `ggml_get_rows(ctx, token_embd, inp_tok)` (`ggml.h` ~line 1726) -> `[n_embd, n_tokens]`.
- **RMSNorm with weight:** `ggml_rms_norm(ctx, x, eps)` (`ggml.h` ~line 1399) followed by `ggml_mul(ctx, normed, norm_weight)`. eps = 1e-5 for the fixture.
- **Linear projections (Q, K, V, output proj, lm_head):** `ggml_mul_mat(ctx, weight, x)` (`ggml.h` ~line 1482).
- **Rotary embedding:** `ggml_rope_ext` (`ggml.h` ~line 1892). Full signature (verified): `(ctx, a, b /* positions */, c /* freq factors, may be NULL */, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)`. For the fixture: `n_dims = head_dim = 8`, `mode = GGML_ROPE_TYPE_NORMAL` (0; `ggml.h` ~line 250), `n_ctx_orig = 64`, `freq_base = 10000`, `freq_scale = 1`, and `ext_factor/attn_factor/beta_fast/beta_slow = 0/1/0/0`. The header comment at `ggml.h` ~lines 1888-1891 documents the NORMAL vs NEOX dim-interleave difference (`GGML_ROPE_TYPE_NEOX` is 2); choosing the wrong one silently produces wrong rotations, so the arch descriptor (section 6) records rope type per arch. The fixture is NORMAL.
- **Attention (explicit, no flash):** permute Q/K/V to `[head_dim, n_tokens, n_head]` with `ggml_permute`; `kq = ggml_mul_mat(ctx, k, q)`; fused scaled softmax `ggml_soft_max_ext(ctx, kq, mask, scale, max_bias)` (`ggml.h` ~line 1813; verified signature `(ctx, a, mask, scale, max_bias)`) with `scale = 1/sqrt(head_dim)` and `max_bias = 0`; `kqv = ggml_mul_mat(ctx, v_transposed, kq)`; permute back and `ggml_cont_2d` to `[n_embd, n_tokens]`. The causal mask is passed via `soft_max_ext`'s `mask` argument once we attend over KV history (section 3.4); for the single-position spike the mask is NULL.
- **MoE routing and weighting** (mirror of `build_moe_ffn`, `src/llama-graph.cpp` ~line 2011):
  - `logits = ggml_mul_mat(ctx, ffn_gate_inp, cur)` -> `[n_expert, n_tokens]`.
  - `probs = ggml_soft_max(ctx, logits)` (softmax gating; some arches use sigmoid, recorded per arch in section 6).
  - `selected = ggml_argsort_top_k(ctx, probs, n_expert_used)` (`ggml.h` ~line 2458; verified signature `(ctx, a, k)`) -> the selected expert ids `[n_expert_used, n_tokens]`.
  - weights via `ggml_get_rows` of the reshaped probs, normalized by `ggml_div` over `ggml_sum_rows` clamped with `ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY)` (`ggml_sum_rows` `ggml.h` ~line 1059; `ggml_clamp` ~line 1782). The clamp constant matches `build_moe_ffn` exactly.
  - expert FFN via `ggml_mul_mat_id` (`ggml.h` ~line 1500; verified signature `(ctx, as, b, ids)` where `as` is the stacked expert tensor, `b` the input, `ids` the selected experts): `up = ggml_mul_mat_id(up_exps, cur, selected)`, `gate = ggml_mul_mat_id(gate_exps, cur, selected)`, `act = ggml_swiglu_split(ctx, gate, up)` (`ggml.h` ~line 1360), `down = ggml_mul_mat_id(down_exps, act, selected)`.
  - weight each expert output by its normalized gate weight (`ggml_mul`) and sum over the `n_expert_used` dimension (`ggml_view_2d` slices + `ggml_add`).
- **Layout ops:** `ggml_cont`, `ggml_reshape_2d/3d`, `ggml_permute`, `ggml_view_2d`, `ggml_transpose` for head reshapes and expert-output reduction.

### 2.3 Allocation

- **Compute buffers:** `ggml_gallocr_new(buft)` (`ggml-alloc.h` ~line 48), then either `ggml_gallocr_reserve(galloc, graph)` (~line 57) to pre-size against a worst-case graph or `ggml_gallocr_alloc_graph(galloc, graph)` (~line 72) to allocate the current graph's intermediate tensors from the backend buffer type. `ggml_gallocr_get_buffer_size` (~line 74) reports the compute-buffer size for planning. Weight tensors that already own data (loaded by our GGUF reader into a data-owning context) are used directly as graph leaves; gallocr skips already-allocated tensors (the spike relies on exactly this).
- **Weights / trunk context:** `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)` (`ggml-alloc.h` ~line 82) allocates all tensors in a context from one buffer type in a single buffer; `ggml_backend_alloc_ctx_tensors_from_buft_size` (~line 80) sizes it first. This is how the engine's trunk tensors (section 5) get backing memory before we fill them from the file.

### 2.4 Execution

- **CPU backend:** `ggml_backend_cpu_init()` (`ggml-cpu.h` ~line 132); `ggml_backend_cpu_set_n_threads(backend, 1)` (`ggml-cpu.h` ~line 135) for determinism (the `CONTRIBUTING.md` greedy-determinism contract). Run with `ggml_backend_graph_compute(backend, gf)` (`ggml-backend.h` ~line 106).
- **Setting inputs / reading outputs:** `ggml_backend_tensor_set(tensor, data, offset, size)` (`ggml-backend.h` ~line 94) after allocation to write the token id and position; `ggml_backend_tensor_get` (~line 95) to read logits and (in the spike) per-layer intermediates.
- **GPU and multi-backend later:** the forward path to CUDA/Metal/Vulkan uses `ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel, op_offload)` (`ggml-backend.h` ~line 321) over multiple buffer types, with `ggml_backend_sched_graph_compute(sched, graph)` (~line 294) running the graph split across backends. The scheduler is also what lets a weights-on-SSD buffer type (section 3) and a VRAM compute buffer coexist in one graph. This is section 8's EC-7 and is explicitly non-trivial (section 10).

### 2.5 Reference and evidence

The construction above is the by-hand replica of `third_party/llama.cpp/src/models/llama.cpp` plus `build_moe_ffn`. The empirical proof that our hand-built graph reproduces llama's math is the FEAT-002 spike (section 7): on the tiny MoE fixture at a single BOS token, the engine graph's greedy argmax matched the oracle exactly and the full logit vector agreed to ~5e-5 max absolute delta. No divergence was found.

---

## 3. Per-token scheduler and per-expert residency

### 3.1 The ordering problem (restated with its source)

[`docs/PHASE3_PLAN.md` section 1.1](./PHASE3_PLAN.md) and [`docs/TASK5_DESIGN.md` sections 1.4-1.5](./TASK5_DESIGN.md): the router `ffn_gate_inp` runs mid-graph, and the MoE kernel reads `tensor->data` directly at compute time. Under one monolithic `llama_decode`, the active expert set is unknown until the whole graph (all layers) has already executed, and there is no per-op seam to fault an expert in on demand. Phase 3 therefore bounds expert *bytes-streamed* per token exactly, but can only bound *resident RAM* per layer (it must keep every expert a layer could route to resident, because it cannot see the router output before the layer's MoE op runs).

### 3.2 The resolution now that we own execution

When we own graph construction and execution, we are no longer forced to build and run one graph for all layers. We **execute the forward pass in per-layer segments**, splitting each layer's MoE into a router subgraph and an expert subgraph:

1. Build and compute the **router subgraph** for layer N: `ffn_norm` -> `ggml_mul_mat(ffn_gate_inp, cur)` -> `ggml_soft_max` -> `ggml_argsort_top_k(n_expert_used)`. Read back the `selected_experts` ids on the host with `ggml_backend_tensor_get`.
2. With the exact top-k `{layer, kind, expert}` set now known on the host, **make exactly those slices resident** via the existing `SlotPool` / `strata_stream_buft` / `StrataReader` (sections 4-5): one aligned read per routed `(layer, expert)` bundle (gate+up+down) from the `.strata` expert region, or the GGUF-direct fallback range.
3. Build and compute layer N's **expert subgraph**: `ggml_mul_mat_id` over the now-resident stacked tensors, swiglu, down projection, weight-and-sum, and the residual add that produces the layer output fed to layer N+1.

This bounds **resident expert RAM to exactly the top-k per layer** (not the whole layer), the property Phase 3 had to defer, because we read the router output *before* we build the expert matmul.

### 3.3 Why a top-k-resident stacked tensor computes correctly

`ggml_compute_forward_mul_mat_id` (ggml-cpu.c ~line 1554+) builds `matrix_row_counts` from the `ids` tensor and, for each expert `cur_a`, `continue`s before dereferencing `src0->data + cur_a*nb02` when `cne1 == 0` (that expert routed zero rows). So a stacked expert tensor in which only the top-k experts' `nb02` slices hold valid, resident bytes computes correctly: the kernel never touches the non-routed slices. This is the same kernel property Phase 3 proved by page-fault probe ([`docs/TASK5_DESIGN.md` sections 1.1-1.3](./TASK5_DESIGN.md): exactly 12 of 48 slices faulted per token = 2 layers x 3 kinds x top-2). Owning execution lets us act on it: we read the router first, then make resident exactly those slices.

### 3.4 KV cache ownership

Dropping libllama's inference path means dropping its KV cache. **Our engine owns the KV cache**, reusing `src/kv/kv_store.*`. Per layer, per token: append the layer's K and V (post-rope for K) to the KV store at the current position, then attend over positions `0..t`. The attention subgraph gathers K and V for `0..t` (as ggml tensors viewing the KV store) and applies the causal mask through `ggml_soft_max_ext`'s `mask` argument (section 2.2). The single-position spike has no history (`n_kv == n_tokens == 1`, NULL mask); EC-2 (section 8) adds the KV store and proves multi-token greedy decode against the oracle sequence.

### 3.5 Mapping onto the scheduler module

This is the per-token DAG from [`docs/PLAN.md` section 3.5](./PLAN.md), landing in `src/sched/scheduler.*` (today a thin `step()` wrapper). The DAG has per-layer compute nodes plus SSD->RAM and RAM->VRAM transfer nodes, double/triple-buffered so layer N+1's predicted experts load while layer N computes:

- The **predictor** (`src/predict/predictor.*`, `ExpertPredictor`) proposes layer N+1's likely experts from the previous token / hidden state so the transfer nodes prefetch ahead of the compute nodes. Prediction only affects *latency* (prefetch hit rate), never correctness: step 1's exact router read-back is authoritative, and a mispredicted prefetch is corrected by a synchronous load before step 3.
- The **planner** (`src/plan/planner.*`, `src/plan/llama_placement.*`) sizes the trunk and the expert slot pool and selects which `(layer, expert)` slices stream vs stay resident, feeding the SlotPool capacity.
- The segmentation adds a graph-build + allocate cost per layer that must be measured against the I/O win (section 10); the mitigation is to cache the per-layer subgraph shapes and reuse the gallocr reservation across tokens (the graph topology per layer is fixed; only input data changes).

---

## 4. Reuse vs replace

### 4.1 Reuse

- **ggml ops and backends** (section 2): all math stays ggml's. We do not reimplement a single kernel.
- **ggml's GGUF reader:** `gguf_init_from_file` / `gguf_get_tensor_*` to pull trunk tensors into our own ggml tensors. `read_shape_from_gguf` (the `no_alloc=true` metadata pre-pass in `src/model/ggml_model.cpp`) stays and is extended for `.strata` magic (section 5).
- **Existing StrataFlow modules, unchanged in contract:** `src/hw/profiler.*` (hardware profile), `src/plan/planner.*` + `src/plan/llama_placement.*` (`PlacementPlan`, `expert_ffn_regex`), `src/tws/weight_store.*` (`SlotPool`: `ExpertId{layer,expert}`, `acquire(id, load_fn)`, LRU, `CacheStats`), `src/tws/async_io.*` (`BlockFile`: `read_at`/`read_async`, 4 KiB align), `src/tws/stream_buft.*` (the custom `ggml_backend_buffer_type`, `is_host=true`, residency pass keyed by `{layer,kind}`, LRU, `StreamBuftStats`), `src/kv/kv_store.*`, `src/predict/predictor.*`.
- **StrataReader** from the WIP branch `phase-3/task-5-strata-format`: the `.strata` superblock/index parsing and `read_blob` data structures are reusable as-is. The broken part was only the llama-load wiring ([`docs/TASK5_BLOCKER.md`](./TASK5_BLOCKER.md)); Option D removes that wiring entirely (we never hand `.strata` to libllama).

### 4.2 Replace

- **The libllama forward pass:** `llama_decode`, llama's graph build, and llama's KV cache. These are replaced by sections 2-3. We still *link* libllama, but we no longer call it for inference.

### 4.3 The sf::Model seam is preserved

`sf::Model` (`src/model/model.h`: `shape()`, `tokenize()`, `detokenize()`, `forward()`, `eos_token()`) is the seam the C ABI (`include/strataflow/strataflow.h`), the CLI, the server, and every test above the seam sit on. **It does not change.** The engine lives *behind* that interface, inside `GgmlModel` (or a new sibling implementation `GgmlModel` delegates to). `GgmlModel::forward(last_token)` keeps its signature and determinism contract; internally it drives the per-layer segmented engine (section 3) instead of `llama_decode`. Nothing above the seam is touched. This is why Option D is safe to land incrementally: EC-5 (section 8) is the only task that flips the default, and it flips it behind an unchanged interface.

### 4.4 Tokenizer / vocab decision: keep libllama's vocab

**Keep libllama's vocab API** (`llama_tokenize`, `llama_token_to_piece`, `llama_vocab_bos`, `llama_vocab_n_tokens`, and the rest of `llama_vocab_*`) for now. Justification:

- Tokenization is **not** the forward pass. It is off the critical path for the SSD/expert-streaming win this whole effort targets.
- The vocab path is well tested and already wired in `GgmlModel::tokenize`/`detokenize`.
- Reimplementing SPM/BPE (and byte-fallback, added-token merges, normalization) is a large, error-prone surface with its own correctness burden and no payoff for the engine goal.

So the engine keeps a `llama_model` loaded *only* for its vocab (and as the oracle in tests); inference never calls `llama_decode`. A later option (recorded, not scheduled) is a native tokenizer to drop the libllama link entirely; it is explicitly out of scope here and off the critical path.

---

## 5. The `.strata` single-file resolution

Owning the loader removes the [`TASK5_BLOCKER`](./TASK5_BLOCKER.md) root cause: because *we* read trunk tensors into *our* ggml tensors and stream experts ourselves, no `llama_model_load_from_file` is ever handed the `.strata` path, so the `invalid magic 'STRA'` failure cannot occur. The single-file format "just works" because the only consumer that rejected it is gone from the load path.

### 5.1 Load path

1. **Open** the `.strata` via `BlockFile` (`src/tws/async_io.*`, 4 KiB aligned reads).
2. **Validate the superblock** ([`docs/TASK5_DESIGN.md` section 2.2](./TASK5_DESIGN.md)): magic `STRATA01`, `version`, `align == kIoAlignment` (4096), and `header_crc32` before trusting any offset.
3. **Parse the embedded GGUF metadata blob** at `gguf_meta_offset` with the `read_shape_from_gguf` style (`gguf_init_from_file`, `no_alloc=true`) for shapes, types, arch, `expert_count`, `expert_used_count`, and tensor names. This is metadata only; no tensor data is read here.
4. **Read the trunk block once** into our resident ggml tensors: create a context holding the trunk tensors (`token_embd`, `output_norm`, `output`, and per layer `attn_*`, `ffn_norm`, `ffn_gate_inp`), allocate it with `ggml_backend_alloc_ctx_tensors_from_buft` (section 2.3), then fill each tensor from the trunk region via `BlockFile::read_at` + `ggml_backend_tensor_set`. The router `ffn_gate_inp` is in the trunk because it must be resident every token.
5. **Register each per-`(layer, expert)` expert blob** with the `SlotPool` / `StrataReader` so the per-layer segmented execution (section 3) faults in only the routed top-k. One aligned `read_blob` per routed `(layer, expert)` yields gate+up+down together ([`docs/TASK5_DESIGN.md` section 3.3](./TASK5_DESIGN.md)).

### 5.2 Plain-GGUF path retained

Detect the file magic: `GGUF` -> plain GGUF path (trunk read directly from the GGUF via `gguf_get_tensor_*`; experts streamed from GGUF tensor byte ranges as the fallback source). `STRATA01` -> the `.strata` path above. Both feed the same engine; only the byte provenance differs, which preserves the byte-identical guarantee by construction (same bytes, same ops).

---

## 6. Scope staging: first ONE architecture

### 6.1 Target: the llama-arch MoE fixture

First target the Mixtral/llama-arch MoE from `tools/testdata/make_tiny_moe_gguf.py`: 2 layers, 8 experts, top-2, RoPE, RMSNorm, standard multi-head attention. Hparams: `n_embd=32`, `n_head=n_head_kv=4`, `n_layer=2`, `n_ff=64`, `n_expert=8`, `n_expert_used=2`, `head_dim=8`, rope dim 8, rms eps 1e-5, context length 64, all F32.

### 6.2 Exact tensors (names and ggml ne-order shapes)

The generator writes numpy shapes in row-major `(outer, ..., inner)`; GGUF/ggml store them reversed as `ne` (inner-first). The engine reads them in ne-order:

- `token_embd.weight` ne `[n_embd, n_vocab]` = `[32, 264]`
- `output_norm.weight` ne `[n_embd]` = `[32]`
- `output.weight` (lm_head) ne `[n_embd, n_vocab]` = `[32, 264]`
- per layer `blk.N.`:
  - `attn_norm.weight` ne `[n_embd]`
  - `attn_q.weight` ne `[n_embd, n_embd]`
  - `attn_k.weight` ne `[n_embd, n_embd]`
  - `attn_v.weight` ne `[n_embd, n_embd]`
  - `attn_output.weight` ne `[n_embd, n_embd]`
  - `ffn_norm.weight` ne `[n_embd]`
  - `ffn_gate_inp.weight` (router) ne `[n_embd, n_expert]` = `[32, 8]`
  - `ffn_gate_exps.weight` ne `[n_embd, n_ff, n_expert]` = `[32, 64, 8]`
  - `ffn_down_exps.weight` ne `[n_ff, n_embd, n_expert]` = `[64, 32, 8]`
  - `ffn_up_exps.weight` ne `[n_embd, n_ff, n_expert]` = `[32, 64, 8]`

The vocab is 264 (3 control + 256 byte-fallback + 5 normal), which is why the spike reports a 264-wide logit vector.

### 6.3 Exact op sequence (the one the spike implements)

Per `spike/engine_core/spike_engine.cpp`, verified against the oracle:

1. `inpL = ggml_get_rows(token_embd, inp_tok)` -> `[n_embd, 1]`.
2. For each layer `il` in `0..n_layer-1`:
   a. `cur = ggml_mul(ggml_rms_norm(inpL, 1e-5), attn_norm_w)`.
   b. `Q/K/V = ggml_mul_mat(wq/wk/wv, cur)`; reshape to `[head_dim, n_head(_kv), 1]`.
   c. `ggml_rope_ext(Q, inp_pos, NULL, head_dim=8, GGML_ROPE_TYPE_NORMAL, n_ctx_orig=64, freq_base=10000, freq_scale=1, 0, 1, 0, 0)`; same for K.
   d. attention, mirroring the non-flash (explicit) branch of `build_attn_mha` in `third_party/llama.cpp/src/llama-graph.cpp` (the `else` branch, ~lines 2692-2748): permute to `[head_dim, n_tokens, n_head]`; `kq = ggml_mul_mat(k, q)`; `ggml_soft_max_ext(kq, NULL, 1/sqrt(8), 0)`; `kqv = ggml_mul_mat(cont(transpose(v)), kq)`; permute back; `ggml_cont_2d` to `[n_embd, 1]`. The reference forces `GGML_PREC_F32` accumulation on `kq` in that branch, which the spike's F32 compute matches.
   e. `cur = ggml_mul_mat(wo, cur)`; `ffn_inp = ggml_add(cur, inpL)` (residual).
   f. `cur = ggml_mul(ggml_rms_norm(ffn_inp, 1e-5), ffn_norm_w)`.
   g. MoE: `router = ggml_mul_mat(gate_inp, cur)`; `probs = ggml_soft_max(router)`; `selected = ggml_argsort_top_k(probs, 2)`; weights via `ggml_get_rows` of reshaped probs, normalized by `ggml_div` over `ggml_clamp(ggml_sum_rows(weights), 6.103515625e-5, INFINITY)`; `up = ggml_mul_mat_id(up_exps, cur3, selected)`; `gate = ggml_mul_mat_id(gate_exps, cur3, selected)`; `act = ggml_swiglu_split(gate, up)`; `experts = ggml_mul_mat_id(down_exps, act, selected)`; `ggml_mul` by weights; sum over the `n_expert_used` dim via `ggml_view_2d` + `ggml_add`.
   h. `inpL = ggml_add(moe_out, ffn_inp)` (residual).
3. `cur = ggml_mul(ggml_rms_norm(inpL, 1e-5), output_norm_w)`; `logits = ggml_mul_mat(output, cur)` -> `[n_vocab, 1]`.

### 6.4 Generalization

Generalize via an **arch-descriptor table** mapping `arch -> graph-builder`, each builder owning its graph:

1. **llama-arch MoE** (above) first.
2. **dense llama** (no MoE: the FFN is a plain `ggml_mul_mat` gate/up + swiglu + down, no router).
3. **other MoE arches**: `qwen2moe` (shared expert + per-expert gate), DeepSeek-style (expert groups, shared experts, sigmoid gating with group-limited top-k) as seen in the variants of `build_moe_ffn`.

Each new arch is a **new graph builder we own and must validate against the oracle** (section 7). The descriptor records per-arch choices that silently change numerics: rope type (NORMAL vs NEOX), gating (softmax vs sigmoid), presence of shared experts, QK norm, bias terms. Adding an arch is never "free"; it is a new validated builder plus its oracle test.

---

## 7. Correctness strategy (libllama as oracle)

### 7.1 The gate for every step

For every engine step, decode the same prompt greedily through `llama_decode` (oracle) and through our engine on the **same** GGUF, and assert:

- **identical greedy token ids** (the hard gate), and
- **logits within a tight tolerance** (the soft gate, for localizing drift).

### 7.2 Measured baseline (FEAT-002 spike)

The spike (`spike/engine_core/`, binary gitignored) ran the full forward (both layers, attention + MoE FFN + final norm + lm_head) as OUR OWN `ggml_cgraph` on the ggml CPU backend (NOT `llama_decode`) and compared to `llama_decode` as oracle on `/projects/sandbox/moe.gguf` at a single BOS token, position 0, 1 CPU thread. Measured:

- engine argmax = 71, oracle argmax = 71 -> **argmax MATCH**
- **max absolute logit delta = 5.3167e-5** over the 264-wide vocab
- **mean absolute logit delta = 1.2078e-5**
- deterministic: identical numbers across repeated runs (1 CPU thread)

The ~5e-5 residual is non-bit-exact and expected. It is NOT caused by flash attention: the spike includes a probe that re-runs the oracle with flash attention forced off (`flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED`) and compares to the default (`AUTO`) run; on this CPU/F32 fixture the two oracle logit vectors are byte-identical (max delta 0), so flash attention is not selected in a way that affects the result and cannot be the residual source here. The residual instead comes from op ordering / fusion / accumulation differences between our hand-built graph and llama's graph builder (both run explicit F32 attention; the reference path forces `GGML_PREC_F32` accumulation on `kq`). The agreement to 5e-5 confirms the math is equivalent, including the riskiest part, the MoE top-2 routing through `ggml_mul_mat_id` over the stacked expert tensors. No divergence was found. This is the evidence base for proceeding.

### 7.3 CI test

Add `tests/test_engine_oracle.cpp` following the existing harness (`tests/test_util.h`, registered in `tests/CMakeLists.txt`, run via `ctest`), gated on the fixture with the `STRATAFLOW_TEST_*` skip-when-absent convention so it skips cleanly when the GGUF fixture or python is unavailable. The test decodes a fixed prompt through the oracle and through the engine and asserts identical greedy token ids and logits within tolerance. Extend later to quantized fixtures: tolerance widens for quant types (the F32 fixture's 5e-5 does not apply to Q4_K), but **argmax must still match** at every step.

### 7.4 Determinism

Single CPU thread (`ggml_backend_cpu_set_n_threads(backend, 1)`) or a fixed reduction order per the `CONTRIBUTING.md` determinism contract. Greedy output must be reproducible across runs and (per the Phase 3 invariant) across memory budgets once streaming lands.

---

## 8. Sequenced task breakdown

Each task is independently testable against the oracle, with files touched and an oracle-based done-criterion. This mirrors [`docs/PHASE3_PLAN.md` section 3](./PHASE3_PLAN.md).

### EC-0 - Spike (DONE, FEAT-002)
- **Builds:** the throwaway proof that a hand-built ggml graph reproduces llama's llama-arch MoE forward.
- **Touches:** `spike/engine_core/spike_engine.cpp`, `spike/engine_core/README.md` (binary gitignored).
- **Done-criterion (met):** engine argmax == oracle argmax on the tiny MoE, logits to 5.3167e-5 max abs delta. `SPIKE PASS`.

### EC-1 - Standalone ggml graph builder for llama-arch MoE, single-token prefill, behind sf::Model
- **Builds:** the engine's single-position forward (section 6.3) as production code behind `sf::Model`, all weights resident, no streaming yet.
- **Touches:** new `src/model/engine/` (graph builder + runner); `src/model/ggml_model.cpp` (delegate `forward` to the engine behind a flag); reuse the GGUF reader.
- **Tested:** new `tests/test_engine_oracle.cpp` (section 7.3), single token.
- **Done-criterion:** engine single-token greedy argmax == oracle on the tiny MoE in CI; logits within tolerance; gcc + clang `-Werror`.

### EC-2 - Our KV cache, multi-token greedy decode
- **Builds:** engine-owned KV cache (section 3.4) so decode over a multi-token prompt + generation matches the oracle sequence.
- **Touches:** `src/kv/kv_store.*` (wire append + attend-over-0..t); `src/model/engine/` (causal mask via `ggml_soft_max_ext`); `src/model/ggml_model.cpp`.
- **Tested:** extend `tests/test_engine_oracle.cpp` to a multi-token prompt and N generated tokens.
- **Done-criterion:** engine greedy token *sequence* == oracle sequence over the fixture, deterministic.

### EC-3 - Per-layer segmented execution with router read-back + SlotPool top-k residency
- **Builds:** section 3's router-subgraph read-back + make-resident + expert-subgraph, with the SlotPool bounded below the expert count.
- **Touches:** `src/sched/scheduler.*` (per-layer segmentation); `src/tws/stream_buft.*`, `src/tws/weight_store.*` (top-k residency); `src/model/engine/`.
- **Tested:** extend the oracle test with a bounded pool; assert output byte-identical to EC-1/EC-2 and `StreamBuftStats` shows hits/misses/evictions.
- **Done-criterion:** segmented top-k-resident decode is byte-identical to the fully-resident engine at a bounded pool; resident expert RAM bounded by top-k per layer.

### EC-4 - .strata single-file loader
- **Builds:** section 5's own trunk read + expert streaming from `.strata`; closes [`TASK5_BLOCKER`](./TASK5_BLOCKER.md).
- **Touches:** reuse `StrataReader` from `phase-3/task-5-strata-format`; `src/model/ggml_model.cpp` (`.strata` magic detection + trunk read); `src/tws/*` (expert blob source).
- **Tested:** byte-identical decode `.strata` vs source GGUF (the headline guard, [`docs/TASK5_DESIGN.md` section 6.4](./TASK5_DESIGN.md)).
- **Done-criterion:** engine decodes a `.strata` byte-identically to its source GGUF; no `llama_model_load_from_file` on `.strata`.

### EC-5 - Remove the llama_decode inference path from GgmlModel; engine is default
- **Builds:** flip the default so `GgmlModel::forward` always uses the engine; keep libllama linked for vocab + oracle only.
- **Touches:** `src/model/ggml_model.cpp` (drop the `llama_decode` branch from inference; keep vocab); keep `sf::Model` unchanged.
- **Tested:** the full existing suite stays green (interface unchanged); the oracle test is now the inference gate.
- **Done-criterion:** no inference call to `llama_decode`; all ~11 existing suites + the oracle test green on gcc + clang `-Werror`; the C ABI/CLI/server unchanged.

### EC-6 - Per-token DAG scheduler + predictor-driven prefetch overlap
- **Builds:** [`docs/PLAN.md` section 3.5](./PLAN.md) double/triple-buffered transfer nodes so layer N+1's predicted experts load while layer N computes.
- **Touches:** `src/sched/scheduler.*` (the DAG); `src/predict/predictor.*` (prefetch proposals); `src/tws/async_io.*` (`read_async`).
- **Tested:** oracle byte-identity preserved; prefetch improves hit rate (measured off-CI); mispredicted prefetch corrected by synchronous load (correctness unaffected).
- **Done-criterion:** output unchanged vs EC-3/EC-4; prefetch overlap demonstrably reduces per-token stall off-CI; CI proves correctness invariance.

### EC-7 - Generalize to a second arch and to GPU backends
- **Builds:** a second arch graph builder (dense llama, then a second MoE arch) and multi-backend execution via `ggml_backend_sched`.
- **Touches:** `src/model/engine/` (arch descriptor table, section 6.4); backend selection (`ggml_backend_sched_new`).
- **Tested:** oracle test per new arch/fixture; GPU path logic-tested on CI (no GPU), hardware-validated off-CI.
- **Done-criterion:** each supported arch matches the oracle argmax; the engine runs a graph split across CPU + a GPU backend off-CI.

---

## 9. CI / hardware coverage

CI is CPU-only. EC-1 through EC-5 are fully validated in CI on the tiny MoE fixture against the libllama oracle (greedy token identity + logit tolerance), gated with the `STRATAFLOW_TEST_*` skip-when-absent pattern. EC-6's prefetch-overlap *speed* and EC-7's GPU path are hardware-validated off-CI; their *correctness invariance* (output unchanged) is proven in CI. No throughput assertions run in CI.

---

## 10. Honest risk, effort, and commitment

- **Permanent per-arch maintenance.** Owning the forward pass means every architecture we support needs its own validated graph builder and an oracle test, and each must be kept rebase-safe as the vendored ggml advances (tag b11379 today; op signatures and defaults can shift). This is a standing commitment, not a one-time cost. The arch-descriptor table (section 6.4) contains the blast radius but does not remove it.
- **Per-layer segmentation overhead.** Splitting execution into per-layer router + expert subgraphs adds graph-build and allocation cost per layer per token. This must be measured against the I/O win; if the overhead dominates for models whose experts already fit in RAM, the engine should fall back to a single full-graph build (no segmentation) for the resident case. Mitigation: cache per-layer subgraph topology and reuse the gallocr reservation across tokens (topology is fixed; only input data changes).
- **Numeric parity across quant types is a continuing test burden.** The spike's 5e-5 agreement is on an F32 fixture. Quantized experts (Q4_K and friends) exercise the tiled `mul_mat_id` path and will have wider, type-dependent logit deltas; the gate stays argmax-match, but the tolerance table and quantized fixtures are ongoing work (parallels [`docs/TASK5_DESIGN.md` section 6.7](./TASK5_DESIGN.md)).
- **GPU multi-backend scheduling is non-trivial.** `ggml_backend_sched` over a weights-on-SSD buffer type plus CPU and GPU compute buffers (EC-7) is the hardest remaining piece and is deferred behind the CPU engine landing and proving out.
- **What the spike de-risks vs what remains open.** De-risked: the central feasibility question. Our own hand-built ggml graph reproduces llama's llama-arch MoE forward, greedy token identical and logits to 5.3167e-5 max abs delta (5.3167e-5 max, 1.2078e-5 mean over 264 vocab), deterministic, with the MoE top-2 routing through `ggml_mul_mat_id` confirmed equivalent. Still open and to be retired by EC-1..EC-7: multi-token KV correctness, segmented top-k residency byte-identity at a bounded pool, `.strata` byte-identity, removing `llama_decode` from inference without regressing the suite, prefetch overlap speed, and generalization to more arches and to GPU.
- **What this commits us to.** Option D makes StrataFlow the owner of its inference arithmetic (not just its I/O). The payoff is exact per-token top-k expert residency and the `.strata` single-file format, both impossible under `llama_decode`. The price is the per-arch builder + oracle-test maintenance above. The spike's PASS is the evidence that the payoff is reachable; the sequenced EC tasks each carry an oracle gate so a regression is caught at the step that introduced it.

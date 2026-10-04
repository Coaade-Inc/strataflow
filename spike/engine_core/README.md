<!-- Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0 -->
# engine_core spike (FEAT-002)

SPIKE -- do not ship. Throwaway de-risking spike for the engine-core design.

## Question under test

Can we build and run OUR OWN ggml compute graph for the tiny MoE forward pass
(NOT `llama_decode`) and reproduce llama.cpp's math on the same model, prompt and
position? This is the single riskiest unknown behind "Option D: own the forward
pass" (see docs/TASK5_DESIGN.md, docs/PHASE3_PLAN.md).

## What the spike does

`spike_engine.cpp` runs three stages on the SAME gguf, prompt and position 0:

1. ORACLE: loads via `llama_model_load_from_file` + `llama_init_from_model`
   (CPU, 1 thread), decodes a single BOS token at position 0 with one
   `llama_decode`, and captures the full logits (`llama_get_logits_ith(ctx,-1)`)
   and greedy argmax. Logits are also written to `/projects/sandbox/oracle_logits.bin`.
   The oracle is run twice: once with the default `flash_attn_type = AUTO` and
   once with it forced to `DISABLED`, so the spike can check whether flash
   attention changes the oracle result at all (a probe for the residual source).
2. ENGINE: opens the gguf a second time with `gguf_init_from_file(no_alloc=false)`
   so ggml allocates and fills every tensor, fetches each weight by its GGUF name
   (`token_embd.weight`, `output_norm.weight`, `output.weight`, and per layer
   `blk.N.{attn_norm,attn_q,attn_k,attn_v,attn_output,ffn_norm,ffn_gate_inp,
   ffn_gate_exps,ffn_down_exps,ffn_up_exps}.weight`), then builds a `ggml_cgraph`
   by hand mirroring `third_party/llama.cpp/src/models/llama.cpp` and
   `build_moe_ffn` in `src/llama-graph.cpp`:
   - input embedding via `ggml_get_rows(token_embd, tok)`
   - per layer: RMSNorm (`ggml_rms_norm` eps=1e-5 then `ggml_mul`) attn_norm;
     Q/K/V `ggml_mul_mat`; reshape to heads; `ggml_rope_ext` NORM (mode 0,
     n_dims=head_dim=8, freq_base=10000, freq_scale=1) on Q and K; attention via
     permute + `ggml_mul_mat(k,q)` + `ggml_soft_max_ext(scale=1/sqrt(8))` +
     `ggml_mul_mat(v,kq)`; output proj + residual add;
   - MoE: `ggml_mul_mat(gate_inp,cur)` -> `ggml_soft_max` ->
     `ggml_argsort_top_k(2)` -> weights via `ggml_get_rows` of reshaped probs,
     normalized by the clamped (`6.103515625e-5`) `ggml_sum_rows` via `ggml_div`;
     `ggml_mul_mat_id(up_exps/gate_exps,cur,ids)`, `ggml_swiglu_split`,
     `ggml_mul_mat_id(down_exps,...)`, weight per expert and sum over the
     n_expert_used dim; residual add;
   - final RMSNorm `output_norm`; lm_head `ggml_mul_mat(output,cur)`.
   Allocated with `ggml_gallocr`, executed on the CPU backend
   (`ggml_backend_cpu_init`, `ggml_backend_cpu_set_n_threads(backend,1)` for
   determinism) via `ggml_backend_graph_compute`. The weight tensors live in the
   data-owning context and are used directly as graph leaves (gallocr skips
   already-allocated tensors).
3. COMPARE: max/mean absolute logit delta over the vocab and the greedy argmax.

## Build and run

Standalone, NOT via CMake (prebuilt static libs under `build/debug/lib`):

```
g++ -std=c++17 -fopenmp -I third_party/llama.cpp/include \
    -I third_party/llama.cpp/ggml/include -I third_party/llama.cpp/ggml/src \
    spike/engine_core/*.cpp -o spike/engine_core/spike_engine \
    -L build/debug/lib -lllama -lggml -lggml-cpu -lggml-base -lpthread -lm

./spike/engine_core/spike_engine /projects/sandbox/moe.gguf
```

The binary (`spike/engine_core/spike_engine`) is gitignored; only the source and
this README are tracked.

## Result (measured)

Scope achieved: FULL forward (both transformer blocks + attention + MoE FFN +
final norm + lm_head), NOT the single-block fallback.

- engine argmax = 71, oracle argmax = 71 -> argmax MATCH
- max  |delta| over the 264-wide vocab = 5.3167e-5
- mean |delta| over the vocab          = 1.2078e-5
- deterministic: identical numbers across repeated runs (1 CPU thread)

PASS (`SPIKE PASS`, exit 0): the engine's own ggml graph reproduces llama's
greedy argmax exactly and the full logit vector to ~5e-5. No divergence found.

The tiny residual delta (~5e-5, not bit-exact) is expected and benign. It is NOT
caused by flash attention. The spike includes a probe: it re-runs the oracle with
flash attention forced off (`flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED`)
and compares to the default `AUTO` run. On this CPU/F32 fixture the two oracle
logit vectors are byte-identical (max delta 0.000000000), so flash attention is
not selected in a way that affects the result and cannot be the residual source
here. The residual instead comes from op ordering / fusion / accumulation
differences between the hand-built graph and llama's graph builder (both run
explicit F32 `mul_mat` + `soft_max_ext` + `mul_mat` attention; the reference
non-flash `build_attn_mha` branch forces `GGML_PREC_F32` accumulation on `kq`).
The agreement to 5e-5 confirms the math is equivalent, including the MoE top-2
routing through `ggml_mul_mat_id` over the stacked expert tensors - the part most
at risk.

## Divergence instrumentation

The spike reads back per-layer intermediates (after attention+residual and after
MoE+residual, for both layers) and prints their min/max/mean and flags any
non-finite values. On an argmax mismatch these localize the first divergent
block. On the current fixture all intermediates are finite and in range, and the
final argmax matches, so no divergence localization was needed.

## Conclusion

Owning the forward pass with a self-built ggml graph is feasible: a hand-built
graph reproduces llama's llama-arch MoE forward (greedy token identical, logits
to ~5e-5) on this model. This de-risks the engine-core design.

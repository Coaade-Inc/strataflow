# Research and Feasibility: a 50B-class MoE on a normal laptop

**Owner: Coaade Inc.**

> **Scope.** This is a research + feasibility design doc, not an implementation.
> It names the two hard ceilings that stop a ~50B-class Mixture-of-Experts (MoE)
> model from running usefully on an everyday no-GPU laptop, analyzes each
> candidate technique for breaking them (mechanism, 50B-on-12GB memory and
> throughput math, oracle-safety, complexity, feasibility verdict), gives a
> concrete memory-budget and throughput model, and ends with a sequenced,
> impact-ordered implementation plan. The orchestrator will drive implementation
> of the feasible techniques from this doc, one oracle-gated PR at a time.
>
> This doc refines and connects to the existing roadmap; it does not contradict
> it. Read alongside [`ROADMAP.md`](./ROADMAP.md), [`PLAN.md`](./PLAN.md),
> [`ENGINE_CORE_DESIGN.md`](./ENGINE_CORE_DESIGN.md), and
> [`RESEARCH.md`](./RESEARCH.md).

---

## 0. The target machine and the mission

The target is a normal laptop, and we calibrate it to the free Colab CPU box,
because that box is both representative and the machine where the real runs
below were measured:

- ~12 GB usable RAM
- 2 CPU cores
- SSD at roughly 4 GiB/s sequential and comparable large-random throughput, QD1
- no GPU

The mission (from `ROADMAP.md`): run a very large MoE in a bounded RAM budget by
streaming experts from disk, in any GGUF weight type, with no GPU required, and
with output byte-identical to the libllama oracle wherever we claim exactness.
This doc narrows that mission to a concrete, researcher-level question: **what
has to change for a ~50B-class MoE to both FIT and run at a usable speed on the
12 GB / 2-core / no-GPU box?**

The user explicitly asked for out-of-the-box thinking, including using the SSD
itself as a virtual-RAM tier so that lack of RAM stops being a hard capability
ceiling. We take that seriously and formalize it (Section 4), while being honest
about what it does and does not buy.

---

## 1. The calibration run: real Mixtral on the free Colab box

These are measured facts from real runs through StrataFlow. They are the
calibration point for every estimate in this doc. We do not invent new numbers;
we extrapolate from these and state assumptions when we do.

Model: Mixtral-8x7B-Instruct, Q3_K_M, packed to `.strata`, ~19 GB on disk.
Shape: 32 layers, 8 experts/layer, top-2 routing.

Measured on the 12 GB / 2-core / ~4 GiB/s SSD / no-GPU box:

| Quantity | Measured |
|---|---|
| Output | coherent, correct greedy decode end to end |
| Resident StrataFlow model weights (trunk only) | ~939 MiB |
| Peak process RSS | ~10.6 - 10.7 GB |
| Bytes streamed from SSD (16 tokens, after PR #39) | ~40 GB (was ~51 GB before #39) |
| Time to first token (TTFT) | ~52 s |
| Decode throughput | ~0.06 - 0.07 tok/s |
| Async prefetch `completed-before-use` | 0 |

Two facts in that table are in tension and define the whole problem:

1. StrataFlow's own resident model weights are tiny (~939 MiB) because the
   bounded SlotPool keeps only the hot experts resident. Yet peak RSS is ~10.6
   GB. **Something other than resident experts is eating the RAM.**
2. For 16 tokens the engine streamed ~40 GB, which is *more than twice* the
   entire ~18 GB expert set. **The hot working set does not fit, so the engine
   re-reads most of the model every handful of tokens.**

> **Peak-RSS reconciliation (one-liner).** The ~10.6 GB above is NOT an engine
> regression from the early runs: early / small and dense models genuinely
> peaked low (~132 MiB on the generated-MoE demos, 785 MiB and 1177 MiB models
> both at ~132 MiB, the real TinyLlama Q4_K_M dense run well under a couple of
> GB). The one-time jump to ~10.6 GB came only with this real large Mixtral at
> its FULL `n_ctx` 32768, where the F32 KV reserved 8192 MiB (8 GiB) - most of
> the peak. KV paging (`STRATAFLOW_ENGINE_KV_WINDOW`, Ceiling A below) drops
> that to ~128 MiB at `W` 512, byte-identically, bringing the real large-model
> peak back toward a projected ~2.5 GB (Colab-pending, not a sandbox
> measurement). See `ROADMAP.md` "Peak RSS over time (reconciliation)".

`completed-before-use = 0` is not a bug. With 2 cores the per-layer compute
window is smaller than the time to read the next layer's experts off the SSD, so
the background prefetch read never finishes before the authoritative load needs
it. I/O cannot hide behind compute on this hardware. This is a structural
property of 2-core + SSD-bound streaming, and the async worker already
self-backs-off when it observes it (PR #39). Any throughput plan has to assume
I/O overlap buys nothing here and win by reading *fewer / larger* bytes instead.

### 1a. From three manual knobs to one dial: `--target-ram-gb`

The FIT levers in this doc (the KV window of section 6.5, the adaptive trunk
streaming of section 6.4, and the bounded expert cache) were first proven by
hand on the real Colab box. On the same real Mixtral-8x7B-Instruct Q3_K_M
`.strata` (32 layers, 8 experts top-2, `n_ctx` 32768, ~19 GB on disk), setting
the three low-level knobs together, `STRATAFLOW_ENGINE_KV_WINDOW=2048
--expert-slots 8 --cache-gb 1`, brought the peak process RSS down to ~2.4 GiB
(2406.7 MiB) with output BYTE-IDENTICAL to the full-resident decode. That proved
the levers work, but it asked the user to reason about three separate numbers.

StrataFlow now folds those three knobs into ONE dial. `--target-ram-gb 2` (ABI
`target_ram_budget`, bytes) is used directly as the usable budget, and the
planner drives the KV window, the adaptive per-layer trunk streaming, and the
expert cache from it in one decision, so the same ~2.4 GiB manual experiment is
now a single number the planner resolves. An explicit target SHRINKS the KV
window (the power-of-two step-down to the smallest safe window) and engages
trunk streaming to hit the number; it fits achievable small targets rather than
refusing them. If a target is genuinely below the achievable floor the planner
refuses LOUDLY, and the refusal names the HONEST minimal-KV floor (the smallest
safe window + maximal trunk streaming + the top-k experts) - for real Mixtral
that true floor is ~2.x GiB (the user already proved 2406 MiB decodes at a
2048-column window), NOT the full-`n_ctx` ~10 GiB figure the old estimate
reported. For the common case the user does not even pick a number:

- **Auto (the default, `--ram-mode auto`, ABI `ram_mode == SF_RAM_MODE_AUTO ==
  0`)** is the good-neighbor policy. It takes only the conservative `usable_ram`
  share of RAM (`free - max(1 GiB, 12.5%)`, capped at 85% of total), leaving
  headroom so other apps on a shared laptop keep running while the model
  decodes. It pulls the resident set down (KV window + trunk streaming) only as
  far as that share requires.
- **Max (`--ram-mode max`, or `--max-performance`, ABI `ram_mode ==
  SF_RAM_MODE_MAX == 1`)** is the opposite end: dedicate the machine. It sizes
  its usable ceiling from ALL PHYSICAL RAM (`ram_total_bytes`), NOT the
  instantaneous free RAM, and holds the MAXIMUM experts resident with a full KV
  window and the trunk fully resident, streaming the LEAST. Sizing from total
  physical RAM makes Max MONOTONIC against Auto on the same machine:
  `max.expert_slots_resident >= auto.expert_slots_resident`, Max's KV window is
  at least Auto's (full when physical RAM affords it), and Max streams the trunk
  no harder than Auto (`max.streamed <= auto.streamed`). Max is never inverted:
  it never ends up with FEWER residents or MORE streaming than Auto. (The earlier
  behavior sized Max from the instantaneous free RAM, so on a box where the big
  model had already consumed RAM, Max resolved a tiny ceiling and held fewer
  experts than Auto - that inversion is fixed.)

Precedence (highest wins): the explicit low-level knobs
(`STRATAFLOW_ENGINE_KV_WINDOW` / `--expert-slots` / `--cache-gb`) still win for
power users; then `--target-ram-gb`; then `--ram-mode max`; then auto.

GPU honesty, to be exact: auto/max may read VRAM and the planner budgets VRAM
PLACEMENT, but the authoritative `.strata` engine decode is CPU-only today, so
full GPU-accelerated COMPUTE is the enterprise roadmap track, not a claim here.
The RAM side is the measured mechanism.

The projected peak the planner reports is now a RECALIBRATED, honest estimate.
It was previously ~2x the engine's real resident footprint (it printed "~10 GiB"
for an auto run that really peaked ~5 GiB, and "~11 GiB" for a max run that
really peaked ~2.2 GiB, because the conservative staging + compute + safety
margin stacked on top of a full-KV charge). That over-estimate is WHAT made the
explicit-target path refuse achievable targets. The estimate now tracks the
engine's real windowed resident footprint (resident KV at the chosen window +
resident experts + one-layer staging + trunk + process floor + a trimmed
margin): slightly high so the fit invariant stays safe, no longer ~2x.

What is proven where: the ~2.4 GiB absolute number on the real Mixtral is the
user's Colab result and the `--target-ram-gb 2` equivalence is a Colab re-run to
confirm (a projection). The sandbox proves the MECHANISM, the mode differences,
the honest refusal floor, estimate sanity, and byte-identity through the REAL
load path on a Mixtral-geometry fixture (`n_ctx` 32768, 32 layers, pack ->
`.strata` -> `load_model` -> planner -> engine). On a tight 3 GiB-total /
2.25 GiB-free simulated profile (free << total, so auto and max DIVERGE): AUTO
resolves usable 1280 MiB, steps the KV window down to 8192 (`< n_ctx`), holds
`slots=104`, and streams some experts; MAX resolves usable 3072 MiB (= all
physical RAM, the fix), keeps a FULL KV window (`kv_window=0`), holds `slots=256`
(the full working set), and streams nothing - so `max.slots > auto.slots`, Max's
window is >= Auto's, and only Auto streams (Max monotonic, not inverted). An
explicit achievable small target (~1.2 GiB) still FITS by shrinking the KV
window + engaging trunk streaming (`fits=true`, `kv_window` in `(0, n_ctx)`,
`trunk_pool_layers > 0`), while a genuinely-too-small target refuses with a
floor built from the SMALLEST safe window (the refusal names a KV figure far
below full KV, e.g. "KV window 256", not the full-`n_ctx` 2048/8192 MiB). The
greedy decode is byte-identical (`|delta| = 0.0`) across explicit-target, auto
and max, and within the soft oracle bound for the random-weight fixture. The
exact Colab commands and expected plan/stats for all three modes are in
[`../colab/README.md`](../colab/README.md).

---

## 2. The two ceilings

### Ceiling A - the fixed resident floor (why peak RSS is ~10.6 GB)

Resident experts are small and bounded. Peak RSS is dominated by a fixed
resident floor that does NOT shrink with `--expert-slots`:

- **Fully-resident trunk.** On both the GGUF and `.strata` paths the trunk
  (attention projections, norms, embeddings, output head, routers for all
  layers) is held fully resident in a ggml weight context (`wctx`), filled once
  at load. See `Engine::load_strata_weights` in
  `src/model/engine/engine.cpp`: the trunk tensors are allocated resident via
  `ggml_backend_alloc_ctx_tensors_from_buft` and filled from the trunk block;
  only the expert tensors live in the unbacked `ectx` and stream. The trunk is
  NOT streamed today.
- **Resident KV cache (now windowed; was fully resident).** Before FEAT-002,
  `src/model/engine/engine.cpp` allocated, in a persistent `kvctx`, one K and one
  V tensor per layer shaped `[head_dim, n_head_kv, n_ctx]` as F32, a fixed
  resident cost that grew with `n_ctx`. FEAT-002 (technique 6.5, lossless KV
  paging) sizes those tensors to a recent WINDOW `W` (ring buffer,
  `[head_dim, n_head_kv, W]`, default `W == n_ctx`), reserving only `W/n_ctx` of
  the old resident cost, and tiers cold columns to a RAM store (BOUNDED - empty
  at load, grown on demand only to the positions actually evicted, never the full
  `n_ctx`), re-assembling them losslessly for attention. `KvStore`
  (`src/kv/kv_store.h`) now owns the paging bookkeeping (window, ring mapping,
  cold/resident split). A real-model empty-output bug - the cold store was first
  pre-allocated at full `n_ctx` (8 GiB on Mixtral) and OOMed at load when a
  sub-`n_ctx` window was set - was found on hardware and fixed by the
  grow-on-demand bound. KV quantization is still deferred (approximate).
- **The process floor.** `libllama` is still linked (for the tokenizer / vocab;
  see the `llama_*` includes in `src/model/engine/engine.cpp` and
  `src/model/ggml_model.cpp`) and, with the ggml backend and vocab, contributes
  a fixed ~132 MiB floor measured in-sandbox (a 3.9 MiB model and a
  `--plan-only` run both measure ~132 MiB; see `ROADMAP.md`
  "Per-layer staging no longer scales with model size").
- **One-layer staging + compute buffer.** A single reused staging buffer sized
  to one layer's FULL `n_expert` footprint (not the slot budget) plus the ggml
  compute buffer. Bounded, but non-trivial for a large layer.

For Mixtral Q3_K_M the trunk is ~939 MiB and the floor is small, so on paper the
resident floor looks like ~1-2 GB. The measured ~10.6 GB peak RSS tells us the
rest is the OS page cache plus transient allocations around the ~40 GB of
streaming (buffered reads, staging churn) sitting on top of the floor. For the
budget model we treat the *controllable* resident floor as trunk + KV + process
floor + one-layer staging + compute buffer, and we treat uncontrolled page-cache
growth as a thing to suppress with direct / unbuffered I/O (Section 7).

The point for a 50B target: a 50B-class MoE has a much larger trunk and (at
useful context) a much larger KV cache than Mixtral-8x7B. If the trunk alone
does not fit in a few GB, the model does not load at all, regardless of how well
experts stream. **The resident floor is the FIT ceiling.**

### Ceiling B - per-token byte traffic (why it is ~0.06 tok/s)

Decode throughput on this box is bounded below by:

```
seconds_per_token  >=  bytes_streamed_per_token / ssd_bandwidth
                       + compute_per_token / cores
```

With ~40 GB over 16 tokens that is ~2.5 GB/token. At ~4 GiB/s that alone is
~0.6 s/token of pure I/O in the best case; the measured ~15 s/token says the
real effective streaming rate is far below peak sequential bandwidth, because
the reads are many smaller blob reads at low queue depth, not one big sequential
gulp, and because `completed-before-use = 0` means none of it overlaps compute.

The root cause of the huge per-token traffic: the model's hot expert working set
exceeds the number of expert bundles that fit in 12 GB. For Mixtral, ~18 GB of
experts means only ~121 of 256 bundles fit in RAM, so more than half the experts
touched each token are cold and must be re-streamed. **Per-token byte traffic is
the SPEED ceiling.**

Layer-stratified residency + prefetch gating (PR #39) already cut this (37%
fewer streamed bytes at a tight budget, async no longer increasing traffic), but
it cannot change the fundamental fact that the working set does not fit. To go
further we must either make the resident set hold more of the working set (fit
more experts in the same RAM) or read fewer bytes per expert (sub-expert
streaming) or touch the big model on fewer steps (speculative decode).

---

## 3. 50B-class extrapolation assumptions

Mixtral-8x7B has no single "50B-class" successor shape, so we state explicit
assumptions and carry them through. We model a representative 50B-class MoE as:

- Total parameters ~50B, quantized ~3.5-4 bits/param -> ~22-25 GB on disk.
- MoE with a modest active fraction (e.g. ~8-16 experts/layer, top-2), active
  params per token ~5-7B, consistent with the gpt-oss-120b-class row in
  `PLAN.md` section 2 scaled down.
- Trunk (attention + norms + embeddings + output + routers across all layers)
  ~3-5 GB at the shipped quant. This is the number that must drop for the model
  to FIT; it is larger than Mixtral's ~0.9 GB trunk because 50B has more / wider
  layers.
- KV cache at a useful context (say 8K-32K): F32 per-layer K and V as today puts
  this in the multi-GB range and growing with context.

These are assumptions, not measurements. The budget model (Section 5) is
expressed symbolically so the verdicts hold as the real 50B shape firms up; the
Mixtral row stays the only measured calibration.

---

## 4. The SSD-as-virtual-RAM idea, done right

The user's framing: instead of needing more RAM, use the SSD as a virtual-RAM
tier so a bigger model can run. This is correct in spirit and is in fact already
the core of what StrataFlow does for experts. The important part is doing it as
a MODEL-AWARE tier, not as naive OS swap or `mmap`-as-generic-RAM.

### Why naive OS swap / mmap thrashes

If you `mmap` the whole 25 GB model and let the OS page it in on fault, or add a
25 GB swapfile and let the kernel evict, you get:

- **Random, page-granular, QD1 faults.** A page fault pulls 4 KiB (or a small
  readahead window) at a time, synchronously, at the worst point in the SSD's
  curve (small random QD1). Our weights want large aligned contiguous reads
  where the SSD runs near its ~4 GiB/s sequential number.
- **No routing knowledge.** The kernel evicts by generic LRU / working-set
  heuristics. It does not know that this expert will not be touched again for 30
  layers, or that the trunk is hot every token. It will happily evict a hot
  trunk page to make room for a cold expert it faulted on.
- **Page-cache double-buffering.** Buffered reads land in the page cache, which
  then competes with our own RAM budget - exactly the ~10.6 GB RSS blowup
  pattern. `PLAN.md` section 3.3 already calls for direct I/O that bypasses the
  page cache for this reason.

### What the model-aware tier is

StrataFlow's existing design already is a userspace, MoE-routing-aware virtual
RAM for *experts*: the `.strata` file is the authoritative weight store on disk
(`StrataReader`, `src/tws/strata_file.h`), reads are large aligned per-expert
blobs via `read_blob` (direct-I/O capable `BlockFile`), eviction is driven by
our router read-back and predictor (layer-stratified `SlotPool`,
`src/tws/weight_store.h`), and residency is bounded to a RAM budget. The
authoritative synchronous load keeps it byte-identical to the oracle.

The "SSD as a proper tier done right" program is to extend this same
model-aware paging from experts to EVERY weight class that currently sits in the
fixed resident floor: the trunk (Section 6.4) and cold KV pages (Section 6.5),
with large aligned reads and routing/position-aware eviction, never OS swap.

**Honest framing.** This REMOVES the hard RAM ceiling for weights: a 50B model's
weights can live on SSD and be paged in model-aware, so the box can *load and
run* a model whose weights far exceed RAM. It does NOT by itself raise token
speed; if anything, paging more weight classes adds bytes to move. Its value is
that it is the framing that makes the FIT levers possible and lets the SPEED
levers (which cut bytes/token) operate on a model that otherwise would not fit
at all. Verdict: this is not a single feature but the organizing principle; the
concrete features that realize it are "stream the trunk" and "page the KV",
analyzed below.

---

## 5. The 50B-on-12GB memory budget model

All figures below are for the representative 50B-class MoE of Section 3 on the
12 GB box. The goal is to get the resident floor comfortably under ~6 GB so
there is headroom for staging, compute, and the OS, while cutting per-token
streaming. "Resident" means must-be-in-RAM; everything else lives on SSD and
streams.

### Before (today's residency model applied to 50B)

| Component | Resident today | Note |
|---|---|---|
| Trunk (attn/norm/embd/output/routers, all layers) | ~3-5 GB | fully resident in `wctx` |
| KV cache (F32, useful context) | ~2-4 GB and growing | fully resident `kvctx` |
| Process floor (libllama + ggml + vocab) | ~0.13 GB | fixed |
| One-layer expert staging | ~0.3-1 GB | one layer's full `n_expert` footprint |
| Compute buffer | ~0.2-0.5 GB | ggml gallocr |
| Resident expert cache | bounded, small | SlotPool |
| **Resident floor subtotal** | **~6-11 GB** | **may not fit; trunk + KV dominate** |

A 50B trunk + full KV can approach or exceed the whole 12 GB before a single
expert is cached. **Conclusion: today's residency model does not fit a 50B on 12
GB. The FIT levers are mandatory, not optional.**

### After (apply the FIT levers)

| Component | Resident after | Lever |
|---|---|---|
| Trunk | bounded resident hot set (a few layers); rest streamed per token | 6.4 stream the trunk (SHIPPED, lossless, budget-driven adaptive) |
| KV cache | ~0.5-1.5 GB (recent window `W` resident; cold columns tiered to RAM, re-assembled losslessly) | 6.5 KV paging (SHIPPED, lossless); SSD spill + quant deferred |
| Process floor | ~0.02-0.05 GB | 6.6 tiny tokenizer, drop libllama |
| One-layer staging | ~0.3-1 GB | unchanged (bounded already) |
| Compute buffer | ~0.2-0.5 GB | unchanged |
| Resident expert cache | fill remaining budget (several GB) | more RAM for experts |
| **Resident floor subtotal** | **~1.5-4.5 GB floor + experts filling to ~6-10 GB** | **fits with headroom** |

The key move is that trunk-streaming and KV-paging convert multi-GB *fixed*
resident costs into *bounded, streamable* costs, which (a) makes the model fit
and (b) frees several GB to enlarge the resident expert cache, which directly
attacks Ceiling B by holding more of the hot working set.

### The budget is enforced by construction (conservative, fit-by-construction)

The table above is the target shape of the floor; the planner now MAKES it hold
rather than hoping it does. Shipped in FEAT-002 (`src/plan/planner.cpp`), the
auto placement planner sizes the resident expert cache from what is LEFT after
EVERY other mandatory resident cost is subtracted from a conservative usable-RAM
ceiling, and treats its own projected peak as a hard bound. The components are:

- **usable_RAM** = `min(ram_free - max(1 GiB, 12.5%), 85% of ram_total)`. This is
  deliberately conservative because MemAvailable (`ram_free`) on a shared Colab
  box over-reports what is really free, so the budget also caps at 85% of
  physical `ram_total`.
- **trunk resident** - the hot trunk bytes placed in RAM (edge tensors already
  streamed per 6.4).
- **resident KV at the chosen window** - `n_layer * 2 * (head_dim * n_head_kv) *
  4 * W` for the resident window `W` (full `n_ctx` if no window), matching the
  engine's `resident_kv_bytes()`. If the shape fields are unknown the planner
  uses a conservative fallback and never under-counts.
- **one-layer expert staging at FULL n_expert** - `n_expert * expert_bytes`, NOT
  `n_experts_used * expert_bytes`, because the engine's `stage_gate/up/down`
  tensors are sized to a full layer's expert count.
- **compute buffer** - a conservative ggml estimate (a floor plus a
  window/shape term, erring high, since `ggml_gallocr_alloc_graph` grows it on
  demand with no fixed reserve).
- **process floor** - at least ~160 MiB (the fixed llama/ggml + vocab floor
  Section 1 and the ROADMAP measure at ~132 MiB).
- **safety margin** - `max(512 MiB, 8% of usable_RAM)`, held fixed.

Only `available_for_experts = usable_RAM - (sum of the above)` divided by
`expert_bytes` gives the resident bundle count, clamped to
`[n_experts_used, full_slots]`. This REPLACES the old
`ram_for_experts = vram_left + ram` (all remaining RAM after only the trunk,
which picked ~122 bundles on Mixtral). Two invariants follow:

- **Honest peak.** `planned_peak_bytes` is now the SUM of all those components,
  a conservative upper bound - not the old dishonest trunk+experts-only figure
  that printed "est. peak ~9 GiB" on a 12 GiB box and still got OOM-killed
  (exit status 137 / SIGKILL) before any output.
- **Fit by construction.** For any non-refused plan, `planned_peak_bytes <=
  usable_RAM`; the planner reduces the resident bundle count down to the
  `n_experts_used` floor rather than ever over-committing.

The KV window interacts with the budget cleanly: when full-`n_ctx` KV would make
the floor infeasible, the planner picks a LOSSLESS auto window `W` (the largest
power-of-two `>= 256` that fits) and the KV saving LOWERS the projected peak - it
is NOT re-consumed by a greedy re-expansion of the expert cache to the brink (the
safety margin stays held). An explicit `STRATAFLOW_ENGINE_KV_WINDOW` still wins.
If even the minimal config (`n_experts_used` bundles + the smallest safe window +
trunk) cannot fit, the planner sets `fits=false` with a clear ASCII reason, the C
ABI returns an error, and the CLI exits non-zero - a loud refusal instead of a
silent SIGKILL. On the synthetic Mixtral-on-12GB shape this takes the plan from
~122 bundles / dishonest "~9 GiB" (OOM) to `slots=19`, auto `kv_window=4096`,
honest projected peak 9495 MiB `<=` usable 9497 MiB; proven by
`test_mixtral_12gb_conservative_budget` in `tests/test_planner.cpp`.

### SPEED levers' effect on bytes/token

With more RAM freed for experts, a larger fraction of the ~22-25 GB expert set
is resident, so fewer cold misses per token. The sub-expert and sub-byte levers
cut the remaining traffic further:

- **Sub-byte resident experts** (2-bit in RAM, dequant-at-compute): 2-3x more
  experts fit in the same RAM, so cold-miss rate drops roughly proportionally.
- **Sub-expert / activation-sparse streaming**: read only the contributing rows
  / columns, cutting bytes per *streamed* expert, potentially several-fold.
- **Sequential batched I/O**: same bytes but nearer the ~4 GiB/s sequential
  number instead of the measured low-QD1 effective rate, cutting wall-clock per
  byte.
- **Compress what streams** (6.7b): fewer bytes on disk per expert, decompressed
  at the read seam; lossless. Modest-to-none on already-K-quantized experts
  (~1.0 on the fixture); the real win scales with the trained-model entropy.
- **Speculative decode**: the expensive expert streaming happens on a fraction
  of steps (verification bursts), cutting *effective* bytes/token across the
  sequence.

---

## 6. Technique-by-technique analysis

Each technique: mechanism, 50B-on-12GB math, oracle-safety (can it stay
byte-identical, or is it approximate), complexity + dependencies, and a verdict:
`feasible-now`, `feasible-with-effort`, `research-risky`, or `not-worth-it`.

"Lossless" below means byte-identical to the libllama oracle and buildable under
the existing correctness gate. "Approximate" means it changes outputs and must
be validated with a quality metric (perplexity / KLD / accuracy vs oracle)
instead of byte-identity.

### 6.1 SSD-as-a-proper-tier (model-aware weight paging)

- **Mechanism.** Section 4: SSD is the authoritative store; large aligned blob
  reads; router/predictor-driven eviction; never OS swap. Already true for
  experts; extend to trunk and KV.
- **50B math.** Removes the hard RAM ceiling for weights: a 25 GB weight file
  runs on 12 GB. No direct speed effect on its own.
- **Oracle-safety.** Lossless (pure residency / byte provenance, same bytes same
  ops), the same property that keeps expert streaming byte-identical today.
- **Complexity / deps.** This is the framing; realized by 6.4 and 6.5. No new
  standalone work beyond those.
- **Verdict: feasible-now (as the organizing principle).** The concrete,
  shippable pieces are 6.4 and 6.5.

### 6.2 Sub-expert / activation-sparse streaming

- **Mechanism.** Stream only the rows / columns of an expert's gate/up/down that
  actually contribute for this token, grounded in the published activation-
  sparsity line of work: Apple "LLM in a Flash" (load only predicted-active
  neuron columns), Deja Vu (contextual sparsity predictors), PowerInfer
  (hot/cold neuron split). `PLAN.md` sections 3.8 and 3.10 already anticipate
  row-column bundling and neuron-bundle reads.
- **50B math.** This is the single biggest lever on Ceiling B. If only ~30-50%
  of each expert's neurons are active for a token, per-expert bytes drop
  2-3x, cutting the ~2.5 GB/token traffic toward ~1 GB/token and shrinking the
  resident footprint of a cached expert correspondingly.
- **Oracle-safety.** This is where byte-identity breaks, and the doc is explicit:
  - Streaming only the experts the router actually selected is LOSSLESS and is
    *already what we do* (top-k residency). That is not this technique.
  - Predicting / pruning *intra-expert* rows is APPROXIMATE: dropping a row that
    turns out to contribute changes the output. There is no lossless variant of
    *predicted* intra-expert sparsity. A lossless variant exists only if the
    activation is exactly zero (e.g. a true ReLU producing exact zeros, which we
    can detect post-activation) - but Mixtral-class SwiGLU/SiLU experts do not
    produce exact structural zeros, so for our target this is inherently lossy.
  - Validation path for the approximate case: measure perplexity / KLD vs the
    oracle on a held-out set at each sparsity threshold, report the quality cost,
    and ship it as an opt-in "fast mode" (consistent with `PLAN.md` principle 4
    and section 3.8's calibrated thresholds), never as the default exact path.
- **Complexity / deps.** High. Needs a per-layer sparsity predictor or threshold
  calibration, sparse GEMV kernels that skip zeroed input channels, a `.strata`
  layout that allows sub-blob row/column reads (today a bundle is one aligned
  blob), and a quality-regression harness. Depends on the quality-metric gate
  existing.
- **Verdict: research-risky (highest-impact-but-approximate).** Biggest
  bytes/token win, but it is lossy for our model family, needs new kernels and a
  new quality gate, and the aligned-blob `.strata` layout would need sub-blob
  addressing. Sequence it after the lossless levers, behind an opt-in flag with
  a measured perplexity cost. De-risk with a spike (Section 9).

### 6.3 Sub-byte resident weights with compute-time dequant

- **Mechanism.** Hold experts in RAM at a lower bit-width than disk (e.g. 2-bit
  trellis / LUT) so 2-3x more experts fit resident; dequantize to the compute
  type at matmul time. Ties to Phase 6 (T-MAC/LUT low-bit CPU kernels) and Phase
  7 (trellis 2-bit) in `PLAN.md`/`ROADMAP.md`.
- **50B math.** If experts resident double or triple, the cold-miss rate per
  token drops roughly proportionally, directly cutting Ceiling B's bytes/token.
  E.g. if 50% of the working set fits today, 2x density could fit ~all of it,
  collapsing re-streaming.
- **Oracle-safety.** Two cases:
  - If the in-RAM sub-byte format is a *different* quantization than the model
    shipped, it is APPROXIMATE (re-quantizing Q3_K experts to 2-bit changes the
    numbers) and needs the quality-metric gate.
  - If the model already ships at that sub-byte format, computing it natively is
    LOSSLESS and is just "support this quant family" (the engine is already
    type-agnostic; it stages each expert at its source ggml type). That is quant
    coverage, not this technique.
- **Complexity / deps.** High on this hardware: dequant costs CPU, and CPU is the
  scarce resource on 2 cores. A 2-bit LUT kernel must be fast enough that the
  extra dequant does not eat the I/O savings. Needs the T-MAC/LUT kernels
  (Phase 6) which do not exist yet.
- **Verdict: feasible-with-effort, but approximate and CPU-gated.** Real win on a
  box with spare CPU; on 2 cores the dequant cost competes with the I/O savings,
  so it must be measured. Sequence after the LUT kernels exist. Lower priority
  than the lossless FIT levers.

### 6.4 Stream the trunk (not just experts) [SHIPPED, lossless, ADAPTIVE]

- **Mechanism [SHIPPED].** The trunk was fully resident (`wctx`) and a big part
  of the resident floor; for a 50B it may not fit. The trunk now streams in two
  tiers, both reusing the v2 `.strata` per-trunk-tensor index
  (`TrunkIndexEntry`, `StrataReader::read_trunk_tensor`) and the engine's
  resident-before-op discipline:
  - **Edge tensors (PR #41):** the two LARGE edge-touched tensors
    (`token_embd.weight`, `output.weight`), each `n_vocab x n_embd` and read
    once per token, live metadata-only in `tctx` with a single shared working
    buffer, fetched immediately before the segment that reads them.
  - **Per-layer attention projections (FEAT-003, ADAPTIVE):** the four large
    per-layer projections (`blk.N.attn_{q,k,v,output}.weight`), the dominant
    remaining resident cost after the KV window, live metadata-only in `pltctx`
    and are fetched into a BOUNDED HOT-TRUNK POOL (a ring of `trunk_pool_layers`
    layers) immediately before each layer's attention segment. The per-layer
    norms + router are tiny and stay HOT (resident).
- **Budget-driven aggressiveness.** The planner decides `trunk_pool_layers` from
  the SAME conservative usable-RAM budget: a big-RAM box keeps the whole trunk
  resident (`trunk_pool_layers == 0`, stream nothing, zero per-token trunk
  traffic); a tight 12 GB-class box streams the per-layer attention trunk hard
  (pool as low as 1 layer). The decision is plumbed to `Engine::load` as a load
  parameter (via `GgmlModel`), exactly like the FEAT-002 KV window - NOT an env
  side-channel - so it applies BY DEFAULT on the real `.strata` load path.
- **50B math.** Converts the ~3-5 GB fixed trunk (on real Mixtral the per-layer
  attention trunk alone is ~5.5 GB across 32 layers) into a bounded resident hot
  set (a few layers) plus streamed cold trunk, heading the honest projected peak
  toward ~2-3 GB on a 12 GB box. This is the dominant FIT lever and is MANDATORY
  for a 50B to load at all on 12 GB.
- **HONEST TRADEOFF.** Per-layer attention weights are touched EVERY token
  (unlike cold experts), so streaming them adds per-token SSD traffic and is
  SLOWER on the 2-core / ~4 GB/s box. This is accepted because the user's FIRST
  goal is to FIT small: the budget-driven policy streams only as much trunk as
  the budget forces, so a machine with headroom pays nothing. The streamed trunk
  bytes are counted by `StrataReader::streamed_trunk_bytes()`, kept DISTINCT
  from the expert `streamed_bytes()` so the per-token expert number is unchanged.
- **Oracle-safety.** LOSSLESS. Pure residency / byte provenance, exactly like
  expert streaming: same bytes, same ops, same graph order, byte-identical to
  the oracle. The authoritative resident-before-op synchronous read is the
  correctness backstop; the fetch stays on the single compute thread (the async
  I/O worker is never involved, so no new TSan surface). The
  `STRATAFLOW_ENGINE_NO_TRUNK_STREAM=1` knob forces the whole trunk resident so
  a test can decode the SAME v2 file both ways and assert bit-identity.
- **Sandbox-proven.** On a Mixtral-mirroring fixture through the REAL load path
  (packer -> `.strata` -> `load_strata_model` -> planner -> engine): the plan
  chooses `trunk_pool_layers` in `(0, n_layers)` by default under a tight
  budget, `resident_weight_bytes()` drops to the bounded hot set, and the
  streamed decode is bit-identical (`max|delta| == 0.0`) to the fully-resident
  decode and within the oracle quant tolerance over a short AND a longer
  sequence (`tests/test_trunk_stream_adaptive.cpp`). The absolute real-Mixtral
  peak-RSS drop is Colab-only (the offline sandbox cannot download Mixtral).
- **Verdict: SHIPPED (lossless, high-impact, FIT).** The top FIT lever: lossless,
  reuses existing streaming machinery, budget-driven, and without it a 50B does
  not fit.

### 6.5 Page / quantize the KV cache

- **Mechanism.** KV grows with context and was pure resident cost before this
  lever (`kvctx`, F32, `[head_dim, n_head_kv, n_ctx]` per layer). Two independent
  sub-levers (Phase 6, `PLAN.md` section 3.9):
  - **KV paging [SHIPPED, lossless]**: keep a recent-window `W` of KV resident
    (the per-layer `kvctx` tensors are now sized `[head_dim, n_head_kv, W]`, a
    ring buffer, instead of the full `n_ctx`, so the resident reservation is just
    `W` columns, i.e. `W/n_ctx` of the full cost), tier cold KV columns to a
    RAM-backed per-layer cold page store (SSD spill is the deferred follow-on,
    option b), and RE-ASSEMBLE the full history losslessly for attention so
    every past position `0..pos` is still attended over with the SAME F32 bytes.
    The cold store is BOUNDED: it is NOT pre-allocated at `n_ctx` - it is empty
    at load and GROWS ON DEMAND, sized only to the positions actually evicted
    (never the full `n_ctx`), so engaging paging on a large-`n_ctx` model does
    not cost a multi-GB up-front allocation. The window `W` defaults to the full
    `n_ctx` (so the DEFAULT path is byte-identical and zero-overhead) and is
    overridable via `STRATAFLOW_ENGINE_KV_WINDOW` to engage paging. Shipped in
    FEAT-002. A follow-up fixed a real-hardware regression: the cold store was
    first pre-allocated at full `n_ctx` (8 GiB on Mixtral: `n_layer` 32 x 2 x
    `n_ctx` 32768 x `floats_per_col` 1024 x 4 B), which OOMed at LOAD on a 12 GB
    box and produced NO output at all (empty cell, before any plan line) when a
    sub-`n_ctx` window was requested; grow-on-demand is the fix, and a genuine
    allocation failure now logs an ASCII error and fails the decode cleanly
    rather than silently yielding nothing. A later increment made the window
    choice AUTOMATIC and lossless (FEAT-002, conservative RAM budget): when
    full-`n_ctx` KV would overflow the conservative memory budget (Section 5),
    the planner steps `W` down to the largest power-of-two (`>= 256`) that fits.
    FEAT-002 Part A then plumbs that chosen `W` to the engine as a DIRECT
    `Engine::load` parameter (`PlacementPlan::kv_window`), applied BY DEFAULT -
    not through the process-global `STRATAFLOW_ENGINE_KV_WINDOW` env it used to
    rely on. The resolution order inside the engine is: an explicit user/test
    `STRATAFLOW_ENGINE_KV_WINDOW` env WINS, else the plan-provided window, else
    full `n_ctx`. This closed two real-path breaks that let the window silently
    not engage on the real Mixtral `.strata`: (1) the real synthesized-metadata
    path read `context_length`/`head_count`/`head_count_kv` with
    `gguf_get_val_u32`, which aborts/returns 0 when the source stored them under
    a different GGUF integer encoding (the `n_ctx_train(0)` tell), so the planner
    saw zero geometry and never windowed - fixed by a robust type-dispatching
    read (`src/common/gguf_meta.h`) shared by planner, engine, and packer so all
    three see IDENTICAL geometry; and (2) the window reached the engine ONLY via
    the env side-channel - now it is a load parameter. So a real large-context
    Mixtral fits by DEFAULT with no env set. The KV saving LOWERS the projected
    peak and is NOT re-consumed by
    more resident experts - the budget model holds the safety margin fixed (see
    Section 5, "The budget is enforced by construction"). This is what lets the
    SAME default and `STRATAFLOW_ENGINE_KV_WINDOW=512` runs that previously
    OOM-killed the 12 GiB box (exit status 137) now decode within budget.
  - **KV quantization [DEFERRED, approximate]**: store K/V at q8 or 4-bit
    (optionally Hadamard-rotated) instead of F32. NOT shipped: it is approximate
    (changes attention numerics) and belongs behind the quality-metric gate
    (Stage 4). Out of scope for the lossless increment.
- **50B math.** F32 KV at useful context is multi-GB. q8 KV is ~4x smaller;
  4-bit ~8x. Paging bounds resident KV to the recent window regardless of total
  context. Together these turn a growing multi-GB fixed cost into a bounded
  sub-GB resident cost, freeing RAM for experts (Ceiling A) and enabling long
  context on the box.
- **Oracle-safety.**
  - KV *paging* is LOSSLESS (same F32 bytes, just moved RAM tiers): the cold
    columns and the resident ring columns are re-assembled in absolute position
    order and the IDENTICAL attention op sequence runs over them, so decode is
    byte-identical to the full-resident path and to the oracle. PROVEN in-sandbox
    (FEAT-002, `tests/test_kv_paging.cpp`): with `STRATAFLOW_ENGINE_KV_WINDOW=4`
    over an 8-token generation (sequence longer than the window, so cold KV is
    paged out and fetched back), the paged greedy token sequence and per-step
    logits are BIT-IDENTICAL to the full-resident engine (`max|delta| == 0.0`)
    and match the libllama oracle (F32 within 1e-3; Q4_K seq-identical). Resident
    KV dropped from 32768 B to 2048 B (F32 MoE, `n_ctx` 64 -> `W` 4) and from
    1048576 B to 16384 B (Q4_K MoE, `n_ctx` 256 -> `W` 4) with byte-identical
    output. The real-model regime (large `n_ctx`, window WELL below it) that the
    tiny W=4 fixtures missed is now also covered by a dedicated gate
    (`test_kv_paging_bigctx_small_window` over the `n_ctx` 4096 fixture
    `make_tiny_moe_bigctx_gguf.py`, `STRATAFLOW_TEST_BIGCTX_MOE_GGUF`) at `W=64`
    (`W/n_ctx` 1/64), in BOTH a short no-overflow and a long overflow regime:
    SHORT (gen=8) `residentKV 32768 B`, `fullKV 2097152 B`, `coldBytes 0 B`,
    `cold_fetches 0`; LONG (gen=80) `residentKV 32768 B`, `fullKV 2097152 B`,
    `coldBytes 12800 B`, `cold_fetches 650`. Both are byte-identical to the
    full-resident engine (`max|delta| = 0.000e+00`) and within 1e-3 of the
    oracle; `residentKV == fullKV/64` confirms the `W/n_ctx` reservation and
    `coldBytes 12800` (proportional to the 25 evicted positions, far below the
    full-`n_ctx` cold size 2097152 B) confirms the cold store is bounded to used
    positions, not `n_ctx`. The DEFAULT-ENGAGEMENT real path (FEAT-002 Part A,
    `tests/test_kv_window_real_path.cpp`) is proven SANDBOX-side on a Mixtral-
    geometry legacy-per-expert fixture (`n_ctx` 32768, `head_dim` 128, GQA
    `n_head_kv` 2, 32 layers, full KV 2048 MiB) packed through the REAL
    `strata-pack` -> `.strata` -> `load_strata_model` -> planner -> engine path
    under a simulated 12 GiB budget with NO env override: the loaded `ModelShape`
    reports `n_ctx`/`head_dim`/`n_head_kv` non-zero (geometry reached the
    planner), the plan chose `kv_window = 16384` (`< n_ctx`), and the engine's
    `resident_kv_bytes()` dropped to 1073741824 (half of the full 2147483648)
    with the window applied through the load parameter alone - and the plan-
    window greedy decode is BIT-IDENTICAL to the full-resident run
    (`max|delta| = 0.000e+00`) and matches the libllama oracle on the matched
    stacked fixture. This is the sandbox proof of the exact DEFAULT path the
    Colab Mixtral run showed broken (resident_kv at full `n_ctx`, no `kv window=`
    in the plan line). The absolute real-Mixtral peak-RSS number is COLAB-ONLY
    (the offline sandbox cannot download the multi-GB model); the sandbox proof
    is the geometry-mirroring fixture through the real load path.
  - KV *quantization* is APPROXIMATE (q8/4-bit K/V changes attention numerics)
    and needs the quality-metric gate; DEFERRED as an opt-in follow-on with
    measured perplexity cost. StreamingLLM/SnapKV-style windowed EVICTION is also
    lossy (it DROPS past positions) and is explicitly out of scope here: the
    shipped paging keeps every past position.
- **Complexity / deps.** Paging shipped with no new-kernel dependency (the cold
  store is plain RAM vectors and the assembly reuses `ggml_cpy` + the existing
  attention ops, all on the single compute thread). SSD spill of cold pages
  (option b) and quantized-KV attention kernels remain the deferred follow-ons.
- **Verdict: KV paging SHIPPED, LOSSLESS (FIT lever, FEAT-002); KV quantization
  DEFERRED and APPROXIMATE.** The lossless RAM-tiering half is done and proven
  byte-identical in-sandbox; SSD spill and quantization are sequenced behind it
  (quantization behind a quality metric).

### 6.6 Shrink the process floor (drop libllama from the runtime)

- **Mechanism.** `libllama` is linked only for the tokenizer / vocab (the
  `llama_*` calls in `src/model/engine/engine.cpp` and
  `src/model/ggml_model.cpp`) yet contributes to the fixed ~132 MiB process
  floor. Replace it with a tiny standalone tokenizer (BPE/SPM reading the vocab
  already embedded in the GGUF/.strata metadata) so the runtime no longer links
  libllama.
- **50B math.** Frees ~0.1 GB of fixed floor for weights. Modest in absolute
  terms, but it is pure lossless headroom and it removes the libllama dependency
  from the shipped runtime (libllama remains only as the test oracle), which is
  architecturally clean and shrinks the "fixed floor" that dominated peak RSS.
- **Oracle-safety.** LOSSLESS for decode (the tokenizer must produce identical
  token ids; validate the standalone tokenizer against `llama_tokenize` on a
  corpus as its own gate). Does not touch the forward pass.
- **Complexity / deps.** Medium: a correct BPE/SPM tokenizer matching
  llama.cpp's behavior (special tokens, byte fallback, normalization) is fiddly
  but well-trodden. Independent of the other levers (can be parallelized).
- **Verdict: feasible-with-effort, lossless, modest-impact.** Worth doing for
  architectural cleanliness and a bit of floor, but it is not what makes 50B fit;
  sequence it after the big FIT levers unless done in parallel.

### 6.7 Sequential / batched I/O to beat random-read stalls

- **Mechanism.** Instead of many separate `read_blob` calls, read several
  upcoming layers' predicted experts in one large SEQUENTIAL read, exploiting
  that SSD sequential (~4 GiB/s) far exceeds low-QD1 random. The co-activation
  placement in `PLAN.md` section 3.10 (experts used together placed adjacently
  on disk) is the enabling layout.
- **50B math.** Does not reduce bytes/token, but raises the *effective* streaming
  rate toward the sequential number. If today's effective rate is well below 4
  GiB/s (the ~15 s/token for ~2.5 GB/token implies ~0.17 GB/s effective), moving
  toward sequential could cut wall-clock per token several-fold without changing
  a single output byte.
- **Oracle-safety.** LOSSLESS (same bytes, same order of use, just fetched in
  bigger reads). Reconciles honestly with `completed-before-use = 0`: overlap
  cannot hide reads on 2 cores, but FEWER and LARGER reads still cut wall-clock
  because they run nearer peak bandwidth and pay less per-read overhead.
- **Complexity / deps.** Medium: batched/coalesced reads in `StrataReader`, a
  predicted-expert batch from the predictor, and a `.strata` layout that places
  co-activated / sequential-layer experts adjacently (packer change). The
  predictor already exists (`src/predict/predictor.h`); the layout and the
  coalesced read are the new work.
- **Verdict: feasible-now, lossless, speed lever.** This is the most promising
  LOSSLESS speed lever on 2-core hardware because it attacks the real
  bottleneck (effective bandwidth, not overlap). Strong candidate right after
  the FIT levers.
- **PROGRESS [SHIPPED, lossless - coalesced bundle reads + coherent AUTO
  retune].** The first half of this lever is now implemented and sandbox-proven.
  (1) COALESCED BUNDLE READS: `StrataReader::read_bundle` issues ONE `read_at`
  over a whole `(layer, expert)` bundle `[blob_offset, blob_offset +
  blob_length)`, then splits it into the gate/up/down slices by their known
  relative offsets, replacing the previous THREE separate `read_blob` calls (one
  per kind). It is byte-identical: the returned bytes `memcmp` equal to the three
  per-kind reads concatenated, the slices are tightly packed gate|up|down inside
  the bundle, and `streamed_bytes()` totals are unchanged (the same bytes are
  counted once). A `read_calls()` atomic counter proves the read-count drops 3:1
  per bundle (3 underlying `read_at` -> 1), proven through the REAL `StrataReader`
  (not a mock) in `tests/test_coalesced_expert_read.cpp`. The three engine
  expert-read sites (`ensure_layer_experts_resident`, `warm_expert`,
  `pf_read_bundle`) all route through `read_bundle` with a per-kind fallback, so
  the resident bytes - and therefore the decode - stay byte-identical. This is
  the effective-bandwidth proxy: fewer, larger reads run nearer peak sequential
  bandwidth and pay less per-read overhead. (2) MORE OF THE WORKING SET RESIDENT:
  the AUTO policy was retuned to a COHERENT good-neighbor share (see the AUTO
  policy note below), which caches MANY more experts than the old 2, directly
  cutting bytes/token: on the Mixtral-geometry (mxg) real fixture the streamed
  bytes/token drop from 29,408,625 B (2-slot baseline) to 3,008,102 B (104-slot
  AUTO), ~9.8x fewer expert bytes re-streamed per token. Within whatever RAM the
  mode affords, residency is biased toward the experts most REUSED across tokens:
  `Engine::forward` refines the layer-stratified `SlotPool` every few tokens via
  `set_layer_weights(ExpertPredictor::layer_activity())`, so a layer's spare
  slots follow its observed per-layer routing frequency (busier layers keep more
  of their reused set resident, reducing thrash). This is residency only - the
  per-layer top-k FLOOR guarantees every routed expert is still staged before the
  matmul, so decode is byte-identical; it is proven end to end by
  `test_predictor_driven_residency` in `tests/test_slot_pool.cpp` (observed
  routing -> `layer_activity()` -> `set_layer_weights` -> the busiest layer wins
  the scarce spare slot and keeps strictly more of its reused experts resident
  across a sweep). Greedy decode stays
  BIT-IDENTICAL across AUTO/TARGET/MAX (`|delta| = 0.000e+00`) and within the
  quant bound vs the libllama oracle (`5.179e-02 < 1e-1` on the random-weight
  q4_k fixture). The REMAINING work in this lever is the adjacent-expert
  multi-bundle coalescing and the co-activation packing LAYOUT (`PLAN.md` 3.10)
  that places co-activated / sequential-layer experts adjacently on disk so even
  more reads fuse into one sequential range.
- **AUTO policy (coherent good-neighbor share).** AUTO now takes a comfortable,
  DOCUMENTED share of total RAM (the conservative `usable_ram` good-neighbor
  budget that leaves headroom for other apps) and co-selects the three fit levers
  - KV window, trunk pool, expert cache - CONSISTENTLY to fill that share: more
  resident means less streaming within the share. The KV-window step-down now
  budgets against the FULL resident trunk, so AUTO windows the KV BEFORE the
  trunk is forced to hard-stream. This enforces a COHERENCE INVARIANT: AUTO never
  keeps the full-`n_ctx` KV resident while simultaneously hard-streaming the
  trunk (if `trunk_pool_layers > 0` and `< n_layers`, then `0 < kv_window <
  n_ctx`); the levers co-reduce. When RAM affords it, AUTO keeps the trunk fully
  resident and caches far more experts than the old 2. On the synthetic 12 GiB
  Mixtral shape: BEFORE was incoherent (windowed KV AND hard trunk streaming
  pool=20/32 AND only 4 experts, peak ~= usable); AFTER is coherent (AUTO
  `kv_window=8192`, trunk fully resident pool=0, 14 experts resident). MAX stays
  `>= AUTO` on every lever and now fits its own budget (BEFORE projected peak
  14097 MiB `>` usable 12288 MiB, fits=0; AFTER peak 12249 `<=` 12288, fits=1,
  `kv_window=16384`, 18 experts). Experts resident went 2 (old) -> 14 (synthetic
  12 GiB) / 104 (mxg real fixture). Residency/placement only, so decode is
  byte-identical across all modes.
- **The honest 2-core I/O-bound ceiling.** This box stays FUNDAMENTALLY
  I/O-bound. With only 2 cores there is no compute/IO overlap to be had - the
  async prefetch worker self-backs-off at `completed-before-use = 0` (reconciled
  honestly in the oracle-safety note above). The coalesced sequential reads do
  NOT change which bytes are read or which experts are resident; they cut
  wall-clock by raising EFFECTIVE bandwidth and cutting per-read overhead, and
  the coherent AUTO share cuts bytes/token by holding more of the working set
  resident. Together these attack the two terms of the Ceiling B inequality
  (bytes/token and effective bandwidth), but they do not make the box
  compute-bound. We therefore do NOT promise a tok/s number: the current real
  figure (~0.06 tok/s, ~16 s/token on the 12 GiB / 2-core / ~4 GiB/s SSD Colab
  box) and any absolute improvement on real Mixtral is a Colab/hardware
  re-measurement, not a sandbox number. The sandbox proves the MECHANISM (plan
  geometry, ~9.8x bytes/token reduction on the fixture, 3:1 read-count cut,
  byte-identity).

### 6.7b Compress what streams (lossless on-disk expert-blob compression) [SHIPPED, lossless]

- **Mechanism.** The sibling of 6.7: 6.7 raises the effective bandwidth, this
  lever cuts the bytes that have to cross the SSD in the first place. At pack
  time each per-expert blob (the tightly packed gate|up|down bundle) is run
  through a small in-tree LZ codec; the SMALLER of {compressed, verbatim} is
  stored on disk, and a per-blob codec id plus compressed/uncompressed lengths
  are recorded in a new `.strata` region. At the read seam
  (`StrataReader::read_bundle` / `read_blob`) the compressed bytes are read into
  a per-call scratch and decompressed into the caller's staging, so the engine's
  compute sees byte-identical expert bytes while FEWER bytes actually stream from
  the SSD. It compresses ONLY the expert weight bytes for the disk transfer - not
  activations, not the KV cache, and it is NOT a lossy requant. This is the
  AirLLM "compress what you stream" idea applied to the expert path.
- **50B math.** Directly reduces `bytes_streamed_per_token` by the achieved
  compression ratio, which on an I/O-bound box is a proportional wall-clock win
  (fewer bytes / effective_bandwidth). The catch is the decompress-vs-SSD
  tradeoff: the saving is only real if the decoder runs well above SSD bandwidth,
  otherwise decompression becomes the new bottleneck and eats the gain. Measured
  decompress throughput of the shipped codec on a 32 MiB mixed buffer is ~6.9
  GiB/s (gcc release; clang ~6.1 GiB/s), comfortably above the ~4 GiB/s SSD
  per-core bandwidth, so on this box decompress is NOT the limiter and the lever
  does not backfire.
- **Honest ratio.** On the already-K-quantized expert fixture
  (`make-legacy-moe-gguf ... q4_k`) the measured ratio is ~1.0000 - i.e. no
  reduction. The experts are ALREADY q4_k quantized and the random-weight fixture
  is high-entropy, so the fast byte-level LZ cannot shrink them: every blob falls
  back to verbatim (`kNone`) and the on-disk expert region is byte-for-byte the
  same size. This is expected and we state it plainly. AirLLM reported larger
  wins because their models were higher-entropy / differently quantized; we do
  NOT claim their ~3x. On already-K-quantized experts the ratio is
  modest-to-none, and the net wall-clock effect scales with whatever ratio the
  real trained-weight Mixtral actually achieves, which is a Colab-only
  measurement (the sandbox cannot download the 19 GB model).
- **Oracle-safety.** LOSSLESS. The decode on a compressed `.strata` is
  byte-identical to the uncompressed one: per-step logit `max|delta| =
  0.000e+00`, and both match the libllama oracle within the quant tolerance
  (`~5.2e-2 < 1e-1`). It is a pure transport optimization - the same bytes are
  reconstructed, just fetched smaller. The mechanism is proven end to end even
  where the fixture does not compress: on a deliberately compressible blob the
  reader stores a 16384-byte blob as 224 bytes on disk, `read_bundle` returns the
  full 16384 uncompressed bytes byte-exact, and `streamed_bytes()` counts 224
  (the compressed size), not 16384 - so the smaller on-disk region, the
  fewer-bytes-read accounting, and the transparent decompress are all correct and
  lossless; only the RATIO on already-quantized experts is modest.
- **Complexity / deps.** Low: a single self-contained LZ codec
  (`src/common/lz_codec.{h,cpp}`, no new submodule, no third-party dependency),
  the pack-time compression and the read-seam decompress. The codec functions are
  pure (per-call scratch only), so they are race-free for the concurrent
  compute-thread and prefetch-worker reads. The `.strata` format bumped to v3,
  OPTIONAL and back-compatible: v1 and v2 files still load and decode
  byte-identically, and an uncompressed v3 pack is byte-for-byte identical to the
  prior packer output (SHA256-verified). `strata-pack` gains `--compress <lz|none>`
  (default uncompressed); `streamed_bytes` now reflects the compressed bytes
  actually read.
- **Verdict: SHIPPED, lossless, modest on already-quantized experts.** The
  plumbing (smaller on-disk expert region, fewer `streamed_bytes`, transparent
  decompress, decompress faster than the SSD) is correct and proven; the win is
  bounded by the real model's entropy, so it is opt-in and the ratio is reported
  honestly rather than promised.

### 6.8 Speculative / draft decoding

- **Mechanism.** A tiny always-resident draft (EAGLE head / MTP head / n-gram /
  prompt-lookup) proposes K tokens; the big model verifies them in one batched
  forward; accepted tokens are emitted. Because verifying K tokens touches the
  expensive experts on roughly one step instead of K, the per-token expert
  streaming is amortized over the accepted run. `PLAN.md` section 3.6 and Phase 5
  already scope this and call it "the biggest single gain for low-end PCs".
- **50B math.** If the draft achieves acceptance length ~L, effective bytes/token
  for the big model drop by ~L (the expert streaming happens once per verified
  burst, not per token). On an I/O-bound box this is potentially the biggest
  *effective* tok/s multiplier, and it composes with every byte-reducing lever.
- **Oracle-safety.** The DRAFT is approximate, but speculative decoding with
  exact verification is designed so the FINAL emitted sequence is identical to
  plain greedy decoding (reject-and-correct). So it is LOSSLESS at the output
  level under greedy, consistent with `PLAN.md` section 7's determinism contract
  ("speculative paths must produce output identical to the plain path"). The
  draft model/head is extra resident state but does not change results.
- **Complexity / deps.** High. Needs a draft source (an n-gram / prompt-lookup
  draft needs no model and is the cheapest entry; EAGLE/MTP need a trained head
  shipped with the model), tree/batch verification in the engine, and adaptive K.
  Verification does a multi-token forward, which interacts with the KV cache and
  the streaming path.
- **Verdict: feasible-with-effort, output-lossless, high effective-speed
  impact.** The n-gram / prompt-lookup variant is the lowest-dependency entry and
  is output-identical under greedy. It is more engine work than the FIT levers,
  so sequence it after fit is solved, but it is the headline speed lever and
  should not be skipped.

### 6.9 Additional sound out-of-box ideas

- **Lower precision only for cold / long-tail experts (asymmetric precision).**
  Keep hot experts at their shipped quant (exact), compute rarely-used cold
  experts at a lower precision streamed from a low-bit side-store. Approximate
  only on the long tail, so the perplexity cost is small and measurable. Ties to
  the mixed-precision `.strata` presets in `PLAN.md` section 3.7. Verdict:
  research-risky (approximate), interesting as a follow-on to 6.3.
- **Cache dequantized hot experts.** If dequant is on the critical path (6.3),
  cache the dequantized form of the few hottest experts so their dequant cost is
  paid once, not per token. Lossless (it is just a compute cache). Verdict:
  feasible-with-effort, a composable optimization on top of 6.3.
- **Cross-token expert-reuse exploitation.** Adjacent tokens often route to
  overlapping experts; the layer-stratified pool already exploits this
  implicitly. A stronger version pins the measured top-N experts per layer as a
  near-permanent resident core and only streams the tail. Lossless (residency
  policy only). Verdict: feasible-now, a refinement of PR #39's policy; low risk,
  modest win.
- **Disaggregate prefill from decode.** Prefill is compute-bound and touches many
  experts; decode is I/O-bound and touches few. Stream experts without caching
  during prefill (the `PLAN.md` section 3.3 "prefill bypass" rule) so prefill
  does not evict the decode hot set, and size residency for the decode pattern.
  Lossless. Verdict: feasible-now, a residency/scheduling refinement; helps TTFT
  and protects the decode cache.

---

## 7. Throughput model

Decode wall-clock per token is bounded below by the max of the I/O term and the
compute term, with no overlap credit on 2 cores:

```
t_token  ~  bytes_streamed_per_token / effective_ssd_bandwidth
            + compute_flops_per_token / (cores * per_core_flops)
```

Honest statement for the target box: with 2 cores and ~4 GiB/s SSD, this is
hardware-bound. We do not promise a tok/s number; we show how each lever moves
the terms.

| Lever | bytes/token | effective_bandwidth | compute/token | net on t_token |
|---|---|---|---|---|
| Stream the trunk (6.4) | slightly up (trunk now streams) | - | - | enables FIT; small speed cost, bought back by more expert RAM |
| Page/quantize KV (6.5) | - | - | - (quant: small up) | frees RAM -> more experts resident -> fewer misses -> bytes/token DOWN |
| Shrink floor (6.6) | - | - | - | frees RAM -> more experts resident -> bytes/token DOWN (small) |
| Sub-byte resident (6.3) | DOWN (more experts fit) | - | UP (dequant) | net win only if dequant cheaper than the saved I/O; measure on 2 cores |
| Sub-expert streaming (6.2) | DOWN (fewer bytes/expert) | - | ~flat | biggest bytes/token cut; approximate |
| Batched sequential I/O (6.7) | flat | UP toward 4 GiB/s | - | biggest LOSSLESS wall-clock cut on this box |
| Compress what streams (6.7b) | DOWN by the ratio | - | UP (decompress ~6.9 GiB/s > SSD, so free here) | LOSSLESS; net win scales with the real ratio (~1.0 on already-K-quant fixture) |
| Speculative decode (6.8) | DOWN by ~acceptance length | - | UP (verify batch) | biggest EFFECTIVE tok/s multiplier; composes with all above |

Direct / unbuffered I/O (already designed in `PLAN.md` 3.3) is assumed
throughout so the OS page cache does not double-count against the RAM budget;
suppressing page-cache growth is also what keeps peak RSS near the controllable
resident floor rather than ballooning as in the Mixtral run.

---

## 8. Sequenced implementation plan

Ordered by (impact x tractability x oracle-safety), separating FIT levers (make
50B load in a small resident floor) from SPEED levers (cut bytes/token or
wall-clock). "Lossless" items build under the existing byte-identity gate;
"approximate" items need a new quality-metric (perplexity/KLD vs oracle) harness
first and ship opt-in.

### Stage 1 - FIT, lossless (makes 50B load at all; highest priority)

1. **Stream the trunk (6.4)** - LOSSLESS, mandatory for 50B to fit. Converts the
   ~3-5 GB fixed trunk into a bounded hot set. Reuses `StrataReader`/`SlotPool`/
   staging. **Recommended FIRST implementation target.**
2. **Page / depth-bound the KV cache (6.5 paging)** - LOSSLESS, SHIPPED
   (FEAT-002). Bounds the growing multi-GB KV to a resident recent window `W`
   (`STRATAFLOW_ENGINE_KV_WINDOW`), tiers cold columns to a RAM store, and
   re-assembles them losslessly for attention; frees RAM for experts. SSD spill
   of cold pages is the deferred follow-on.
3. **Shrink the process floor / standalone tokenizer (6.6)** - LOSSLESS, modest,
   independent (parallelizable with 1-2).

### Stage 2 - SPEED, lossless (cut wall-clock without touching outputs)

4. **Batched sequential I/O + co-activation packing (6.7)** - LOSSLESS. Attacks
   the real 2-core bottleneck (effective bandwidth). Needs a packer layout
   change plus coalesced reads.
4b. **Compress what streams (6.7b)** - LOSSLESS, SHIPPED. Opt-in on-disk
   expert-blob compression (`strata-pack --compress lz`), decompressed at the
   read seam; cuts bytes/token by the achieved ratio (modest-to-none on
   already-K-quantized experts, scales with real-model entropy). Decompress
   ~6.9 GiB/s stays above SSD bandwidth, so it does not backfire.
5. **Cross-token reuse pinning + prefill/decode disaggregation (6.9)** - LOSSLESS
   residency/scheduling refinements on top of PR #39; low risk.

### Stage 3 - SPEED, higher effort, output-lossless under greedy

6. **Speculative decode, n-gram/prompt-lookup first (6.8)** - output-identical
   under greedy; the headline effective-tok/s lever. Start with the
   no-extra-model n-gram draft, then MTP/EAGLE heads when a Coaade model ships
   them.

### Stage 4 - approximate, opt-in, needs a quality gate (research-risky)

7. **KV quantization (6.5 quant)** - approximate; q8/4-bit KV with measured
   perplexity cost.
8. **Sub-byte resident experts + dequant (6.3)** - approximate (unless the model
   ships at that quant); needs Phase 6 LUT kernels; CPU-gated on 2 cores.
9. **Sub-expert / activation-sparse streaming (6.2)** - approximate, highest
   bytes/token upside, highest risk; needs sparse kernels, sub-blob `.strata`
   addressing, and the quality gate.
10. **Asymmetric cold-expert precision + dequant caching (6.9)** - approximate
    follow-ons to 6.3.

Prerequisite for Stage 4: a perplexity / KLD regression harness vs the oracle
(the `PLAN.md` section 7 quality plan), since byte-identity no longer applies.

---

## 9. Risks and unknowns (what needs a spike)

- **Trunk streaming lifetime discipline.** Trunk tensors are referenced
  throughout graph build (`get_weight`); making them stream-backed must preserve
  the "resident before the op is built" contract the expert path relies on.
  Spike: prototype trunk streaming on Mixtral and confirm byte-identity + bounded
  resident floor.
- **Does trunk streaming hurt speed more than it helps?** Trunk is touched every
  token, so streaming it adds per-token bytes. The thesis is that freeing RAM for
  experts nets positive. Spike: measure bytes/token and tok/s with trunk
  streamed vs resident at a 50B-shaped budget on Colab.
- **Effective sequential bandwidth on the real box.** The ~0.17 GB/s implied
  effective rate vs ~4 GiB/s sequential is the single biggest unknown for the
  speed story. Spike: micro-benchmark `read_blob` coalesced vs per-expert on the
  Colab SSD and confirm the gain is real at QD1.
- **Activation-sparsity quality cost for SwiGLU experts.** Unknown perplexity
  cost; may be too high to be useful. Spike: offline sparsity-vs-perplexity sweep
  before committing kernel work.
- **Dequant cost on 2 cores.** Whether a 2-bit LUT dequant is cheap enough that
  the I/O savings survive. Spike: micro-benchmark a LUT kernel on the 2-core box.
- **Speculative acceptance length on Coaade models.** The effective-tok/s win is
  proportional to acceptance length, which is model/prompt dependent. Spike:
  measure n-gram/prompt-lookup acceptance on representative prompts.
- **50B trunk/KV absolute sizes.** The budget model is symbolic; the real 50B
  shape must be measured to confirm the resident floor lands under ~6 GB after
  the FIT levers.

---

## 10. Summary of verdicts and the recommended first target

| # | Technique | Oracle-safety | Verdict |
|---|---|---|---|
| 6.4 | Stream the trunk | lossless | feasible-now (FIT, top) |
| 6.5 | KV paging (lossless RAM tiering) | lossless | SHIPPED (FIT, FEAT-002); SSD spill deferred |
| 6.6 | Standalone tokenizer / drop libllama | lossless | feasible-with-effort (modest) |
| 6.7 | Batched sequential I/O + co-activation packing | lossless | feasible-now (SPEED, top lossless) |
| 6.7b | Compress what streams (on-disk expert-blob LZ) | lossless | SHIPPED (SPEED, opt-in; ~1.0 on already-K-quant, scales with real entropy) |
| 6.9 | Cross-token pinning / prefill-decode split | lossless | feasible-now (refinement) |
| 6.8 | Speculative decode (n-gram first) | output-lossless (greedy) | feasible-with-effort (headline SPEED) |
| 6.5 | KV quantization | approximate | feasible-with-effort (opt-in) |
| 6.3 | Sub-byte resident + dequant | approximate (unless shipped quant) | feasible-with-effort (CPU-gated) |
| 6.2 | Sub-expert / activation-sparse streaming | approximate | research-risky (highest upside) |
| 6.1 | SSD-as-tier (model-aware paging) | lossless | feasible-now (organizing principle; realized by 6.4/6.5) |

**Lossless, feasible-now set, in order:** (1) stream the trunk, (2) batched
sequential I/O + co-activation packing, (3) cross-token pinning / prefill-decode
split, with KV paging and the standalone tokenizer as the next lossless items.

**Recommended FIRST implementation target: stream the trunk (6.4).** It is
lossless (byte-identical under the existing gate), it reuses the existing
`StrataReader`/`SlotPool`/staging machinery, and it is the one lever without
which a 50B-class model cannot fit in the 12 GB resident floor at all. Everything
else either depends on the model fitting first or is a speed multiplier on top of
a model that fits.

Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0

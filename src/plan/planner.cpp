// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "plan/planner.h"

#include <algorithm>
#include <sstream>

namespace sf {
namespace {

constexpr uint64_t kMiB = 1024ull * 1024;
constexpr uint64_t kGiB = 1024ull * 1024 * 1024;

// CONSERVATIVE usable-RAM ceiling (FEAT-002). The old auto_ram_budget trusted
// MemAvailable (ram_free_bytes) and merely shaved a 2 GiB/12.5% headroom, then
// handed ALL of the rest to the expert cache. On a shared Colab box
// MemAvailable is optimistic (it counts reclaimable page cache that the kernel
// may not actually free in time), so a plan sized to it over-commits.
//
// usable_RAM is the SMALLER of:
//   (a) ram_free_bytes minus a headroom (so we never claim more than the OS
//       currently reports free), and
//   (b) a conservative FRACTION of ram_total_bytes (so an optimistic
//       MemAvailable on a shared box cannot inflate the budget beyond a share
//       of physical RAM).
// Both guards are documented and model-independent. The fraction (85% of total)
// and the free-RAM headroom (max(1 GiB, 12.5%)) are deliberately conservative;
// the per-component subtraction and the safety margin in plan_placement do the
// rest of the protection.
// Fallback usable-RAM budget for an UNKNOWN hardware profile (both RAM fields
// 0). Some code paths (the dry-run/ABI smoke path and any profiler that could
// not read the OS RAM figures) hand the planner a zero-filled HardwareProfile.
// A literal 0-byte budget would make the fit invariant refuse EVERY model, so
// we treat "RAM unknown" as a generous, realistic default rather than "no RAM".
// This only affects the unknown case; a REAL profile with nonzero RAM still
// gets the conservative ceiling below and the fit refusal is unchanged.
constexpr uint64_t kUnknownRamFallbackBytes = 16ull * 1024 * 1024 * 1024;

uint64_t usable_ram(const HardwareProfile &hw) {
    const uint64_t total = hw.ram_total_bytes;

    // RAM UNKNOWN: neither total nor free reported. Fall back to a sane,
    // non-zero default so the planner does not refuse a model purely because
    // the profile lacked RAM numbers (e.g. the dry-run ABI path).
    if (total == 0 && hw.ram_free_bytes == 0) {
        return kUnknownRamFallbackBytes;
    }

    // Each field being 0 means "unknown"; prefer the known one. If free is
    // unknown fall back to total; if total is unknown the free-based ceiling
    // below simply has no 85%-of-total guard to apply.
    const uint64_t avail = hw.ram_free_bytes ? hw.ram_free_bytes : total;

    uint64_t from_free = avail;
    const uint64_t free_headroom = std::max<uint64_t>(1 * kGiB, avail / 8);
    from_free = avail > free_headroom ? avail - free_headroom : avail / 2;

    uint64_t usable = from_free;
    if (total > 0) {
        const uint64_t from_total = total / 100 * 85;  // 85% of physical RAM
        if (from_total < usable) usable = from_total;
    }
    return usable;
}

// ALL-available RAM for Max mode: dedicate the WHOLE MACHINE to the model.
// Unlike usable_ram() this applies NO conservative shave (no free-headroom, no
// 85% cap) - it hands the planner the full PHYSICAL RAM so it holds the full KV
// window, the maximal expert cache, and the trunk fully resident when RAM
// affords it.
//
// BUG 2 FIX (task-fix-target-and-max-mode-semantics): this used to prefer the
// INSTANTANEOUS ram_free_bytes. That inverted Max on a box where the model (or
// anything else) had already consumed RAM: MemAvailable (ram_free_bytes) was
// small, so Max resolved a TINY ceiling and ended up holding FEWER resident
// experts and streaming MORE than Auto - the opposite of "dedicate the
// machine". Auto (usable_ram) is a good-neighbor SHARE sized from TOTAL RAM, so
// a free-based Max could resolve BELOW Auto. Max must size to ALL PHYSICAL RAM
// so that "Max >= Auto" holds on the SAME machine: Max dedicating the box can
// never be smaller than Auto's conservative share of that same box.
//
// Prefer ram_total_bytes (dedicate the machine). Fall back to ram_free_bytes
// only when total is unknown (0), then to the non-zero unknown-profile fallback
// usable_ram() uses (so the zero-RAM/dry-run path never collapses to 0).
uint64_t all_available_ram(const HardwareProfile &hw) {
    if (hw.ram_total_bytes) return hw.ram_total_bytes;  // all physical RAM
    if (hw.ram_free_bytes) return hw.ram_free_bytes;    // total unknown
    return kUnknownRamFallbackBytes;                    // profile unknown
}

// Resolve the usable-RAM ceiling by the documented precedence (see
// plan_placement in the header): an explicit target wins, then the explicit
// ram_budget, then Max mode uses all-available RAM, else Auto uses the
// conservative good-neighbor share. The result is always non-zero (the
// unknown-profile fallback is carried through every branch).
uint64_t resolve_usable_ram(const HardwareProfile &hw, uint64_t ram_budget,
                            RamMode mode, uint64_t target_ram_budget) {
    if (target_ram_budget) return target_ram_budget;  // explicit exact target
    if (ram_budget) return ram_budget;                // explicit --ram-gb
    if (mode == RamMode::Max) return all_available_ram(hw);  // dedicate machine
    return usable_ram(hw);                            // Auto good-neighbor share
}

// Conservative resident KV estimate matching Engine::resident_kv_bytes():
//   n_layer * 2 (K+V) * (head_dim * n_head_kv) * sizeof(float) * W
// where W is the chosen window (columns). When the shape fields are unknown
// (0), fall back to a value that never UNDER-counts: use the trunk size as a
// coarse proxy floor so the planner stays safe on a model whose KV geometry we
// could not read.
uint64_t kv_bytes_for_window(const ModelShape &model, uint64_t window) {
    if (model.n_layers == 0 || model.head_dim == 0 || model.n_head_kv == 0 ||
        window == 0) {
        // Unknown geometry: a conservative FIXED reserve. We cannot size the
        // real KV without head_dim/n_head_kv/n_ctx, so charge a non-trivial
        // fixed amount (1 GiB) that keeps the expert cache from being sized
        // into KV's space without catastrophically over-charging a model whose
        // geometry simply was not plumbed. Real models populate the geometry
        // (ggml_model.cpp), so this fallback only affects synthetic/dry-run
        // shapes and tiny fixtures.
        return 1 * kGiB;
    }
    const uint64_t floats_per_col =
        static_cast<uint64_t>(model.head_dim) * model.n_head_kv;
    return static_cast<uint64_t>(model.n_layers) * 2 * floats_per_col *
           sizeof(float) * window;
}

// Full-n_ctx resident KV (what the engine reserves by default). Falls back to
// the conservative proxy when geometry is unknown.
uint64_t full_kv_bytes(const ModelShape &model) {
    return kv_bytes_for_window(model, model.n_ctx);
}

// Conservative ggml compute-buffer estimate. The engine grows this on demand
// via ggml_gallocr_alloc_graph (no fixed reserve), so we cannot read it; we err
// HIGH. The compute graph scales with the embedding width and the resident KV
// window (the attention tensors), so a scale-free floor plus a small per-column
// term keeps the estimate honest for both tiny fixtures and real Mixtral.
uint64_t compute_estimate_bytes(const ModelShape &model, uint64_t kv_window) {
    // A fixed floor that errs high for a single-token greedy decode.
    //
    // BUG 3 CALIBRATION (task-fix-target-and-max-mode-semantics): the engine
    // grows its ggml compute scratch ON DEMAND (ggml_gallocr, no fixed reserve)
    // and a single-token greedy decode's scratch is a few hundred MiB, not the
    // ~0.5 GiB this floor used to charge on TOP of the full staging + margin +
    // KV stack. Stacking a 512 MiB compute floor with the 512 MiB margin floor
    // roughly DOUBLED the honest resident footprint (AUTO est ~10 GiB vs 5005
    // MiB real; MAX est ~11 GiB vs 2212 MiB real), and that over-estimate is
    // precisely WHAT made an achievable TARGET refuse. Trim the floor to a
    // still-conservative 256 MiB (err high, never under-count) so the estimate
    // tracks reality (slightly high, not ~2x) while the fit invariant stays
    // safe (the per-window term below still grows with a large resident KV).
    uint64_t est = 256 * kMiB;
    // Grow with the attended window so a very large resident KV also carries a
    // larger scratch estimate (bounded, conservative).
    if (model.head_dim && model.n_head_kv && kv_window) {
        const uint64_t floats_per_col =
            static_cast<uint64_t>(model.head_dim) * model.n_head_kv;
        est += floats_per_col * sizeof(float) * kv_window / 4;
    }
    return est;
}

// Process floor: README cites ~132 MiB peak RSS for small models. Charge at
// least that for the interpreter/runtime/allocator overhead the engine runs on
// top of its tracked buffers.
constexpr uint64_t kProcessFloorBytes = 160 * kMiB;

// Safety margin: a fraction of usable RAM AND a fixed minimum, whichever is
// larger. This is the headroom the fit invariant preserves; a KV window that
// frees RAM must not be re-consumed into this margin.
//
// BUG 3 CALIBRATION (task-fix-target-and-max-mode-semantics): the old
// max(512 MiB, 8%) margin stacked with the (old 512 MiB) compute floor on top
// of a full staging + KV charge, roughly doubling the honest peak and making an
// achievable TARGET refuse. Trim to max(256 MiB, 6%): still real headroom
// (never under-counts, the fit invariant holds) but it no longer inflates the
// estimate to ~2x the engine's real resident footprint.
uint64_t safety_margin_bytes(uint64_t usable) {
    return std::max<uint64_t>(256 * kMiB, usable / 100 * 6);  // >=256 MiB or 6%
}

} // namespace

PlacementPlan plan_placement(const HardwareProfile &hw,
                             const ModelShape &model,
                             uint64_t vram_budget,
                             uint64_t ram_budget,
                             uint64_t cache_budget,
                             RamMode mode,
                             uint64_t target_ram_budget) {
    PlacementPlan plan;

    uint64_t vram = vram_budget;
    if (vram == 0 && !hw.gpus.empty()) {
        vram = hw.gpus.front().vram_bytes;
    }
    // Usable RAM ceiling resolved by precedence: explicit target > explicit
    // ram_budget > Max (all available RAM) > Auto (conservative good-neighbor
    // share). An explicit numeric budget (target or ram_budget) is used as the
    // EXACT ceiling. The result is always non-zero (unknown-profile fallback).
    const uint64_t usable =
        resolve_usable_ram(hw, ram_budget, mode, target_ram_budget);
    plan.usable_ram_bytes = usable;

    const uint64_t per_layer_trunk =
        model.n_layers ? model.trunk_bytes / model.n_layers : 0;

    // Rule 1: always-active trunk goes on the fastest device first (VRAM, then
    // RAM). This is the KTransformers / llama.cpp-offload insight.
    uint64_t vram_left = vram;
    uint64_t ram_left = usable;
    if (per_layer_trunk > 0) {
        while (plan.trunk_layers_in_vram < model.n_layers &&
               vram_left >= per_layer_trunk) {
            vram_left -= per_layer_trunk;
            ++plan.trunk_layers_in_vram;
        }
        while (plan.trunk_layers_in_vram + plan.trunk_layers_in_ram < model.n_layers &&
               ram_left >= per_layer_trunk) {
            ram_left -= per_layer_trunk;
            ++plan.trunk_layers_in_ram;
        }
    }

    // Only the trunk layers that landed in RAM count toward the RAM peak; VRAM
    // layers live on the GPU (the no-GPU OOM case this feature fixes has all
    // trunk layers in RAM).
    const uint64_t trunk_in_ram =
        uint64_t(plan.trunk_layers_in_ram) * per_layer_trunk;

    if (!(model.is_moe && model.expert_bytes > 0)) {
        // Dense model: no experts to place. The honest peak is the trunk plus
        // KV + compute + floor + margin (dense models still pay those).
        const uint64_t kv = full_kv_bytes(model);
        const uint64_t compute = compute_estimate_bytes(model, model.n_ctx);
        const uint64_t margin = safety_margin_bytes(usable);
        plan.expert_slots_resident = 0;
        plan.stream_experts = false;
        plan.kv_window = 0;
        plan.kv_resident_bytes = kv;
        plan.staging_bytes = 0;
        plan.compute_bytes = compute;
        plan.process_floor_bytes = kProcessFloorBytes;
        plan.safety_margin_bytes = margin;
        plan.planned_peak_bytes =
            trunk_in_ram + kv + compute + kProcessFloorBytes + margin;
        plan.fits = plan.planned_peak_bytes <= usable;
        if (!plan.fits) {
            std::ostringstream os;
            os << "cannot fit dense model in budget: projected peak "
               << (plan.planned_peak_bytes / kMiB) << " MiB exceeds usable RAM "
               << (usable / kMiB) << " MiB (trunk-in-RAM "
               << (trunk_in_ram / kMiB) << " MiB + KV " << (kv / kMiB)
               << " MiB + compute " << (compute / kMiB) << " MiB + floor "
               << (kProcessFloorBytes / kMiB) << " MiB + margin "
               << (margin / kMiB) << " MiB)";
            plan.refusal_reason = os.str();
        }
        return plan;
    }

    // Rule 2: routed experts fill the remaining VRAM, then RAM, as a cache -
    // but ONLY the RAM left after ALL other mandatory resident costs are
    // subtracted (FEAT-002 conservative budget). The old code sized the expert
    // cache from vram_left + ram (all remaining RAM after only the trunk),
    // ignoring KV + staging + compute + floor + margin, so the real peak blew
    // the box. We now reserve those FIRST, then size experts into the remainder.
    const uint64_t full_slots =
        uint64_t(model.n_layers) * uint64_t(model.n_experts);
    const uint64_t per_bundle = model.expert_bytes;

    // Mandatory resident costs that are INDEPENDENT of the expert slot count:
    //   staging  = ONE layer's experts at FULL n_expert (the engine's
    //              stage_gate/up/down are sized to a full layer, NOT top-k).
    //   compute  = conservative ggml scratch (err high).
    //   floor    = process RSS floor.
    //   margin   = preserved safety headroom.
    // KV depends on the chosen window and is decided below.
    const uint64_t staging = uint64_t(model.n_experts) * per_bundle;
    const uint64_t process_floor = kProcessFloorBytes;
    const uint64_t margin = safety_margin_bytes(usable);

    // VRAM expert cache sizing is reported as-is (GPU path); the fit invariant
    // below is about the RAM peak (the no-GPU OOM case this feature fixes).
    plan.expert_slots_vram =
        per_bundle > 0 ? static_cast<uint32_t>(vram_left / per_bundle) : 0;

    // AUTO KV-WINDOW decision (lossless). Try the full n_ctx first; if the
    // fixed (expert-independent) costs plus the n_expert_used expert floor
    // already overflow the budget, step the KV window down to a conservative
    // minimum so a real large-ctx model (Mixtral n_ctx=32768) can fit by
    // DEFAULT. The KV saving lowers the peak; it is NOT re-consumed by more
    // experts because the margin is held fixed and experts are sized from
    // whatever remains after KV.
    //
    // Candidate windows: full n_ctx, then a descending set of powers-of-two
    // bounded by a sane minimum (kMinKvWindow). We pick the LARGEST window
    // whose fixed+floor budget leaves room for at least n_experts_used experts;
    // if even the smallest window cannot, we refuse.
    const uint32_t kMinKvWindow = 256;
    const uint64_t lower_experts =
        model.n_experts_used ? model.n_experts_used : 1;

    // Trunk footprint the KV step-down budgets against (FEAT-002 COHERENCE).
    //
    // COHERENCE INVARIANT: AUTO (and any mode) must NEVER keep the FULL n_ctx
    // KV resident while simultaneously HARD-STREAMING the per-layer attention
    // trunk (trunk_pool_layers>0 and << n_layers). The two resident levers must
    // CO-REDUCE: on a tight box the KV window steps down FIRST (it is lossless
    // paging) so the full trunk can stay resident; the trunk only streams if
    // even a windowed KV + the full resident trunk cannot fit.
    //
    // The old code budgeted the KV step-down against the MINIMAL (streamed)
    // trunk floor (non-attn + one hot layer). That let AUTO keep FULL KV
    // because full KV "fit" once the trunk was assumed streamed - and then the
    // trunk-streaming pass below streamed the trunk hard, yielding the forbidden
    // full-KV + hard-trunk-streaming combination (observed on real Mixtral:
    // 8 GiB resident KV with hot=9/32 trunk). Budgeting the KV window against
    // the FULL resident trunk instead forces the KV to step down before the
    // trunk is streamed, so the levers co-reduce and the invariant holds by
    // construction. When nothing is streamable (attn_trunk_bytes==0) this is
    // the full trunk either way, so dense/non-streamable behavior is unchanged.
    const uint64_t trunk_floor_for_kv = trunk_in_ram;

    const uint64_t fixed_wo_kv =
        trunk_floor_for_kv + staging + process_floor + margin +
        compute_estimate_bytes(model, model.n_ctx) + lower_experts * per_bundle;

    uint32_t chosen_window = model.n_ctx;  // 0 if n_ctx unknown
    uint64_t kv_resident = full_kv_bytes(model);
    bool used_auto_window = false;
    if (model.n_ctx > 0 && model.head_dim && model.n_head_kv) {
        // Does the full window already fit the minimal expert config?
        if (fixed_wo_kv + full_kv_bytes(model) > usable) {
            // Step the window down to the largest power-of-two <= n_ctx that
            // fits (and >= kMinKvWindow). Lossless: cold positions page out.
            uint32_t w = 1;
            while (w * 2u <= model.n_ctx) w *= 2u;  // largest pow2 <= n_ctx
            uint32_t best = 0;
            for (; w >= kMinKvWindow; w /= 2u) {
                const uint64_t kvw = kv_bytes_for_window(model, w);
                const uint64_t compute_w = compute_estimate_bytes(model, w);
                const uint64_t fixed_w =
                    trunk_floor_for_kv + staging + process_floor + margin +
                    compute_w + lower_experts * per_bundle;
                if (fixed_w + kvw <= usable) {
                    best = w;  // first (largest) that fits
                    break;
                }
            }
            if (best > 0 && best < model.n_ctx) {
                chosen_window = best;
                kv_resident = kv_bytes_for_window(model, best);
                used_auto_window = true;
            }
            // If none fit, keep the full window; the refusal below will fire.
        }
    }

    uint64_t compute = compute_estimate_bytes(model, chosen_window);

    // FEAT-003 ADAPTIVE PER-LAYER TRUNK STREAMING (technique 6.4). After the KV
    // window, the dominant resident cost on a tight box is the per-layer
    // attention trunk (attn_trunk_bytes, the q/k/v/output projections summed
    // over all layers, part of trunk_in_ram). If the full-resident trunk plus
    // the minimal expert floor + KV(window) + staging + compute + floor +
    // margin still overflows the budget, STREAM the per-layer attention
    // projections and keep only a BOUNDED HOT pool of `trunk_pool_layers`
    // layers resident, so the resident attention-trunk footprint drops from all
    // n_layers to the pool. The decision is BUDGET-DRIVEN: a big-RAM box keeps
    // the whole trunk resident (pool stays 0 = stream nothing); a tight box
    // streams it hard (pool as low as 1). HONEST TRADEOFF: these weights are
    // touched every token, so streaming adds per-token SSD traffic and is
    // slower on a slow box - accepted because the FIRST goal is to FIT small.
    //
    // We only stream the part of trunk_in_ram that is the per-layer attention
    // projections; the rest of the trunk (edge tensors, norms, router) stays as
    // accounted in trunk_in_ram. per_layer_attn is the hot-pool slot cost.
    const uint64_t attn_trunk_total =
        model.n_layers ? std::min(model.attn_trunk_bytes, trunk_in_ram) : 0;
    const uint64_t per_layer_attn =
        model.n_layers ? attn_trunk_total / model.n_layers : 0;
    uint32_t trunk_pool_layers = 0;      // 0 = whole trunk resident (no stream)
    uint64_t trunk_in_ram_eff = trunk_in_ram;  // trunk RAM after any streaming
    if (per_layer_attn > 0) {
        // Fixed resident costs that do NOT depend on the trunk pool, at the
        // chosen KV window and the minimal expert floor. If even the full
        // trunk + this minimum overflows, stream the attention trunk.
        const uint64_t non_attn_trunk =
            trunk_in_ram > attn_trunk_total ? trunk_in_ram - attn_trunk_total : 0;
        const uint64_t fixed_wo_attn = non_attn_trunk + kv_resident + staging +
                                       compute + process_floor + margin +
                                       lower_experts * per_bundle;
        if (fixed_wo_attn + attn_trunk_total > usable) {
            // Largest pool (most layers hot) whose peak still fits; floor 1 so a
            // tight box still runs (the engine's resident-before-op load is the
            // correctness backstop). The remaining room after the fixed costs
            // sets the pool depth.
            const uint64_t room =
                usable > fixed_wo_attn ? usable - fixed_wo_attn : 0;
            uint64_t fit_layers = room / per_layer_attn;  // layers that fit hot
            if (fit_layers >= model.n_layers) {
                // Everything fits hot after all: no need to stream.
                trunk_pool_layers = 0;
            } else {
                if (fit_layers < 1) fit_layers = 1;  // keep at least one hot
                trunk_pool_layers = static_cast<uint32_t>(fit_layers);
                trunk_in_ram_eff = non_attn_trunk +
                                   per_layer_attn * trunk_pool_layers;
            }
        }
    }

    // FEAT-002 COHERENCE GUARANTEE (explicit, step iii). The KV window above
    // was budgeted against the FULL resident trunk, so by construction the
    // trunk only hard-streams after the KV has already stepped down. Enforce
    // the invariant EXPLICITLY as a backstop so no future change to the lever
    // ordering can reintroduce the forbidden combination: if the trunk is
    // hard-streaming (trunk_pool_layers>0 and < n_layers) the KV window MUST be
    // windowed (0 < kv_window < n_ctx). If somehow it is still full, step it
    // down to the largest window that fits alongside the (now streamed) trunk,
    // so the two levers co-reduce. Lossless (cold KV columns page out).
    if (trunk_pool_layers > 0 && trunk_pool_layers < model.n_layers &&
        !used_auto_window && model.n_ctx > 0 && model.head_dim &&
        model.n_head_kv) {
        const uint64_t non_attn_trunk =
            trunk_in_ram > attn_trunk_total ? trunk_in_ram - attn_trunk_total : 0;
        const uint64_t trunk_hot = non_attn_trunk +
                                   per_layer_attn * trunk_pool_layers;
        uint32_t w = 1;
        while (w * 2u <= model.n_ctx) w *= 2u;  // largest pow2 <= n_ctx
        for (; w >= kMinKvWindow; w /= 2u) {
            if (w >= model.n_ctx) continue;  // must be a true window (< n_ctx)
            const uint64_t kvw = kv_bytes_for_window(model, w);
            const uint64_t compute_w = compute_estimate_bytes(model, w);
            const uint64_t fixed_w = trunk_hot + staging + process_floor +
                                     margin + compute_w +
                                     lower_experts * per_bundle;
            if (fixed_w + kvw <= usable) {
                chosen_window = w;
                kv_resident = kvw;
                used_auto_window = true;
                break;
            }
        }
        // If nothing fit, the refusal path below fires; we never return a plan
        // that keeps full KV while hard-streaming the trunk and still "fits".
        compute = compute_estimate_bytes(model, chosen_window);
    }

    // Size the expert cache from the REMAINDER after every other mandatory
    // resident cost (the conservative budget). This REPLACES the old
    // ram_for_experts = vram_left + ram. trunk_in_ram_eff reflects any
    // FEAT-003 trunk streaming (the bounded hot pool instead of the full trunk).
    const uint64_t reserved =
        trunk_in_ram_eff + kv_resident + staging + compute + process_floor + margin;
    uint64_t available_for_experts = usable > reserved ? usable - reserved : 0;
    // VRAM is a separate pool; a VRAM expert cache adds to what RAM affords.
    available_for_experts += vram_left;

    // An explicit --cache-gb cap only ever LOWERS the RAM available to experts.
    if (cache_budget > 0 && cache_budget < available_for_experts) {
        available_for_experts = cache_budget;
    }

    uint64_t affordable =
        per_bundle > 0 ? available_for_experts / per_bundle : 0;

    // Lower bound: a single layer's top-k must always fit (n_experts_used).
    uint64_t lower = lower_experts;
    // Keep a whole layer resident when it comfortably fits (model-adaptive, no
    // within-layer reload), but never raise above affordable here.
    const uint64_t one_layer_slots = model.n_experts;
    if (affordable >= one_layer_slots && one_layer_slots > lower) {
        lower = one_layer_slots;
    }

    uint64_t slots = affordable;
    if (slots > full_slots) slots = full_slots;  // never exceed working set
    if (slots < lower)      slots = lower;        // top-k must fit

    // Slot-count-INDEPENDENT fields (the Max monotonicity guard below may still
    // raise slots / KV window / trunk pool, so the slot/window/trunk/peak fields
    // - including expert_slots_ram, which the guard can change via the clamped
    // reserved cost - are set AFTER the guard).
    plan.staging_bytes = staging;
    plan.process_floor_bytes = process_floor;
    plan.safety_margin_bytes = margin;

    // BUG 2 FIX (task-fix-target-and-max-mode-semantics): MAX-vs-AUTO
    // MONOTONICITY. Max dedicates the machine, so on the SAME model+machine it
    // must NEVER end up with FEWER residents or MORE streaming than Auto. The
    // all-physical-RAM ceiling above makes max.usable >= auto.usable, which
    // makes the KV window and expert sizing monotonic in usable - EXCEPT the
    // one case the task calls out: when full KV + maximal experts cannot
    // coexist, a Max plan that spends its larger budget on a LARGER KV window
    // could be left with a SMALLER expert remainder than Auto (Auto stepped the
    // window down and bought more experts). Guard against that explicitly:
    // compute the Auto reference plan on this exact model+machine and clamp Max
    // UP to at least Auto's resident experts, Auto's KV window, and no more
    // trunk streaming than Auto. Clamping up is always safe: Auto's residency
    // fits inside auto.usable <= max.usable, so adopting it keeps max's peak
    // within max's budget. Only engaged for a PURE Max run (no explicit target
    // / ram_budget, which override the mode and are not "dedicate the machine").
    if (mode == RamMode::Max && target_ram_budget == 0 && ram_budget == 0) {
        const PlacementPlan ref =
            plan_placement(hw, model, vram_budget, /*ram_budget=*/0, cache_budget,
                           RamMode::Auto, /*target_ram_budget=*/0);
        if (ref.fits) {
            // Experts: hold at least as many resident as Auto (never fewer).
            if (ref.expert_slots_resident > slots) {
                slots = ref.expert_slots_resident;
                if (slots > full_slots) slots = full_slots;
            }
            // KV window: keep a window at least as large as Auto's. 0 == full
            // n_ctx (the largest). If Max stepped down to a SMALLER window than
            // Auto, raise it to Auto's (recomputing the resident KV charge).
            const uint32_t max_eff_win = used_auto_window ? chosen_window : model.n_ctx;
            const uint32_t ref_eff_win = ref.kv_window ? ref.kv_window : model.n_ctx;
            if (ref_eff_win > max_eff_win && model.n_ctx > 0) {
                if (ref.kv_window == 0) {
                    used_auto_window = false;  // full n_ctx
                    kv_resident = full_kv_bytes(model);
                } else {
                    used_auto_window = true;
                    chosen_window = ref.kv_window;
                    kv_resident = kv_bytes_for_window(model, ref.kv_window);
                }
            }
            // Trunk: stream no harder than Auto (a hot pool at least as deep as
            // Auto's; 0 == whole trunk resident == the least streaming).
            const uint32_t max_hot =
                trunk_pool_layers ? trunk_pool_layers : model.n_layers;
            const uint32_t ref_hot =
                ref.trunk_pool_layers ? ref.trunk_pool_layers : model.n_layers;
            if (ref_hot > max_hot) {
                if (ref.trunk_pool_layers == 0) {
                    trunk_pool_layers = 0;
                    trunk_in_ram_eff = trunk_in_ram;  // whole trunk resident
                } else {
                    trunk_pool_layers = ref.trunk_pool_layers;
                    const uint64_t non_attn_trunk =
                        trunk_in_ram > attn_trunk_total
                            ? trunk_in_ram - attn_trunk_total
                            : 0;
                    trunk_in_ram_eff =
                        non_attn_trunk + per_layer_attn * trunk_pool_layers;
                }
            }
            // Re-fit the slot count to the (possibly raised) KV window + trunk
            // so Max never overflows its own budget after the clamp: size slots
            // from the remainder at the final reserved cost, but keep at least
            // Auto's resident count (which is guaranteed to fit because Auto's
            // whole config fits auto.usable <= max.usable). This preserves BOTH
            // the fit invariant AND max.slots >= auto.slots.
            const uint64_t compute_clamped = compute_estimate_bytes(
                model, used_auto_window ? chosen_window : model.n_ctx);
            const uint64_t reserved_clamped = trunk_in_ram_eff + kv_resident +
                                              staging + compute_clamped +
                                              process_floor + margin;
            uint64_t room_clamped =
                usable > reserved_clamped ? usable - reserved_clamped : 0;
            room_clamped += vram_left;
            uint64_t affordable_clamped =
                per_bundle > 0 ? room_clamped / per_bundle : slots;
            if (affordable_clamped > full_slots) affordable_clamped = full_slots;
            if (slots > affordable_clamped) slots = affordable_clamped;
            if (slots < ref.expert_slots_resident)
                slots = ref.expert_slots_resident;
            if (slots > full_slots) slots = full_slots;
        }
    }

    plan.expert_slots_resident = static_cast<uint32_t>(slots);
    plan.stream_experts = slots < full_slots;
    plan.kv_window = used_auto_window ? chosen_window : 0;
    plan.kv_resident_bytes = kv_resident;
    const uint64_t compute_final = compute_estimate_bytes(
        model, used_auto_window ? chosen_window : model.n_ctx);
    plan.compute_bytes = compute_final;

    // Diagnostic: how many expert bundles the RAM remainder affords after the
    // non-expert reserved cost. Computed from the FINAL (post-guard) reserved
    // terms so a clamped Max plan reports a figure consistent with its final
    // resident experts / KV window, not the pre-clamp remainder.
    const uint64_t reserved_final = trunk_in_ram_eff + kv_resident + staging +
                                    compute_final + process_floor + margin;
    plan.expert_slots_ram =
        per_bundle > 0 ? static_cast<uint32_t>(
                             (usable > reserved_final ? usable - reserved_final
                                                      : 0) /
                             per_bundle)
                       : 0;
    plan.trunk_pool_layers = trunk_pool_layers;
    plan.trunk_resident_bytes = trunk_pool_layers > 0
                                    ? per_layer_attn * trunk_pool_layers
                                    : attn_trunk_total;

    // HONEST, CONSERVATIVE peak: trunk-in-RAM (hot pool if streaming) +
    // resident experts + KV(chosen window) + one-layer staging at full
    // n_expert + compute + floor + margin.
    const uint64_t resident_experts = slots * per_bundle;
    plan.planned_peak_bytes = trunk_in_ram_eff + resident_experts + kv_resident +
                              staging + compute_final + process_floor + margin;

    // FIT INVARIANT: for a returned non-refused plan, planned_peak_bytes <=
    // usable. We sized slots from the remainder, so this holds by construction
    // whenever available_for_experts >= lower*per_bundle. If even the minimal
    // config (slots == n_experts_used floor) with the smallest safe KV window
    // STILL overflows, refuse LOUDLY rather than hand back an OOM plan.
    if (plan.planned_peak_bytes > usable) {
        plan.fits = false;

        // BUG 1 FIX (task-fix-target-and-max-mode-semantics): report an HONEST
        // minimal-config floor. The KV step-down above keeps chosen_window ==
        // n_ctx when EVEN the smallest window does not fit, so the refusal used
        // to name FULL n_ctx KV (8192 MiB for Mixtral) as the "minimal config"
        // -> a dishonest ~10 GiB floor, though the user proved ~2.4 GiB works at
        // a 2048 window. Recompute the TRUE minimal configuration the engine can
        // actually run: the SMALLEST SAFE KV window (kMinKvWindow), MAXIMAL
        // per-layer trunk streaming (hot pool = 1 layer), and the n_experts_used
        // expert floor. The reported floor/KV figures are then the smallest this
        // model fits in, so the user can raise the target to an achievable value
        // (an achievable target >= this floor FITS via the step-down + streaming
        // paths above). When geometry is unknown we fall back to the chosen
        // window / current trunk so a synthetic/dry-run shape still reports
        // something coherent.
        //
        // IMPORTANT: this ONLY recomputes figures for the honest refusal_reason
        // message. It does NOT clobber plan.expert_slots_resident /
        // planned_peak_bytes / kv_window etc. with the minimal floor, because
        // those are the natural plan the caller still inspects (e.g. the VRAM
        // -resident-experts case where the RAM-peak accounting is pessimistic
        // but the slot choice is legitimately full). The API/CLI refuse the
        // load on plan.fits == false and surface this reason; the floor it names
        // is the TRUE smallest-window minimal config.
        uint32_t floor_window = plan.kv_window ? plan.kv_window : chosen_window;
        uint64_t floor_kv = kv_resident;
        if (model.n_ctx > 0 && model.head_dim && model.n_head_kv) {
            floor_window = kMinKvWindow <= model.n_ctx
                               ? kMinKvWindow
                               : static_cast<uint32_t>(model.n_ctx);
            floor_kv = kv_bytes_for_window(model, floor_window);
        }
        const uint64_t floor_compute = compute_estimate_bytes(model, floor_window);
        // Maximal trunk streaming: keep a single attention layer hot (the
        // engine's floor), so the resident trunk is the non-attention trunk plus
        // one hot layer. When nothing is streamable this equals the full trunk.
        uint32_t floor_pool = 0;
        uint64_t floor_trunk = trunk_in_ram;
        if (per_layer_attn > 0) {
            const uint64_t non_attn_trunk =
                trunk_in_ram > attn_trunk_total ? trunk_in_ram - attn_trunk_total
                                                : 0;
            floor_trunk = non_attn_trunk + per_layer_attn;  // 1 hot layer
            floor_pool = 1;
        }
        const uint64_t floor_experts = lower_experts * per_bundle;
        const uint64_t floor_peak = floor_trunk + floor_experts + floor_kv +
                                    staging + floor_compute + process_floor +
                                    margin;
        const bool floor_windowed =
            floor_window && model.n_ctx && floor_window < model.n_ctx;

        std::ostringstream os;
        os << "cannot fit model '" << model.name << "' in budget: minimal "
           << "configuration (" << lower_experts << " expert bundles";
        if (floor_windowed)
            os << ", KV window " << floor_window;
        if (floor_pool)
            os << ", trunk stream[hot=" << floor_pool << " layers]";
        os << ") projects peak " << (floor_peak / kMiB) << " MiB";
        if (floor_peak > usable) {
            // The minimal floor itself exceeds the budget: name the overage so
            // the user knows how much more RAM/budget the floor needs.
            os << ", which is " << ((floor_peak - usable) / kMiB)
               << " MiB over the usable RAM " << (usable / kMiB)
               << " MiB. The achievable floor (this minimal projected peak of "
               << (floor_peak / kMiB) << " MiB) is the smallest this model fits "
               << "in; raise the target to at least that.";
        } else {
            // The minimal floor would fit under usable, but the natural plan's
            // peak still refuses (the pessimistic VRAM-resident-experts RAM
            // accounting the comment above describes). Do NOT claim a "0 MiB
            // over" overage - that would imply the load fits while it is
            // refused. Frame the floor as the achievable target to aim at and
            // make clear the refusal is driven by the natural plan's peak.
            os << " against the usable RAM " << (usable / kMiB)
               << " MiB. This minimal projected peak is the achievable floor "
               << "(the smallest this model fits in); the load is refused "
               << "because the natural plan's projected peak "
               << (plan.planned_peak_bytes / kMiB)
               << " MiB exceeds usable, so target this floor instead.";
        }
        os << " Dominant components: trunk-in-RAM "
           << (floor_trunk / kMiB) << " MiB, experts "
           << (floor_experts / kMiB) << " MiB, KV " << (floor_kv / kMiB)
           << " MiB, staging " << (staging / kMiB) << " MiB, compute "
           << (floor_compute / kMiB) << " MiB, floor " << (process_floor / kMiB)
           << " MiB, margin " << (margin / kMiB) << " MiB.";
        plan.refusal_reason = os.str();
    }

    // NOTE: we intentionally do NOT log the plan summary here. plan_placement
    // is a pure function called several times on a single load (a cheap GGUF
    // metadata pre-pass plus the authoritative post-load plan), and on an
    // explicit --expert-slots run the reconciliation to the forced resident
    // count happens later in sf_context_create. The authoritative summary is
    // logged once by sf_context_create AFTER reconciliation (and the CLI prints
    // it via sf_describe_plan), so the logged line always matches what the
    // engine runs.
    return plan;
}

std::string PlacementPlan::to_summary() const {
    std::ostringstream os;
    os << "trunk[vram=" << trunk_layers_in_vram
       << " ram=" << trunk_layers_in_ram << " layers]";
    if (expert_slots_resident > 0 || stream_experts) {
        os << ", experts[resident=" << expert_slots_resident << " bundles], "
           << (stream_experts ? "experts STREAMED (bounded)"
                              : "experts fully resident");
    } else {
        os << ", dense (no experts)";
    }
    // When an auto KV window was chosen (lossless paging), surface it so the
    // Colab plan line shows the conservative decision.
    if (kv_window > 0) {
        os << ", kv window=" << kv_window;
    }
    // FEAT-003: when the per-layer attention trunk streams, surface the bounded
    // hot-trunk pool depth so the Colab plan line shows the FIT decision.
    if (trunk_pool_layers > 0) {
        os << ", trunk stream[hot=" << trunk_pool_layers << " layers]";
    }
    // Report the peak at the largest unit that does not floor to 0, so small
    // fixtures read truthfully instead of showing "~0 GiB".
    const uint64_t kMiBl = 1024ull * 1024;
    const uint64_t kGiBl = 1024ull * 1024 * 1024;
    if (planned_peak_bytes >= kGiBl) {
        os << ", est. peak ~" << (planned_peak_bytes / kGiBl) << " GiB";
    } else if (planned_peak_bytes >= kMiBl) {
        os << ", est. peak ~" << (planned_peak_bytes / kMiBl) << " MiB";
    } else {
        os << ", est. peak ~" << (planned_peak_bytes / 1024ull) << " KiB";
    }
    if (!fits) {
        os << " [DOES NOT FIT: " << refusal_reason << "]";
    }
    return os.str();
}

} // namespace sf

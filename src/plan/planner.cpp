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
uint64_t usable_ram(const HardwareProfile &hw) {
    const uint64_t total = hw.ram_total_bytes;
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
    uint64_t est = 512 * kMiB;
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
uint64_t safety_margin_bytes(uint64_t usable) {
    return std::max<uint64_t>(512 * kMiB, usable / 100 * 8);  // >=512 MiB or 8%
}

} // namespace

PlacementPlan plan_placement(const HardwareProfile &hw,
                             const ModelShape &model,
                             uint64_t vram_budget,
                             uint64_t ram_budget,
                             uint64_t cache_budget) {
    PlacementPlan plan;

    uint64_t vram = vram_budget;
    if (vram == 0 && !hw.gpus.empty()) {
        vram = hw.gpus.front().vram_bytes;
    }
    // CONSERVATIVE usable RAM ceiling. An explicit ram_budget overrides it (the
    // caller asked for an exact number); 0 = auto from the profile.
    const uint64_t usable = ram_budget ? ram_budget : usable_ram(hw);
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
    const uint64_t fixed_wo_kv =
        trunk_in_ram + staging + process_floor + margin +
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
                    trunk_in_ram + staging + process_floor + margin +
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

    const uint64_t compute = compute_estimate_bytes(model, chosen_window);

    // Size the expert cache from the REMAINDER after every other mandatory
    // resident cost (the conservative budget). This REPLACES the old
    // ram_for_experts = vram_left + ram.
    const uint64_t reserved =
        trunk_in_ram + kv_resident + staging + compute + process_floor + margin;
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

    plan.expert_slots_resident = static_cast<uint32_t>(slots);
    plan.expert_slots_ram =
        per_bundle > 0 ? static_cast<uint32_t>(
                             (usable > reserved ? usable - reserved : 0) /
                             per_bundle)
                       : 0;
    plan.stream_experts = slots < full_slots;

    plan.kv_window = used_auto_window ? chosen_window : 0;
    plan.kv_resident_bytes = kv_resident;
    plan.staging_bytes = staging;
    plan.compute_bytes = compute;
    plan.process_floor_bytes = process_floor;
    plan.safety_margin_bytes = margin;

    // HONEST, CONSERVATIVE peak: trunk-in-RAM + resident experts + KV(chosen
    // window) + one-layer staging at full n_expert + compute + floor + margin.
    const uint64_t resident_experts = slots * per_bundle;
    plan.planned_peak_bytes = trunk_in_ram + resident_experts + kv_resident +
                              staging + compute + process_floor + margin;

    // FIT INVARIANT: for a returned non-refused plan, planned_peak_bytes <=
    // usable. We sized slots from the remainder, so this holds by construction
    // whenever available_for_experts >= lower*per_bundle. If even the minimal
    // config (slots == n_experts_used floor) with the smallest safe KV window
    // STILL overflows, refuse LOUDLY rather than hand back an OOM plan.
    if (plan.planned_peak_bytes > usable) {
        plan.fits = false;
        const uint64_t over = plan.planned_peak_bytes - usable;
        std::ostringstream os;
        os << "cannot fit model '" << model.name << "' in budget: minimal "
           << "configuration (" << slots << " expert bundles";
        if (plan.kv_window)
            os << ", KV window " << plan.kv_window;
        os << ") projects peak " << (plan.planned_peak_bytes / kMiB)
           << " MiB, which is " << (over / kMiB) << " MiB over the usable RAM "
           << (usable / kMiB) << " MiB. Dominant components: trunk-in-RAM "
           << (trunk_in_ram / kMiB) << " MiB, experts "
           << (resident_experts / kMiB) << " MiB, KV " << (kv_resident / kMiB)
           << " MiB, staging " << (staging / kMiB) << " MiB, compute "
           << (compute / kMiB) << " MiB, floor " << (process_floor / kMiB)
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

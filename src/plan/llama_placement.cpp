// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "plan/llama_placement.h"

namespace sf {

LlamaPlacement derive_llama_placement(const PlacementPlan &plan,
                                      const ModelShape &model,
                                      const std::vector<GpuInfo> &gpus) {
    LlamaPlacement out;

    // No GPU => everything stays on the CPU. n_gpu_layers must be 0 and no
    // tensor overrides are needed (all tensors already land on the host buffer
    // type by default). This keeps CPU-only behaviour identical to Phase 1b.
    if (gpus.empty()) {
        return out;
    }

    // Trunk layers the planner chose to pin in VRAM map straight onto
    // llama_model_params::n_gpu_layers. The plan never pins more layers than
    // the model has, but clamp defensively anyway.
    uint32_t ngl = plan.trunk_layers_in_vram;
    if (ngl > model.n_layers) {
        ngl = model.n_layers;
    }
    out.n_gpu_layers = static_cast<int32_t>(ngl);

    // Expert offload only applies to MoE models. We force the routed expert
    // FFN tensors onto the CPU/host buffer type whenever the plan says the
    // experts cannot all live resident in VRAM: either it decided to stream
    // them, or the VRAM expert cache cannot hold every expert of every layer.
    if (model.is_moe && model.n_experts > 0 && model.n_layers > 0) {
        const uint64_t total_experts =
            static_cast<uint64_t>(model.n_experts) * model.n_layers;
        const bool experts_all_in_vram =
            !plan.stream_experts &&
            static_cast<uint64_t>(plan.expert_slots_vram) >= total_experts;
        if (!experts_all_in_vram) {
            out.offload_experts_to_cpu = true;
            out.expert_override_pattern = expert_ffn_regex();
        }
    }

    return out;
}

} // namespace sf

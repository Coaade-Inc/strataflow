// Translation layer: turns an abstract sf::PlacementPlan into the concrete
// knobs llama.cpp understands (n_gpu_layers + an expert->CPU tensor override).
//
// This is deliberately a PURE function over plain structs: it takes the plan,
// the model shape, and the detected GPU list, and returns plain data. It does
// NOT touch any llama.cpp / ggml type, so it is unit-testable with no GPU and
// no real model. The actual ggml_backend_buffer_type_t resolution (turning
// "put experts on CPU" into a buft pointer and a NULL-terminated override
// array) happens in model/ggml_model.cpp, where the llama headers live.
// See docs/PLAN.md section 3.2 and the user instruction for Phase 2.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "hw/profiler.h"
#include "plan/planner.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sf {

// Concrete llama.cpp load parameters derived from a PlacementPlan.
struct LlamaPlacement {
    // Maps directly to llama_model_params::n_gpu_layers: the number of trunk
    // layers to store in VRAM. Zero on a CPU-only machine.
    int32_t n_gpu_layers = 0;

    // When true, the MoE expert FFN tensors must be routed to the CPU/host
    // buffer type (the llama.cpp equivalent of --n-cpu-moe / -ot "exps=CPU").
    // Set only for MoE models that cannot keep every expert resident in VRAM.
    bool offload_experts_to_cpu = false;

    // The llama.cpp tensor-name regex used for the expert override. Matches the
    // ffn expert tensors (blk.N.ffn_{gate,down,up,gate_up}_exps). Empty unless
    // offload_experts_to_cpu is true. Kept here (rather than hard-coded in the
    // model TU) so tests can assert on it.
    std::string expert_override_pattern;
};

// llama.cpp's regex for the routed MoE expert FFN tensors. Replicated from the
// vendored common/common.h (LLM_FFN_EXPS_REGEX) because that TU (common) is not
// compiled into our build. Keep in sync with the pinned llama.cpp tag.
inline const char *expert_ffn_regex() {
    return "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
}

// Derive the concrete llama.cpp placement from the abstract plan. Pure: no
// side effects, no llama/ggml calls. On a CPU-only machine (gpus empty) this
// returns {0, false, ""} so load behaviour is identical to a plain CPU load.
LlamaPlacement derive_llama_placement(const PlacementPlan &plan,
                                      const ModelShape &model,
                                      const std::vector<GpuInfo> &gpus);

} // namespace sf

// strata_stream_buft: the streaming ggml backend buffer type that owns the
// routed-expert FFN tensors (the TWS's integration point with llama.cpp).
//
// This is the production form of the Phase 3 feasibility spike
// (spike/stream_decode/spike_stream.cpp), which proved that llama.cpp can
// decode a MoE graph byte-identically while expert tensors are owned by a
// custom ggml_backend_buffer_type whose bytes are faulted in on demand, with
// zero edits to ggml or llama. See docs/PHASE3_PLAN.md sections 2 and 8.
//
// Design points inherited from the spike:
//   - is_host() is true: the CPU MoE kernel reads the slot bytes directly
//     through tensor->data at compute time (there is no per-op ggml hook).
//   - alloc_buffer() allocates a REAL contiguous region of the requested size:
//     ggml's linear allocator places every tensor at an offset inside one
//     get_base() region, so a dummy base fails with "not enough space in the
//     buffer". For Task 2 the region is sized to the full expert footprint
//     (simplest correct v1); the bounded slot pool is Task 3.
//   - init_tensor() parses the expert tensor name (blk.N.ffn_{gate,down,up}
//     _exps) into {layer, kind} and records the tensor's byte range in the
//     backing store; get_alloc_size() returns ggml_nbytes(tensor).
//   - set_tensor() stores expert bytes into the backing store at load time;
//     get_tensor() reads them back.
//
// Deferred (do NOT implement here):
//   - Task 3: residency pre-pass + SlotPool-under-decode + bounded memory
//     (a region smaller than the full footprint). The SlotPool/BlockFile seam
//     is kept clean so Task 3 can plug those in.
//   - Task 4: native direct-I/O async backends behind BlockFile.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// This header does NOT include any ggml header: the pure name-parsing and
// identity types are usable by unit tests without llama/ggml. The ggml bridge
// (the opaque buffer-type pointer) is exposed via forward-declared getters.
struct ggml_backend_buffer_type;

namespace sf {

// Identifies one routed-expert FFN weight tensor within the model. A Mixtral/
// llama-arch MoE layer stacks all experts into three 3-D tensors
// (ffn_gate_exps, ffn_down_exps, ffn_up_exps), so the identity is the layer
// plus which of the three stacked tensors this is. (The per-expert id used by
// the SlotPool in Task 3 is {layer, expert}; here, at buffer-ownership level,
// the unit is the whole stacked tensor, which is what ggml allocates.)
enum class ExpertTensorKind : uint8_t {
    kGate = 0,  // blk.N.ffn_gate_exps
    kDown = 1,  // blk.N.ffn_down_exps
    kUp   = 2,  // blk.N.ffn_up_exps
    kGateUp = 3,  // blk.N.ffn_gate_up_exps (fused gate+up, some arches)
};

struct ExpertTensorId {
    uint32_t         layer = 0;
    ExpertTensorKind kind  = ExpertTensorKind::kGate;
    bool valid = false;  // false when the name did not parse as an expert tensor

    bool operator==(const ExpertTensorId &o) const {
        return valid == o.valid && layer == o.layer && kind == o.kind;
    }
};

// Parse a ggml tensor name into an ExpertTensorId. Recognizes the routed-expert
// FFN tensors "blk.<N>.ffn_{gate,down,up,gate_up}_exps[.weight]" (matching
// llama.cpp's LLM_FFN_EXPS_REGEX, see expert_ffn_regex()). Returns a result
// with valid == false for any non-expert tensor name. Pure; no ggml needed, so
// it is directly unit-testable.
ExpertTensorId parse_expert_tensor_name(const std::string &name);

// Opaque accounting the integration test and unit tests assert on. One global
// instance tracks every registration/store that flowed through the active
// stream buffer type, so a test can prove experts were owned by this buft.
struct StreamBuftStats {
    uint64_t registrations = 0;   // init_tensor calls that parsed as experts
    uint64_t non_expert    = 0;   // init_tensor calls for other tensors
    uint64_t stored_bytes  = 0;   // total bytes written via set_tensor
    uint64_t buffers       = 0;   // alloc_buffer calls
};

// Returns the process-wide stats for the streaming buffer type. Reset with
// reset_stream_buft_stats() at the start of a test.
const StreamBuftStats &stream_buft_stats();
void reset_stream_buft_stats();

// Returns the singleton strata_stream_buft, bound to the CPU device so llama's
// allocator and scheduler treat it as a host buffer. The pointer is stable for
// the process lifetime; load_ggml_model points the Phase 2 expert override at
// it when streaming. Returns nullptr only if the CPU device is unavailable
// (never on a normal CPU build). Defined in stream_buft.cpp, which includes the
// private ggml-backend-impl.h.
ggml_backend_buffer_type *strata_stream_buft();

// The name reported by the buffer type's get_name() (and stamped on tensors
// via llama's buffer-type query). Exposed for tests.
const char *strata_stream_buft_name();

}  // namespace sf

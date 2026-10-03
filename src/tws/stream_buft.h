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
//     buffer".
//   - init_tensor() parses the expert tensor name (blk.N.ffn_{gate,down,up}
//     _exps) into {layer, kind} and records the tensor's byte range in the
//     backing store; get_alloc_size() returns ggml_nbytes(tensor).
//   - set_tensor() stores expert bytes into the backing store at load time;
//     get_tensor() reads them back.
//
// Task 3 (THIS task) — residency + exact-mode correctness:
//   The buffer is now backed by a bounded SlotPool (src/tws/weight_store.h).
//   Each whole stacked expert tensor (blk.N.ffn_{gate,down,up}_exps — the ggml
//   allocation and streaming unit; see the mul_mat_id note below) is a cache
//   entry keyed by {layer, kind}. A residency pass
//   (stream_buft_ensure_decode_residency) runs before every llama_decode and
//   acquires every registered expert tensor through the SlotPool, loading its
//   recorded byte range from the backing store (via BlockFile when a file path
//   is configured, else the in-memory store) into the tensor's resident
//   region. This drives the LRU hits/misses/evictions and bounds the resident
//   working copies to n_slots * slot_bytes.
//
//   CORRECTNESS MODEL (A), confirmed against the vendored llama.cpp (b11379):
//   one llama_decode builds and runs the FULL compute graph for all layers in a
//   single call. The MoE CPU kernel ggml_compute_forward_mul_mat_id addresses
//   expert cur_a as `src0->data + cur_a*nb02` — a direct offset from the ONE
//   stacked tensor's base pointer (ggml-cpu.c:1683). ggml exposes no per-op
//   residency hook that would let us fault individual stacked tensors in and
//   out *within* one decode (the eval callback only batches node ranges; it
//   cannot relocate a weight leaf's data mid-graph safely). THEREFORE every
//   stacked expert tensor a decode will read must be resident at its own
//   distinct address when the graph runs. The streaming unit is the WHOLE
//   stacked tensor per (layer, kind), NOT an individual expert — placing only
//   the top-k experts of a layer in a slot would make i02*nb02 land on the
//   wrong bytes. "Bounded" therefore means: bounded to the model's
//   expert-tensor working set within a decode, with LRU eviction/reload across
//   the SlotPool budget *across decodes*. True per-expert sub-tensor streaming
//   (top-k rows only) needs a custom kernel / the .strata layout and is
//   DEFERRED (Task 5+). Native direct-I/O (io_uring/IOCP/F_NOCACHE) is Task 4.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
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

    // Task 3 residency accounting, surfaced from the backing SlotPool so a test
    // can prove bounded, cache-driven residency.
    uint64_t cache_hits      = 0;  // residency acquire served from a slot
    uint64_t cache_misses    = 0;  // residency acquire required a store load
    uint64_t cache_evictions = 0;  // LRU evictions forced by the pool budget
    uint64_t disk_reads      = 0;  // byte-range loads from the backing store
    uint64_t resident_slots  = 0;  // slots currently holding a tensor
    uint64_t slot_bytes      = 0;  // bytes per slot (max stacked-tensor nbytes)
    uint64_t n_slots         = 0;  // configured pool size

    // Resident bytes held by the cache tier; bounded by n_slots * slot_bytes.
    uint64_t resident_cache_bytes() const { return resident_slots * slot_bytes; }
};

// Returns the process-wide stats for the streaming buffer type. Reset with
// reset_stream_buft_stats() at the start of a test.
const StreamBuftStats &stream_buft_stats();
void reset_stream_buft_stats();

// Task 3 configuration knob. Set BEFORE the model is loaded (so alloc_buffer
// sizes the SlotPool). n_slots == 0 means "auto": size the pool to hold the
// whole expert-tensor working set (the always-correct, no-eviction default).
// A positive n_slots forces a bounded pool so a test can drive eviction/reload
// across decodes. The planner wires this from PlacementPlan::expert_slots_*; a
// test can override it explicitly via set_stream_buft_slots().
struct StreamBuftConfig {
    uint32_t n_slots = 0;  // 0 = auto (hold the full working set)
};

// Set/get the slot-pool size used by the NEXT alloc_buffer call. Thread-safe.
void     set_stream_buft_slots(uint32_t n_slots);
uint32_t stream_buft_slots();

// Task 5: configure a .strata byte source for the streaming buffer. When set
// BEFORE the model is loaded, the residency pass sources each stacked expert
// tensor's bytes from the .strata expert region (one aligned read per
// {layer,expert,kind} slice, reassembled into the stacked tensor at its
// per-expert nb02 offsets) instead of the in-memory store filled by set_tensor.
// This only changes the PROVENANCE of the bytes (aligned .strata blob vs the
// scattered GGUF range), not their values, so decode stays byte-identical.
//
// The pointer is borrowed; the caller (load_ggml_model) owns the StrataReader
// and must keep it alive for the model's lifetime. Pass nullptr to clear
// (restoring the GGUF-direct / in-memory fallback). Forward-declared so this
// header stays ggml/llama-free.
class StrataReader;
void          set_stream_buft_strata_reader(StrataReader *reader);
StrataReader *stream_buft_strata_reader();

// Residency pass: ensure every registered expert tensor is resident in its
// region before llama_decode runs the graph. Call this immediately before each
// llama_decode (and before the prompt decode). Loads each stacked expert
// tensor's recorded byte range from the backing store through the SlotPool,
// updating the LRU/stats. `prefill` marks a prompt/micro-batch so a batch that
// touches more tensors than the pool has slots streams without polluting the
// decode hot set (PLAN 3.3 prefill-bypass). No-op when no stream buffer has
// been allocated. Safe to call when streaming is inactive.
void stream_buft_ensure_decode_residency(bool prefill);

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

// Engine core (EC-1): StrataFlow's own ggml forward pass for the llama-arch
// MoE, replacing libllama's llama_decode inference path. See
// docs/ENGINE_CORE_DESIGN.md sections 6 (arch + exact op sequence) and 7
// (oracle correctness gate).
//
// This header deliberately pulls in NO llama/ggml headers: it is the clean seam
// that GgmlModel and the oracle test sit on. The implementation
// (model/engine/engine.cpp) is the only engine TU that includes ggml.
//
// EC-3 scope: per-layer SEGMENTED execution with top-k expert residency
// (sections 3.2, 3.3, 3.5). Each layer is split into a ROUTER segment (attn +
// ffn_norm + router + argsort top-k, computed and then read back to the host)
// and an EXPERT segment (mul_mat_id over the now-resident experts). Because the
// engine reads the router output BEFORE building the expert matmul, it makes
// resident EXACTLY the top-k routed experts per layer via a bounded SlotPool
// (src/tws/weight_store.h), bounding resident expert RAM to top-k per layer.
// EC-2's engine-owned KV cache and the exact math (section 6.3) are unchanged,
// so output stays byte-identical to EC-1/EC-2.
//
// EC-4 scope: the .strata single-file loader (section 5). load() detects a
// .strata (magic STRATA01) and reads the model shape from the embedded GGUF
// metadata blob, the TRUNK tensors from the .strata trunk block, and streams
// EXPERT tensors from the .strata expert region through the SAME bounded
// SlotPool (now sourcing bytes from StrataReader::read_blob instead of resident
// GGUF tensors). No llama_model_load_from_file is ever handed a .strata, so the
// TASK5_BLOCKER root cause is gone. The plain-GGUF path (sections 5.2, EC-1/2/3)
// is retained unchanged; only the weight byte provenance differs, so decode
// stays byte-identical. The engine path is opt-in behind GgmlModel; one
// architecture only (the llama-arch MoE fixture; the arch-descriptor table is
// EC-7).
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sf {
namespace engine {

// Hyperparameters for the one architecture EC-1 targets (the llama-arch MoE).
// Read from GGUF metadata at load time and validated against tensor shapes.
struct EngineHParams {
    int   n_embd        = 0;
    int   n_head        = 0;
    int   n_head_kv     = 0;
    int   n_layer       = 0;
    int   n_ff          = 0;   // kept for reference; derived from expert tensors
    int   n_expert      = 0;
    int   n_expert_used = 0;
    int   head_dim      = 0;   // n_embd / n_head
    int   n_ctx_orig    = 0;   // context length (rope n_ctx_orig)
    int   n_vocab       = 0;
    float rms_eps       = 1e-5f;
    float freq_base     = 10000.0f;
    float freq_scale    = 1.0f;
};

// Top-k expert residency accounting (EC-3), surfaced from the engine's bounded
// SlotPool so a test can prove bounded, cache-driven residency. The hits/misses
// /evictions mirror src/tws/weight_store.h CacheStats; resident_bytes is the
// hard upper bound on resident expert working-copy RAM (n_slots * slot_bytes).
struct EngineCacheStats {
    uint64_t hits        = 0;   // routed expert served from a resident slot
    uint64_t misses      = 0;   // routed expert required a load from the source
    uint64_t evictions   = 0;   // LRU evictions forced by the pool budget
    uint64_t n_slots     = 0;   // configured pool size (expert bundles)
    uint64_t slot_bytes  = 0;   // bytes per slot (gate+up+down for one expert)
    uint64_t full_bytes  = 0;   // all-experts-resident footprint, for comparison
    // EC-6 predictor-driven prefetch (section 3.5): experts warmed into the pool
    // ahead of a layer, and how many that layer's exact routing then used. The
    // hit rate is the measurable EC-6 win; prefetch never affects correctness
    // (the exact router read-back is authoritative).
    uint64_t prefetch_warmed = 0;
    uint64_t prefetch_used   = 0;
    double prefetch_hit_rate() const {
        return prefetch_warmed ? double(prefetch_used) / double(prefetch_warmed) : 0.0;
    }
    // Resident expert RAM upper bound; bounded by top-k per layer when the pool
    // is sized below the expert count.
    uint64_t resident_bytes() const { return n_slots * slot_bytes; }
};

// Owns the model's weight tensors (read resident from a GGUF) and runs OUR OWN
// ggml graph forward on the ggml CPU backend. EC-3 executes each layer in a
// router segment and an expert segment, making only the top-k routed experts
// resident through a bounded SlotPool. One instance per model; forward() is not
// thread-safe (single-token greedy decode path).
class Engine {
public:
    ~Engine();

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    // Load the llama-arch MoE weights from `path` resident in a ggml context.
    // `path` may be a plain GGUF or a .strata single file (EC-4, auto-detected
    // by magic). Returns nullptr on any failure (bad file, unexpected arch,
    // missing tensors); the caller then stays on the libllama path.
    //
    // `expert_slots` bounds the top-k residency SlotPool (EC-3): the number of
    // expert bundles (gate+up+down for one (layer,expert)) that may be resident
    // at once. 0 means "auto": size the pool to the full expert working set
    // (n_layer * n_expert, no eviction). A positive value smaller than
    // n_expert forces streaming with LRU eviction/reload so a test can prove
    // bounded residency. Must be >= n_expert_used so a single layer's top-k
    // always fits; load() clamps up to that minimum.
    static std::unique_ptr<Engine> load(const std::string &path,
                                        uint32_t expert_slots = 0);

    const EngineHParams &hparams() const;

    // Top-k residency cache stats accumulated across all forward() calls since
    // load() (EC-3). Lets the oracle test assert hits/misses/evictions and the
    // bounded resident-byte footprint.
    const EngineCacheStats &cache_stats() const;

    // Run a forward for `token` at `pos`, writing the full logit vector (length
    // n_vocab) to `out`. `token` is appended to the engine-owned KV cache at
    // `pos` and the query attends over all cached positions 0..pos (causal).
    // Callers must drive positions in order from 0 for a given sequence; call
    // reset_kv() before starting a new sequence. `pos` must be < n_ctx (the
    // trained context length) or the call fails. Returns false on build/compute
    // failure or an out-of-bounds position.
    bool forward(int32_t token, int32_t pos, std::vector<float> &out);

    // Clear the KV cache so the next forward() starts a fresh sequence at
    // position 0. Does not free the cache buffers (they are reused).
    void reset_kv();

private:
    Engine();

    struct Impl;

    // EC-3: load the routed top-k experts of layer `il` into the staging
    // stacked tensors via the bounded SlotPool. `ids` holds `n_ids` selected
    // expert ids read back from the router segment. Returns false on failure.
    bool ensure_layer_experts_resident(int il, const int32_t *ids, int n_ids);

    // EC-6: warm one (layer, expert) bundle into the SlotPool WITHOUT copying it
    // to the staging tensors - a speculative prefetch. Loads the slot on a miss
    // using the same byte source as ensure_layer_experts_resident, so a later
    // authoritative acquire of the same id becomes a cache hit. Best-effort: a
    // failed/out-of-range warm is ignored (correctness is unaffected because the
    // authoritative residency pass re-acquires exactly what the router selects).
    void warm_expert(int il, int expert);
    const EngineHParams &hp_() const;

    // EC-4: open a .strata single file, parse its embedded GGUF metadata into
    // im.gc, build the resident weight context (im.wctx) from that metadata,
    // fill the TRUNK tensors from the .strata trunk block, and retain the
    // StrataReader (im.reader) as the expert byte source. Returns false on any
    // failure (bad superblock, metadata parse, trunk read). Defined in
    // model/engine/engine.cpp (the only TU that may include ggml headers).
    static bool load_strata_weights(Impl &im, const std::string &path);

    std::unique_ptr<Impl> impl_;
};

// Oracle helper for the EC-1 correctness gate (docs/ENGINE_CORE_DESIGN.md 7.3).
// Runs libllama's llama_decode on the SAME gguf for one token at position 0 and
// writes the resulting logits to `out`. This lives in the engine TU because it
// is the only place that may include <llama.h> for tests above the sf::Model
// seam. Returns false on load/decode failure. Test-only; not on the hot path.
bool run_oracle_single_token(const std::string &path, int32_t token,
                             std::vector<float> &out);

// The BOS token id for `path`'s vocab, or -1 on failure. Used by the oracle
// test to pick the same seed token the oracle and engine both run.
int32_t vocab_bos_token(const std::string &path);

// Multi-token oracle helper for the EC-2 sequence gate (docs/ENGINE_CORE_DESIGN
// section 7.3). Tokenizes `prompt` with the model vocab, runs libllama's
// llama_decode over the prompt and then greedily generates `n_generate` tokens
// (argmax each step, feeding the previous token back), carrying the llama KV
// cache. Writes the generated token id SEQUENCE to `out_tokens` and, when
// `out_last_logits` is non-null, the full logit vector produced at each step
// (prompt-final step first, then one per generated token) flattened in order.
// This lives in the engine TU because it is the only place that may include
// <llama.h> for tests above the sf::Model seam. Returns false on load/decode
// failure. Test-only; not on the hot path.
bool run_oracle_sequence(const std::string &path, const std::string &prompt,
                         int n_generate, std::vector<int32_t> &out_tokens,
                         std::vector<std::vector<float>> *out_step_logits);

// Tokenize `prompt` with `path`'s vocab (add_special=true, matching
// run_oracle_sequence), writing the ids to `out`. Lets the engine test drive
// the SAME prompt tokens the oracle used without pulling <llama.h> above the
// seam. Returns false on load/tokenize failure.
bool vocab_tokenize(const std::string &path, const std::string &prompt,
                    std::vector<int32_t> &out);

} // namespace engine
} // namespace sf

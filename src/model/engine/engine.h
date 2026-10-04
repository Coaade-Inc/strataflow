// Engine core (EC-1): StrataFlow's own ggml forward pass for the llama-arch
// MoE, replacing libllama's llama_decode inference path. See
// docs/ENGINE_CORE_DESIGN.md sections 6 (arch + exact op sequence) and 7
// (oracle correctness gate).
//
// This header deliberately pulls in NO llama/ggml headers: it is the clean seam
// that GgmlModel and the oracle test sit on. The implementation
// (model/engine/engine.cpp) is the only engine TU that includes ggml.
//
// EC-2 scope: a prefill-then-decode forward over an engine-owned KV cache
// (section 3.4). Each forward() processes one query token at its position,
// appends that layer's post-rope K and V to a per-layer KV store, and attends
// over all cached positions 0..pos (causal mask through ggml_soft_max_ext).
// Still ALL expert weights resident (no streaming - that is EC-3), one
// architecture only (the llama-arch MoE fixture - the arch-descriptor table is
// EC-7). The engine path is opt-in behind GgmlModel.
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

// Owns the model's weight tensors (read resident from a GGUF) and runs OUR OWN
// ggml graph for a single-token forward on the ggml CPU backend. One instance
// per model; forward() is not thread-safe (single-token greedy decode path).
class Engine {
public:
    ~Engine();

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    // Load the llama-arch MoE weights from `path` (plain GGUF) resident in a
    // ggml context. Returns nullptr on any failure (bad file, unexpected arch,
    // missing tensors); the caller then stays on the libllama path.
    static std::unique_ptr<Engine> load(const std::string &path);

    const EngineHParams &hparams() const;

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

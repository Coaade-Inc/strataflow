// Paged KV cache. See docs/PLAN.md section 3.9 and
// docs/ENGINE_CORE_DESIGN.md section 3.4 (engine KV cache ownership).
//
// Phase 1 shipped a bookkeeping stub tracking context length and byte usage.
// EC-2 keeps that bookkeeping role (the engine owns the actual per-layer K/V
// ggml tensors, whose lifetimes must match the ggml weight context) and uses
// KvStore as the single source of truth for the current fill position and the
// n_ctx bound. Quantized/paged storage, MLA, and RAM/SSD spill land in Phase 6.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstdint>

namespace sf {

class KvStore {
public:
    KvStore(uint32_t n_ctx, uint64_t bytes_per_token);

    bool advance(uint32_t n_tokens);   // false if it would exceed n_ctx
    void reset() { used_ = 0; }

    // True if `n_tokens` more positions still fit within n_ctx. The engine
    // checks this before building a forward graph so it never writes past the
    // trained context length (EC-2 bounds the cache by n_ctx).
    bool can_fit(uint32_t n_tokens) const {
        return uint64_t(used_) + n_tokens <= n_ctx_;
    }

    uint32_t used() const { return used_; }
    uint32_t capacity() const { return n_ctx_; }
    uint64_t bytes() const { return uint64_t(used_) * bytes_per_token_; }

private:
    uint32_t n_ctx_;
    uint64_t bytes_per_token_;
    uint32_t used_ = 0;
};

} // namespace sf

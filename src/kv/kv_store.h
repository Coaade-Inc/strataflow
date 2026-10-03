// Paged KV cache. See docs/PLAN.md section 3.9.
// Phase 1: a bookkeeping stub tracking context length and byte usage.
// Quantized/paged storage, MLA, and RAM/SSD spill land in Phase 6.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstdint>

namespace sf {

class KvStore {
public:
    KvStore(uint32_t n_ctx, uint64_t bytes_per_token);

    bool advance(uint32_t n_tokens);   // false if it would exceed n_ctx
    void reset() { used_ = 0; }

    uint32_t used() const { return used_; }
    uint32_t capacity() const { return n_ctx_; }
    uint64_t bytes() const { return uint64_t(used_) * bytes_per_token_; }

private:
    uint32_t n_ctx_;
    uint64_t bytes_per_token_;
    uint32_t used_ = 0;
};

} // namespace sf

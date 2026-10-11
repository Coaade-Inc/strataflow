// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "kv/kv_store.h"

namespace sf {

KvStore::KvStore(uint32_t n_ctx, uint64_t bytes_per_token, uint32_t window)
    : n_ctx_(n_ctx), bytes_per_token_(bytes_per_token) {
    // A window of 0 (or >= n_ctx) means full-resident: no paging engages and
    // the behavior is byte-identical to the pre-FEAT-002 path.
    window_ = (window == 0 || window > n_ctx_) ? n_ctx_ : window;
}

bool KvStore::advance(uint32_t n_tokens) {
    if (used_ + n_tokens > n_ctx_) return false;
    used_ += n_tokens;
    return true;
}

} // namespace sf

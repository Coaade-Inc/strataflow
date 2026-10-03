// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "kv/kv_store.h"

namespace sf {

KvStore::KvStore(uint32_t n_ctx, uint64_t bytes_per_token)
    : n_ctx_(n_ctx), bytes_per_token_(bytes_per_token) {}

bool KvStore::advance(uint32_t n_tokens) {
    if (used_ + n_tokens > n_ctx_) return false;
    used_ += n_tokens;
    return true;
}

} // namespace sf

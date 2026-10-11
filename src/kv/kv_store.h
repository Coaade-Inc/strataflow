// Paged KV cache. See docs/PLAN.md section 3.9 and
// docs/ENGINE_CORE_DESIGN.md section 3.4 (engine KV cache ownership).
//
// Phase 1 shipped a bookkeeping stub tracking context length and byte usage.
// EC-2 keeps that bookkeeping role (the engine owns the actual per-layer K/V
// ggml tensors, whose lifetimes must match the ggml weight context) and uses
// KvStore as the single source of truth for the current fill position and the
// n_ctx bound.
//
// FEAT-002 (technique 6.5, LOSSLESS KV paging): KvStore now also owns the
// paging POLICY/bookkeeping. The resident per-layer K/V tensors are bounded to
// a recent WINDOW of W columns (a ring buffer) instead of reserving the full
// trained n_ctx up front; positions that scroll out of the window are tiered to
// a RAM-backed per-layer cold page store (owned by the engine, which holds the
// ggml-touching copies) and re-assembled losslessly for attention so every past
// position 0..pos is still attended over with the SAME F32 bytes. KvStore
// answers the pure bookkeeping questions (window size, whether paging is
// engaged, the ring column for an absolute position, how many positions are
// resident vs cold, which absolute position a ring write evicts) and leaves all
// ggml tensor copies to the engine. When W >= n_ctx no paging engages and the
// behavior is byte-identical to the full-resident path. SSD spill of cold pages
// and KV quantization remain deferred (Phase 6 / Stage-4, approximate).
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstdint>

namespace sf {

class KvStore {
public:
    // `window` is the resident ring-buffer column count W. The engine passes
    // min(n_ctx, requested W); when W >= n_ctx paging never engages. A window
    // of 0 is treated as n_ctx (full-resident), so the default stays unchanged.
    KvStore(uint32_t n_ctx, uint64_t bytes_per_token, uint32_t window = 0);

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

    // ---- FEAT-002 lossless paging bookkeeping ----------------------------

    // Resident ring-buffer column count W (always <= n_ctx).
    uint32_t window() const { return window_; }

    // True when W < n_ctx, i.e. older positions can scroll out of the resident
    // window and must be tiered to the cold store. False means the full-
    // resident path (byte-identical, zero paging overhead).
    bool paging() const { return window_ < n_ctx_; }

    // Ring column index for absolute position `pos` (pos % W). Valid for any
    // position the engine writes; only the most recent W columns are live.
    uint32_t ring_col(uint32_t pos) const { return pos % window_; }

    // Number of resident (in-window) positions when the current fill position
    // is `pos`: the most recent min(pos+1, W) positions. The remaining
    // (pos+1 - that) older positions live in the cold store.
    uint32_t resident_count(uint32_t pos) const {
        const uint32_t n_kv = pos + 1;
        return n_kv < window_ ? n_kv : window_;
    }

    // Number of cold (evicted) positions when the fill position is `pos`:
    // positions 0..(cold_count-1) are in the cold store.
    uint32_t cold_count(uint32_t pos) const {
        return (pos + 1) - resident_count(pos);
    }

    // When writing absolute position `pos` into its ring column overwrites a
    // still-live older position, returns that older absolute position via
    // `*evicted` and true; false when nothing live is overwritten (pos < W).
    // The engine copies the evicted column into the cold store BEFORE the ring
    // write so no past KV is lost.
    bool evicts(uint32_t pos, uint32_t *evicted) const {
        if (pos < window_) return false;
        if (evicted != nullptr) *evicted = pos - window_;
        return true;
    }

private:
    uint32_t n_ctx_;
    uint64_t bytes_per_token_;
    uint32_t used_ = 0;
    uint32_t window_ = 0;
};

} // namespace sf

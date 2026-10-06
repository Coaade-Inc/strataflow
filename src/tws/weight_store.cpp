// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "tws/weight_store.h"

#include <algorithm>

namespace sf {

SlotPool::SlotPool(uint32_t n_slots, uint64_t slot_bytes)
    : n_slots_(n_slots), slot_bytes_(slot_bytes) {
    buffers_.resize(n_slots_);
    free_slots_.reserve(n_slots_);
    for (uint32_t i = 0; i < n_slots_; ++i) {
        buffers_[i].assign(static_cast<size_t>(slot_bytes_), 0);
        free_slots_.push_back(n_slots_ - 1 - i);  // hand out 0,1,2,... in order
    }
}

SlotPool::SlotPool(uint32_t n_slots, uint64_t slot_bytes, uint32_t n_layer,
                   uint32_t n_expert, uint32_t n_expert_used)
    : n_slots_(n_slots), slot_bytes_(slot_bytes) {
    buffers_.resize(n_slots_);
    free_slots_.reserve(n_slots_);
    for (uint32_t i = 0; i < n_slots_; ++i) {
        buffers_[i].assign(static_cast<size_t>(slot_bytes_), 0);
        free_slots_.push_back(n_slots_ - 1 - i);  // hand out 0,1,2,... in order
    }

    // Fall back to the global single-LRU pool when the shape is degenerate.
    // (A zero-layer or zero-expert model has no per-layer sweep to stratify.)
    if (n_layer == 0 || n_expert == 0) return;

    stratified_   = true;
    n_layer_      = n_layer;
    n_expert_     = n_expert;
    n_expert_used_ = n_expert_used == 0 ? 1 : n_expert_used;
    layers_.resize(n_layer_);
    cap_.assign(n_layer_, 0);

    // Start from an even shape-derived split; frequency refines spare slots
    // later via set_layer_weights (frequency is only known after decoding).
    recompute_caps(/*weights=*/{});
}

void SlotPool::recompute_caps(const std::vector<uint64_t> &weights) {
    if (!stratified_) return;

    // Per-layer floor: a single layer's top-k must always fit, but never more
    // than the layer actually has experts, and at least 1 so eviction always
    // has a victim.
    uint32_t floor = std::min(n_expert_used_, n_expert_);
    if (floor == 0) floor = 1;

    // The floor must be affordable across all layers within the total budget.
    // If the budget cannot even give every layer its floor, shrink the floor to
    // the largest value every layer can be guaranteed (>= 1). This keeps the
    // sum(cap_) <= n_slots_ invariant that the acquire fast path relies on.
    //
    // CACHING DEGRADATION, NEVER A CORRECTNESS BREAK: when the total budget is
    // between n_expert_used and n_layer*n_expert_used (reachable in production
    // when a caller requests such a slot count), this shrinks a layer's cap
    // BELOW the single-layer top-k (n_expert_used). That only costs cache hits:
    // the authoritative ensure_layer_experts_resident still acquire()s EXACTLY
    // the router-selected experts before the matmul, and acquire always finds a
    // victim (same-layer LRU, then a free slot, then the most-over-capacity
    // donor layer), so the top-k is always staged with the correct bytes and
    // decode stays byte-identical to the oracle. A sub-top-k cap just means that
    // layer re-streams more of its top-k per token (fewer resident across the
    // sweep); it never changes which bytes compute or the graph order, and the
    // sum(cap_) <= n_slots_ invariant still holds by construction. See
    // test_capacity_below_topk_still_correct in tests/test_slot_pool.cpp.
    if (static_cast<uint64_t>(floor) * n_layer_ > n_slots_) {
        floor = n_slots_ / n_layer_;
        if (floor == 0) floor = 1;
    }

    // Capacity never exceeds the layer's expert count (resident set can't be
    // larger than the number of distinct experts).
    const uint32_t ceil = n_expert_;

    // Assign the floor to every layer first.
    uint64_t used = 0;
    for (uint32_t l = 0; l < n_layer_; ++l) {
        cap_[l] = std::min(floor, ceil);
        used += cap_[l];
    }

    // Distribute the remaining budget. Prefer busier layers (by `weights`) so
    // the hot set stays resident; fall back to an even round-robin when no
    // frequency is known yet. Never exceed a layer's ceil.
    uint64_t spare = used < n_slots_ ? (n_slots_ - used) : 0;
    if (spare == 0) return;

    // Order layers by descending weight (stable by index for ties / no data).
    std::vector<uint32_t> order(n_layer_);
    for (uint32_t l = 0; l < n_layer_; ++l) order[l] = l;
    const bool have_weights = weights.size() == n_layer_;
    if (have_weights) {
        std::stable_sort(order.begin(), order.end(),
                         [&weights](uint32_t a, uint32_t b) {
                             return weights[a] > weights[b];
                         });
    }

    // Hand out spare slots one at a time, cycling in weight order, until the
    // budget is exhausted or every layer is at its ceil. Weighted layers get
    // served first each cycle, so busier layers reach their ceil sooner.
    bool progress = true;
    while (spare > 0 && progress) {
        progress = false;
        for (uint32_t oi = 0; oi < order.size() && spare > 0; ++oi) {
            const uint32_t l = order[oi];
            if (cap_[l] < ceil) {
                ++cap_[l];
                --spare;
                progress = true;
            }
        }
    }
}

void SlotPool::set_layer_weights(const std::vector<uint64_t> &weights) {
    if (!stratified_) return;
    if (weights.size() != n_layer_) return;

    // Recompute capacities from the floor + the new weights. This may shrink a
    // layer's capacity below its current resident count; that is fine: already
    // resident experts stay resident and valid, and the next miss in that layer
    // evicts a same-layer victim as usual until resident <= cap. We never move
    // bytes or change provenance here, only the future eviction scope.
    recompute_caps(weights);
}

void SlotPool::touch(const ExpertId &id) {
    auto it = map_.find(id);
    if (it == map_.end()) return;
    if (!stratified_) {
        lru_.erase(it->second.second);
        lru_.push_front(id);
        it->second.second = lru_.begin();
        return;
    }
    const uint32_t L = id.layer < layers_.size() ? id.layer : 0;
    LayerState &ls = layers_[L];
    ls.lru.erase(it->second.second);
    ls.lru.push_front(id);
    it->second.second = ls.lru.begin();
}

} // namespace sf

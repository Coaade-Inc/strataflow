// Tiered Weight Store (TWS): the VRAM/RAM/SSD cache hierarchy for weight units
// (one expert = gate+up+down for one layer). See docs/PLAN.md section 3.3.
//
// Phase 1 implements the slot-pool bookkeeping, the expert-id -> slot-id remap,
// and LRU eviction, with an in-memory backing store so it is fully testable.
// Phase 3 swaps the backing store for the async-IO SSD path and real device
// buffers.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace sf {

// Globally identifies one weight unit: (layer, expert) within the model.
struct ExpertId {
    uint32_t layer;
    uint32_t expert;
    bool operator==(const ExpertId &o) const {
        return layer == o.layer && expert == o.expert;
    }
};

struct ExpertIdHash {
    size_t operator()(const ExpertId &e) const {
        return (static_cast<size_t>(e.layer) << 20) ^ e.expert;
    }
};

struct CacheStats {
    uint64_t hits   = 0;   // served from a resident slot
    uint64_t misses = 0;   // required a load from the backing store
    uint64_t evictions = 0;
    double   hit_rate() const {
        uint64_t total = hits + misses;
        return total ? double(hits) / double(total) : 0.0;
    }
};

// A fixed pool of equal-sized slots in one tier (VRAM or RAM). Experts are
// mapped to slot indices; a miss evicts the least-recently-used slot.
//
// LEVER B (FEAT-002): the MoE access pattern is a per-LAYER sweep every token
// (layer 0 routes its top-k, then layer 1, ... across n_layer layers), so a
// single GLOBAL LRU evicts a layer's hot experts via the intervening layers
// before the next token sweeps back - re-streaming almost every expert per
// token even with a large resident budget. The fix is LAYER-STRATIFIED
// residency: the total slot budget is DISTRIBUTED across the layers (a
// per-layer LRU scope and a per-layer capacity), so acquiring a {layer,expert}
// only ever evicts another expert OF THE SAME LAYER. Each layer then keeps a
// stable resident set across tokens, so a layer's hot experts survive the
// intervening layers' sweeps and the per-token working set hits cache.
//
// The per-layer capacities are derived PURELY from model shape + budget (no
// machine-specific constants): base = n_slots / n_layer, floored at
// n_expert_used (a single layer's top-k must always fit), with any remainder
// distributed by observed per-layer routing frequency (busier layers get the
// spare slots) via set_layer_weights(). When a layer's share >= n_expert it
// never evicts (its whole expert set stays resident); only a smaller layer's
// long tail streams.
//
// Fully-resident fast path: when the pool is constructed WITHOUT shape (the
// legacy 2-arg ctor) or with n_slots >= n_layer*n_expert, it behaves exactly as
// the original single global LRU (byte-identical, used by the auto-residency
// oracle path and the existing unit tests).
class SlotPool {
public:
    // Legacy / global-LRU pool (no layer stratification). Used by the
    // fully-resident fast path and the existing unit tests.
    SlotPool(uint32_t n_slots, uint64_t slot_bytes);

    // Layer-stratified pool. The total budget `n_slots` is partitioned across
    // `n_layer` layers; eviction is scoped per layer. `n_expert`/`n_expert_used`
    // size the per-layer capacities from shape. When n_slots >= n_layer*n_expert
    // the pool is effectively fully resident and never evicts (identical to the
    // global path).
    SlotPool(uint32_t n_slots, uint64_t slot_bytes, uint32_t n_layer,
             uint32_t n_expert, uint32_t n_expert_used);

    uint32_t n_slots() const { return n_slots_; }
    uint64_t slot_bytes() const { return slot_bytes_; }

    // Returns the slot index holding `id`, loading it (and possibly evicting)
    // if absent. `load_fn(dst)` fills the slot's buffer; it is only called on a
    // miss. Returns the slot index, or UINT32_MAX if the pool has zero slots.
    template <typename LoadFn>
    uint32_t acquire(const ExpertId &id, LoadFn &&load_fn);

    bool resident(const ExpertId &id) const { return map_.count(id) != 0; }
    void *slot_data(uint32_t slot) { return buffers_[slot].data(); }

    // True when acquiring a NOT-yet-resident id would NOT need to evict: either
    // the pool is not full or there is a reclaimed free slot. Const, read-only.
    // FEAT-003: lets a speculative async-prefetch install (on the compute
    // thread) decline when installing would evict a still-needed authoritative
    // resident expert; a prefetch then only ever fills otherwise-idle slots and
    // never amplifies disk traffic by evicting a resident the layer still uses.
    //
    // Layer-stratified pool: "free" means SOME layer is below its capacity, so
    // an acquire for that layer would not evict. For the common case a caller
    // cares about a specific layer; see has_free_slot_in_layer().
    bool has_free_slot() const {
        if (!stratified_) {
            return !free_slots_.empty() || map_.size() < n_slots_;
        }
        if (!free_slots_.empty()) return true;
        for (uint32_t l = 0; l < layers_.size(); ++l) {
            if (layers_[l].resident < cap_[l]) return true;
        }
        return false;
    }

    // Layer-stratified: true when acquiring a not-yet-resident expert in
    // `layer` would NOT evict (the layer is below its capacity and a slot is
    // free). On the global pool this reduces to has_free_slot().
    bool has_free_slot_in_layer(uint32_t layer) const {
        if (!stratified_) return has_free_slot();
        if (layer >= layers_.size()) return false;
        return !free_slots_.empty() && layers_[layer].resident < cap_[layer];
    }

    // Layer-stratified: refine the per-layer spare-slot distribution from
    // observed per-layer routing frequency (busier layers get the spare slots).
    // `weights[l]` is any nonnegative measure of layer l's routing activity
    // (e.g. total expert activations). Never drops a layer below its floor of
    // min(n_expert_used, n_expert); never changes the TOTAL budget. Already
    // resident experts stay resident; only future eviction scope changes. A
    // no-op on the global pool. Safe to call repeatedly as frequency sharpens.
    void set_layer_weights(const std::vector<uint64_t> &weights);

    // Layer-stratified: the capacity (max resident experts) currently assigned
    // to `layer`. On the global pool returns n_slots. Exposed for tests.
    uint32_t layer_capacity(uint32_t layer) const {
        if (!stratified_) return n_slots_;
        return layer < cap_.size() ? cap_[layer] : 0;
    }

    bool stratified() const { return stratified_; }

    const CacheStats &stats() const { return stats_; }

private:
    struct LayerState {
        std::list<ExpertId> lru;   // front = most recent, within this layer
        uint32_t resident = 0;     // experts of this layer currently resident
    };

    uint32_t n_slots_;
    uint64_t slot_bytes_;
    std::vector<std::vector<uint8_t>> buffers_;      // backing memory per slot
    // Global-pool LRU (stratified_ == false): one list across all layers.
    std::list<ExpertId> lru_;                        // front = most recent
    std::unordered_map<ExpertId, std::pair<uint32_t, std::list<ExpertId>::iterator>,
                       ExpertIdHash> map_;
    std::vector<uint32_t> free_slots_;
    CacheStats stats_;

    // Layer-stratified state (stratified_ == true).
    bool stratified_ = false;
    uint32_t n_layer_ = 0;
    uint32_t n_expert_ = 0;
    uint32_t n_expert_used_ = 0;
    std::vector<LayerState> layers_;   // per-layer LRU + resident count
    std::vector<uint32_t> cap_;        // per-layer capacity (sum <= n_slots_)

    void touch(const ExpertId &id);
    void recompute_caps(const std::vector<uint64_t> &weights);
};

template <typename LoadFn>
uint32_t SlotPool::acquire(const ExpertId &id, LoadFn &&load_fn) {
    if (n_slots_ == 0) return UINT32_MAX;

    auto it = map_.find(id);
    if (it != map_.end()) {
        ++stats_.hits;
        touch(id);
        return it->second.first;
    }

    ++stats_.misses;
    uint32_t slot;

    if (!stratified_) {
        if (!free_slots_.empty()) {
            slot = free_slots_.back();
            free_slots_.pop_back();
        } else {
            // Evict LRU (back of list).
            ExpertId victim = lru_.back();
            lru_.pop_back();
            auto vit = map_.find(victim);
            slot = vit->second.first;
            map_.erase(vit);
            ++stats_.evictions;
        }
        load_fn(buffers_[slot].data());
        lru_.push_front(id);
        map_[id] = {slot, lru_.begin()};
        return slot;
    }

    // Layer-stratified: eviction is scoped to this id's layer. The invariant
    // sum(cap_) <= n_slots_ guarantees a free slot exists whenever this layer
    // is below its capacity, so we only evict a SAME-LAYER resident once the
    // layer is at capacity.
    const uint32_t L = id.layer < layers_.size() ? id.layer : 0;
    LayerState &ls = layers_[L];
    if (ls.resident < cap_[L] && !free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
    } else if (!ls.lru.empty()) {
        // Evict this layer's own LRU victim (back of its own list) - the common
        // path: the layer is at capacity, so we recycle within the layer and
        // never touch another layer's resident set.
        ExpertId victim = ls.lru.back();
        ls.lru.pop_back();
        auto vit = map_.find(victim);
        slot = vit->second.first;
        map_.erase(vit);
        --ls.resident;
        ++stats_.evictions;
    } else if (!free_slots_.empty()) {
        // The layer is below its capacity but no slot is free because another
        // layer is transiently over its (just-shrunk) capacity. Take the free
        // slot if one exists.
        slot = free_slots_.back();
        free_slots_.pop_back();
    } else {
        // No free slot and this layer has no resident of its own to evict (it
        // was empty). This only happens transiently right after
        // set_layer_weights shrinks caps while another layer still over-holds
        // physical slots. Evict the LRU victim from the most over-capacity
        // layer so the budget invariant is restored without disturbing a layer
        // that is within its share.
        uint32_t donor = L;
        int64_t worst_over = -1;
        for (uint32_t l = 0; l < layers_.size(); ++l) {
            if (layers_[l].lru.empty()) continue;
            const int64_t over =
                static_cast<int64_t>(layers_[l].resident) -
                static_cast<int64_t>(cap_[l]);
            if (over > worst_over) {
                worst_over = over;
                donor = l;
            }
        }
        LayerState &dls = layers_[donor];
        ExpertId victim = dls.lru.back();
        dls.lru.pop_back();
        auto vit = map_.find(victim);
        slot = vit->second.first;
        map_.erase(vit);
        --dls.resident;
        ++stats_.evictions;
    }

    load_fn(buffers_[slot].data());
    ls.lru.push_front(id);
    map_[id] = {slot, ls.lru.begin()};
    ++ls.resident;
    return slot;
}

} // namespace sf

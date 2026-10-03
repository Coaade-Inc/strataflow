// Tiered Weight Store (TWS): the VRAM/RAM/SSD cache hierarchy for weight units
// (one expert = gate+up+down for one layer). See docs/PLAN.md section 3.3.
//
// Phase 1 implements the slot-pool bookkeeping, the expert-id -> slot-id remap,
// and LRU eviction, with an in-memory backing store so it is fully testable.
// Phase 3 swaps the backing store for the async-IO SSD path and real device
// buffers.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
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
class SlotPool {
public:
    SlotPool(uint32_t n_slots, uint64_t slot_bytes);

    uint32_t n_slots() const { return n_slots_; }
    uint64_t slot_bytes() const { return slot_bytes_; }

    // Returns the slot index holding `id`, loading it (and possibly evicting)
    // if absent. `load_fn(dst)` fills the slot's buffer; it is only called on a
    // miss. Returns the slot index, or UINT32_MAX if the pool has zero slots.
    template <typename LoadFn>
    uint32_t acquire(const ExpertId &id, LoadFn &&load_fn);

    bool resident(const ExpertId &id) const { return map_.count(id) != 0; }
    void *slot_data(uint32_t slot) { return buffers_[slot].data(); }

    const CacheStats &stats() const { return stats_; }

private:
    uint32_t n_slots_;
    uint64_t slot_bytes_;
    std::vector<std::vector<uint8_t>> buffers_;      // backing memory per slot
    std::list<ExpertId> lru_;                        // front = most recent
    std::unordered_map<ExpertId, std::pair<uint32_t, std::list<ExpertId>::iterator>,
                       ExpertIdHash> map_;
    std::vector<uint32_t> free_slots_;
    CacheStats stats_;

    void touch(const ExpertId &id);
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

} // namespace sf

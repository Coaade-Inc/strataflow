// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "tws/weight_store.h"

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

void SlotPool::touch(const ExpertId &id) {
    auto it = map_.find(id);
    if (it == map_.end()) return;
    lru_.erase(it->second.second);
    lru_.push_front(id);
    it->second.second = lru_.begin();
}

} // namespace sf

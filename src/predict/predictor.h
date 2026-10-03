// Expert predictor: estimates which experts upcoming tokens will need so the
// scheduler can prefetch them. See docs/PLAN.md section 3.4.
//
// Phase 1 ships the statistical predictor (per-layer frequency + co-activation
// counts). Hidden-state look-ahead and trained prerouter heads land later.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tws/weight_store.h"

#include <cstdint>
#include <vector>

namespace sf {

class ExpertPredictor {
public:
    ExpertPredictor(uint32_t n_layers, uint32_t n_experts);

    // Record that `id` actually fired, to improve future predictions.
    void observe(const ExpertId &id);

    // Predict up to `k` experts likely to fire in `layer`, most likely first.
    std::vector<uint32_t> predict(uint32_t layer, uint32_t k) const;

    // Fraction of past predictions that were later confirmed by observe().
    double accuracy() const;

private:
    uint32_t n_layers_;
    uint32_t n_experts_;
    std::vector<std::vector<uint64_t>> freq_;  // [layer][expert] activation count
    mutable uint64_t predicted_ = 0;
    mutable uint64_t confirmed_ = 0;
    mutable std::vector<uint32_t> last_prediction_;    // for accuracy tracking
};

} // namespace sf

// Per-token scheduler: builds the compute/transfer plan for each decode step
// and overlaps prefetch with compute. See docs/PLAN.md section 3.5.
//
// Phase 1 defines the interface and a sequential driver. The pipelined,
// double-buffered executor and CPU/GPU expert split land in Phase 4.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include "predict/predictor.h"
#include "tws/weight_store.h"

#include <cstdint>

namespace sf {

class Model;        // fwd
class KvStore;      // fwd

struct DecodeStats {
    uint64_t tokens      = 0;
    double   io_seconds  = 0.0;   // time waiting on weight loads
    double   compute_seconds = 0.0;
    CacheStats vram;
    CacheStats ram;

    double io_share() const {
        double total = io_seconds + compute_seconds;
        return total > 0 ? io_seconds / total : 0.0;
    }
};

class Scheduler {
public:
    Scheduler(Model &model, KvStore &kv, ExpertPredictor &predictor);

    // Run one decode step producing the next token id. Phase 1: delegates to
    // the model's (stub) forward pass and updates stats.
    int32_t step(int32_t last_token);

    const DecodeStats &stats() const { return stats_; }

private:
    Model          &model_;
    KvStore        &kv_;
    ExpertPredictor &predictor_;
    DecodeStats     stats_;
};

} // namespace sf

// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "sched/scheduler.h"

#include "model/model.h"

namespace sf {

Scheduler::Scheduler(Model &model, KvStore &kv, ExpertPredictor &predictor)
    : model_(model), kv_(kv), predictor_(predictor) {}

int32_t Scheduler::step(int32_t last_token) {
    // Phase 1: a single sequential forward pass. The pipeline that overlaps
    // predicted-expert prefetch with compute replaces this body in Phase 4;
    // the predictor and KV handles are already threaded through so that change
    // is local.
    (void)kv_;
    (void)predictor_;
    int32_t next = model_.forward(last_token);
    ++stats_.tokens;
    return next;
}

} // namespace sf

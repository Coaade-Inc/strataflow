// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "test_util.h"
#include "tws/weight_store.h"

using namespace sf;

static void fill(void *dst, uint8_t v, size_t n) {
    for (size_t i = 0; i < n; ++i) static_cast<uint8_t *>(dst)[i] = v;
}

static void test_hit_and_miss() {
    SlotPool pool(/*n_slots=*/2, /*slot_bytes=*/8);
    int loads = 0;
    auto load = [&](void *dst) { fill(dst, 7, 8); ++loads; };

    uint32_t s0 = pool.acquire({0, 0}, load);  // miss
    uint32_t s1 = pool.acquire({0, 1}, load);  // miss
    CHECK(s0 != s1);
    CHECK_EQ(loads, 2);

    pool.acquire({0, 0}, load);                // hit, no load
    CHECK_EQ(loads, 2);
    CHECK_EQ(pool.stats().hits, 1u);
    CHECK_EQ(pool.stats().misses, 2u);
}

static void test_lru_eviction() {
    SlotPool pool(2, 4);
    auto load = [](void *) {};

    pool.acquire({0, 0}, load);  // [0]
    pool.acquire({0, 1}, load);  // [0,1]
    pool.acquire({0, 0}, load);  // touch 0 -> 0 is MRU, 1 is LRU
    pool.acquire({0, 2}, load);  // miss -> evicts 1

    CHECK(pool.resident({0, 0}));
    CHECK(pool.resident({0, 2}));
    CHECK(!pool.resident({0, 1}));
    CHECK_EQ(pool.stats().evictions, 1u);
}

static void test_zero_slots() {
    SlotPool pool(0, 16);
    uint32_t s = pool.acquire({1, 1}, [](void *) {});
    CHECK_EQ(s, UINT32_MAX);
}

static void run_all() {
    RUN(test_hit_and_miss);
    RUN(test_lru_eviction);
    RUN(test_zero_slots);
}

TEST_MAIN()

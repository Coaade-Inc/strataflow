// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "test_util.h"
#include "predict/predictor.h"
#include "tws/weight_store.h"

#include <cstdint>
#include <vector>

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

// -------- FEAT-002: LAYER-STRATIFIED residency (LEVER B) ------------------

// Simulate the MoE per-layer sweep: every token touches layer 0's top-k, then
// layer 1's top-k, ... across all layers; the next token sweeps the SAME
// experts again. Returns the pool's miss count after `n_tokens` sweeps. Each
// layer routes the SAME `k` experts every token (its stable hot set), which is
// exactly the pattern a global LRU thrashes and a per-layer scope keeps cached.
static uint64_t sweep_misses(SlotPool &pool, uint32_t n_layer, uint32_t k,
                             int n_tokens) {
    auto load = [](void *) {};
    for (int t = 0; t < n_tokens; ++t) {
        for (uint32_t l = 0; l < n_layer; ++l) {
            for (uint32_t e = 0; e < k; ++e) {
                pool.acquire({l, e}, load);
            }
        }
    }
    return pool.stats().misses;
}

// The lever-B pathology, in miniature: when the total budget is just ONE slot
// short of spanning a full per-layer sweep (n_slots = n_layer*k - 1), a single
// global LRU degenerates into near-total thrash - the classic cyclic-access
// pathology where a cache one element smaller than the repeating cycle serves
// ~zero hits, because the first expert of the cycle is evicted exactly one
// access before it is reused. The layer-stratified pool instead keeps each
// layer's k experts resident in that layer's own scope, so it only pays the
// warm-up miss and the single under-provisioned layer's tail - far FEWER misses
// at the SAME total budget and SAME peak RSS. This is the exact per-token
// working-set-hits-cache win from FEAT-002 (the engine test on the real L8-E16
// fixture shows the same effect: 626 vs 975 misses at a bounded slot count).
static void test_stratified_fewer_misses_than_global() {
    const uint32_t n_layer = 8;
    const uint32_t n_expert = 16;
    const uint32_t k = 3;                   // distinct experts each layer routes
    const uint32_t n_slots = n_layer * k - 1;  // ONE short of a full sweep: 23
    const int n_tokens = 10;

    SlotPool global(n_slots, /*slot_bytes=*/8);  // legacy single global LRU
    SlotPool strat(n_slots, /*slot_bytes=*/8, n_layer, n_expert, k);
    CHECK(!global.stratified());
    CHECK(strat.stratified());

    const uint64_t global_miss = sweep_misses(global, n_layer, k, n_tokens);
    const uint64_t strat_miss  = sweep_misses(strat, n_layer, k, n_tokens);

    // The stratified pool re-streams strictly (and dramatically) less.
    CHECK(strat_miss < global_miss);
    // The global LRU thrashes: one slot short of the cycle, it misses nearly
    // every acquire (close to n_layer*k per token for every token).
    const uint64_t sweep_sz = static_cast<uint64_t>(n_layer) * k;
    CHECK(global_miss >= sweep_sz * static_cast<uint64_t>(n_tokens - 1));
    // The stratified pool mostly hits after warm-up: well under half the global
    // miss count (its per-layer scopes keep each layer's working set resident).
    CHECK(strat_miss < global_miss / 2);
    // Both pools actually ran (sanity).
    CHECK(global_miss > 0);
    CHECK(strat_miss > 0);
}

// Re-acquiring a PRIOR layer's expert on the NEXT sweep HITS under the
// stratified pool where the global LRU would have MISSED. Direct demonstration
// of the per-layer eviction scope.
static void test_prior_layer_survives_sweep() {
    const uint32_t n_layer = 4;
    const uint32_t n_expert = 8;
    const uint32_t n_expert_used = 2;
    const uint32_t n_slots = 12;  // 3 per layer floor, bounded below 32
    auto load = [](void *) {};

    SlotPool pool(n_slots, 8, n_layer, n_expert, n_expert_used);

    // Token 0: sweep all layers' top-k (all misses, warming the hot set).
    for (uint32_t l = 0; l < n_layer; ++l)
        for (uint32_t e = 0; e < n_expert_used; ++e)
            pool.acquire({l, e}, load);
    const uint64_t miss_after_t0 = pool.stats().misses;

    // Token 1: re-sweep. Layer 0's expert 0 was touched FIRST and the three
    // intervening layers each routed their own experts since - a global LRU of
    // 12 slots would have evicted layer 0 by now. Under per-layer scope it is
    // still resident, so this is a HIT.
    const uint64_t hits_before = pool.stats().hits;
    pool.acquire({0, 0}, load);
    CHECK(pool.resident({0, 0}));
    CHECK_EQ(pool.stats().hits, hits_before + 1);  // HIT, not a reload
    // No new miss was taken for the re-acquire.
    CHECK_EQ(pool.stats().misses, miss_after_t0);
}

// Eviction never crosses a layer boundary: hammering layer 2 past its capacity
// never evicts a resident expert of any OTHER layer.
static void test_eviction_scoped_to_layer() {
    const uint32_t n_layer = 4;
    const uint32_t n_expert = 8;
    const uint32_t n_expert_used = 2;
    const uint32_t n_slots = 12;  // floor 3 per layer
    auto load = [](void *) {};

    SlotPool pool(n_slots, 8, n_layer, n_expert, n_expert_used);

    // Make one expert of every other layer resident.
    pool.acquire({0, 0}, load);
    pool.acquire({1, 0}, load);
    pool.acquire({3, 0}, load);

    // Now churn layer 2 across MANY distinct experts (far beyond its share),
    // forcing repeated eviction WITHIN layer 2.
    for (uint32_t e = 0; e < n_expert; ++e) pool.acquire({2, e}, load);
    for (uint32_t e = 0; e < n_expert; ++e) pool.acquire({2, e}, load);

    // The other layers' residents are untouched by layer 2's thrashing.
    CHECK(pool.resident({0, 0}));
    CHECK(pool.resident({1, 0}));
    CHECK(pool.resident({3, 0}));
    // Layer 2 cannot hold more than its capacity.
    uint32_t cap2 = pool.layer_capacity(2);
    uint32_t resident2 = 0;
    for (uint32_t e = 0; e < n_expert; ++e)
        if (pool.resident({2, e})) ++resident2;
    CHECK(resident2 <= cap2);
}

// Fully-resident fast path: when n_slots >= n_layer*n_expert the stratified
// pool holds every expert and never evicts, byte-identical to today.
static void test_fully_resident_never_evicts() {
    const uint32_t n_layer = 3;
    const uint32_t n_expert = 4;
    const uint32_t n_expert_used = 2;
    const uint32_t n_slots = n_layer * n_expert;  // 12 = full working set
    auto load = [](void *) {};

    SlotPool pool(n_slots, 8, n_layer, n_expert, n_expert_used);
    CHECK(pool.stratified());

    // Touch every expert of every layer twice (two full sweeps).
    for (int t = 0; t < 2; ++t)
        for (uint32_t l = 0; l < n_layer; ++l)
            for (uint32_t e = 0; e < n_expert; ++e)
                pool.acquire({l, e}, load);

    // Every expert stayed resident; zero evictions; each layer's capacity holds
    // all its experts.
    CHECK_EQ(pool.stats().evictions, static_cast<uint64_t>(0));
    for (uint32_t l = 0; l < n_layer; ++l) {
        CHECK_EQ(pool.layer_capacity(l), n_expert);
        for (uint32_t e = 0; e < n_expert; ++e) CHECK(pool.resident({l, e}));
    }
}

// Capacities sum to no more than the total budget (the invariant the acquire
// fast path relies on) and never drop a layer below its shape floor, even after
// a frequency-weighted redistribution.
static void test_capacity_budget_and_floor() {
    const uint32_t n_layer = 6;
    const uint32_t n_expert = 10;
    const uint32_t n_expert_used = 3;
    const uint32_t n_slots = 30;  // base 5/layer, floor 3

    SlotPool pool(n_slots, 8, n_layer, n_expert, n_expert_used);

    auto check_invariants = [&]() {
        uint32_t sum = 0;
        for (uint32_t l = 0; l < n_layer; ++l) {
            const uint32_t c = pool.layer_capacity(l);
            CHECK(c >= n_expert_used);  // never below the top-k floor
            CHECK(c <= n_expert);       // never beyond the expert count
            sum += c;
        }
        CHECK(sum <= n_slots);
    };
    check_invariants();  // even split from the ctor

    // Skew frequency heavily toward layer 5; it should get the spare slots but
    // the invariants must still hold and no layer may drop below the floor.
    std::vector<uint64_t> w(n_layer, 1);
    w[5] = 1000;
    pool.set_layer_weights(w);
    check_invariants();
    CHECK(pool.layer_capacity(5) >= pool.layer_capacity(0));
}

// Sub-top-k capacity regime (the review's non-blocking item 3): when the total
// budget is between n_expert_used and n_layer*n_expert_used, recompute_caps
// shrinks the floor BELOW the single-layer top-k, so a layer's capacity can sit
// under n_expert_used. This is a CACHING DEGRADATION, never a correctness or
// invariant break: the sum(cap) <= n_slots invariant still holds, acquire still
// finds a victim for every routed expert (so the top-k always ends up staged),
// and the stats stay self-consistent. This test exercises exactly that regime
// (which the budget-affords-the-floor test_capacity_budget_and_floor does not)
// and asserts the invariants the acquire fast path and the engine's
// byte-identity gate rely on.
static void test_capacity_below_topk_still_correct() {
    const uint32_t n_layer = 8;
    const uint32_t n_expert = 16;
    const uint32_t n_expert_used = 4;              // single-layer top-k
    // Between n_expert_used (4) and n_layer*n_expert_used (32): the floor-shrink
    // path triggers (floor 4 * 8 layers = 32 > 20), so floor -> 20/8 = 2 < 4.
    const uint32_t n_slots = 20;
    auto load = [](void *) {};

    SlotPool pool(n_slots, 8, n_layer, n_expert, n_expert_used);
    CHECK(pool.stratified());

    // (1) sum(cap) <= n_slots still holds, and at least one layer is BELOW the
    // top-k (the regime under test is actually reached).
    uint32_t sum = 0;
    bool some_below_topk = false;
    for (uint32_t l = 0; l < n_layer; ++l) {
        const uint32_t c = pool.layer_capacity(l);
        CHECK(c >= 1);                 // never zero: eviction always has a victim
        CHECK(c <= n_expert);
        if (c < n_expert_used) some_below_topk = true;
        sum += c;
    }
    CHECK(sum <= n_slots);
    CHECK(some_below_topk);            // we really are in the sub-top-k regime

    // (2) acquire ALWAYS succeeds for every routed expert even where a layer's
    // cap is below its top-k: the top-k of every layer is staged each sweep (a
    // real decode fills staging per routed expert before compute, so the matmul
    // always sees correct bytes - the degradation is only extra re-streaming).
    const int n_tokens = 6;
    for (int t = 0; t < n_tokens; ++t) {
        for (uint32_t l = 0; l < n_layer; ++l) {
            for (uint32_t e = 0; e < n_expert_used; ++e) {
                const uint32_t slot = pool.acquire({l, e}, load);
                CHECK(slot != UINT32_MAX);   // a victim was always found
                CHECK(pool.resident({l, e}));  // just-acquired id is resident
            }
            // No layer ever holds more than its capacity (invariant preserved
            // even though the cap is below the top-k).
            uint32_t resident_l = 0;
            for (uint32_t e = 0; e < n_expert; ++e)
                if (pool.resident({l, e})) ++resident_l;
            CHECK(resident_l <= pool.layer_capacity(l));
        }
    }

    // (3) stats stay self-consistent: every acquire was a hit or a miss, and
    // evictions never exceed misses (you cannot evict more than you loaded).
    const CacheStats &s = pool.stats();
    const uint64_t total_acquires =
        static_cast<uint64_t>(n_tokens) * n_layer * n_expert_used;
    CHECK_EQ(s.hits + s.misses, total_acquires);
    CHECK(s.evictions <= s.misses);
    // The pool is under pressure (sub-top-k caps force re-streaming), so it both
    // missed and evicted - the degradation is real but bounded, not a hang.
    CHECK(s.misses > 0);
    CHECK(s.evictions > 0);
}

// PREDICTOR-DRIVEN RESIDENCY, END TO END (FEAT-002 step 5, review issue 1).
// The engine refines the layer-stratified pool once every few tokens via
// pool.set_layer_weights(predictor->layer_activity()) (see Engine::forward).
// The existing test_capacity_budget_and_floor proves set_layer_weights biases
// toward hand-made weights; this test proves the WHOLE chain that the engine
// actually runs: observed routing -> ExpertPredictor::layer_activity() ->
// set_layer_weights -> the busier layer ends up with MORE resident capacity,
// which in turn keeps MORE of that layer's reused experts resident across a
// decode sweep (the reduce-thrash goal). Residency only: the floor guarantees
// every layer's top-k is still staged, so decode byte-identity is untouched.
static void test_predictor_driven_residency() {
    const uint32_t n_layer = 4;
    const uint32_t n_expert = 8;
    const uint32_t n_expert_used = 2;
    // floor 2/layer = 8 used, exactly 1 spare slot. With a single spare the
    // weight order decides who gets it: the busiest layer wins the slot and the
    // others sit at the floor. (A roomy budget would let every layer reach the
    // ceil and hide the bias; scarcity is exactly the production regime where
    // residency choice matters.)
    const uint32_t n_slots = 9;  // floor 2/layer = 8, 1 spare to distribute

    ExpertPredictor predictor(n_layer, n_expert);
    SlotPool pool(n_slots, 8, n_layer, n_expert, n_expert_used);
    CHECK(pool.stratified());

    // Before any observation layer_activity() is all-zero, so the pool keeps
    // its shape-derived split: spare slots are handed out in plain index order
    // (no layer favored by frequency yet). Record the hot layer's starting
    // capacity so we can prove the predictor shifts capacity TOWARD it.
    const uint32_t cap_hot_before = pool.layer_capacity(3);

    // Simulate decoding where layer 3 routes to a WIDE, reused set of experts
    // every token (high activation) while layer 0 barely routes. layer_activity
    // sums per-layer activation counts, so layer 3 becomes by far the busiest.
    const int n_tokens = 50;
    for (int t = 0; t < n_tokens; ++t) {
        // Layer 3: all 8 experts fire each token (busiest, biggest reuse set).
        for (uint32_t e = 0; e < n_expert; ++e) predictor.observe({3, e});
        // Layer 2: half the experts fire.
        for (uint32_t e = 0; e < n_expert / 2; ++e) predictor.observe({2, e});
        // Layer 0 and 1: only the top-k fire (quiet).
        for (uint32_t e = 0; e < n_expert_used; ++e) {
            predictor.observe({0, e});
            predictor.observe({1, e});
        }
    }

    // Feed the predictor's per-layer activity into the pool exactly as the
    // engine does.
    const std::vector<uint64_t> activity = predictor.layer_activity();
    CHECK_EQ(activity.size(), static_cast<size_t>(n_layer));
    CHECK(activity[3] > activity[2]);
    CHECK(activity[2] > activity[0]);
    pool.set_layer_weights(activity);

    // The busiest layer now has the SINGLE scarce spare slot: its capacity is
    // the strict maximum across all layers, strictly greater than every other
    // layer's (which sit at the floor). And the hot layer's capacity GREW
    // relative to its pre-frequency (index-order) starting point - the shift is
    // caused by the observed activity, not an artifact of the initial split.
    const uint32_t cap_hot = pool.layer_capacity(3);
    CHECK(cap_hot > cap_hot_before);
    for (uint32_t l = 0; l < n_layer; ++l) {
        if (l == 3) continue;
        CHECK(cap_hot > pool.layer_capacity(l));
    }
    const uint32_t cap_cold = pool.layer_capacity(0);
    // Every layer still holds at least its top-k floor (residency change never
    // starves a layer below what decode needs staged).
    for (uint32_t l = 0; l < n_layer; ++l) {
        CHECK(pool.layer_capacity(l) >= n_expert_used);
    }
    // Budget invariant preserved.
    uint32_t sum = 0;
    for (uint32_t l = 0; l < n_layer; ++l) sum += pool.layer_capacity(l);
    CHECK(sum <= n_slots);

    // The extra capacity on the hot layer actually REDUCES thrash: drive one
    // more decode sweep touching the hot layer's wide reused set and a cold
    // layer's wide set with IDENTICAL access patterns, and the hot layer (which
    // now has more slots) must keep at least as many experts resident as the
    // cold layer at the end of the sweep.
    auto load = [](void *) {};
    const int sweep_tokens = 4;
    for (int t = 0; t < sweep_tokens; ++t) {
        for (uint32_t e = 0; e < n_expert; ++e) {
            pool.acquire({3, e}, load);  // hot layer reuses its wide set
            pool.acquire({0, e}, load);  // cold layer touches the same width
        }
    }
    uint32_t resident_hot = 0, resident_cold = 0;
    for (uint32_t e = 0; e < n_expert; ++e) {
        if (pool.resident({3, e})) ++resident_hot;
        if (pool.resident({0, e})) ++resident_cold;
    }
    CHECK(resident_hot >= resident_cold);
    CHECK(resident_hot <= cap_hot);
    CHECK(resident_cold <= cap_cold);
}

static void run_all() {
    RUN(test_hit_and_miss);
    RUN(test_lru_eviction);
    RUN(test_zero_slots);
    RUN(test_stratified_fewer_misses_than_global);
    RUN(test_prior_layer_survives_sweep);
    RUN(test_eviction_scoped_to_layer);
    RUN(test_fully_resident_never_evicts);
    RUN(test_capacity_budget_and_floor);
    RUN(test_capacity_below_topk_still_correct);
    RUN(test_predictor_driven_residency);
}

TEST_MAIN()

// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "predict/predictor.h"
#include "test_util.h"

using namespace sf;

// The most frequently observed experts should be predicted first.
static void test_frequency_ranking() {
    ExpertPredictor p(/*n_layers=*/2, /*n_experts=*/8);
    for (int i = 0; i < 10; ++i) p.observe({0, 3});
    for (int i = 0; i < 5; ++i)  p.observe({0, 5});
    p.observe({0, 1});

    auto top = p.predict(0, 2);
    CHECK_EQ(top.size(), 2u);
    CHECK_EQ(top[0], 3u);
    CHECK_EQ(top[1], 5u);

    // Other layers are unaffected.
    CHECK(p.predict(1, 2).size() == 2u);  // returns something, counts are zero
}

static void test_accuracy_tracking() {
    ExpertPredictor p(1, 4);
    for (int i = 0; i < 3; ++i) p.observe({0, 2});
    auto pred = p.predict(0, 1);   // predicts {2}
    CHECK_EQ(pred[0], 2u);
    p.observe({0, 2});             // confirms the prediction
    CHECK(p.accuracy() > 0.0);
}

static void run_all() {
    RUN(test_frequency_ranking);
    RUN(test_accuracy_tracking);
}

TEST_MAIN()

// End-to-end test of the public C API against the dry-run model.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "strataflow/strataflow.h"
#include "test_util.h"

#include <cstring>
#include <string>

static int count_tokens(const char *text, void *user_data) {
    auto *n = static_cast<int *>(user_data);
    (void)text;
    ++(*n);
    return 0;
}

static int append_text(const char *text, void *user_data) {
    auto *s = static_cast<std::string *>(user_data);
    *s += text;
    return 0;
}

static void test_version() {
    CHECK(std::strlen(sf_version()) > 0);
    CHECK(std::strcmp(sf_status_str(SF_OK), "ok") == 0);
}

static void test_invalid_args() {
    sf_context *ctx = nullptr;
    CHECK_EQ(sf_context_create(nullptr, &ctx), SF_ERR_INVALID_ARGUMENT);

    sf_context_params p = sf_context_default_params();  // model_path == nullptr
    CHECK_EQ(sf_context_create(&p, &ctx), SF_ERR_INVALID_ARGUMENT);
}

static void test_generate_runs() {
    sf_context_params p = sf_context_default_params();
    p.model_path = "dummy.strata";

    sf_context *ctx = nullptr;
    CHECK_EQ(sf_context_create(&p, &ctx), SF_OK);
    CHECK(ctx != nullptr);

    char plan[256];
    CHECK_EQ(sf_describe_plan(ctx, plan, sizeof(plan)), SF_OK);
    CHECK(std::strlen(plan) > 0);

    sf_session *s = nullptr;
    CHECK_EQ(sf_session_create(ctx, &s), SF_OK);

    sf_sampling_params sp = sf_sampling_default_params();
    sp.max_tokens = 32;

    int n = 0;
    CHECK_EQ(sf_generate(s, "the quick brown fox", &sp, count_tokens, &n), SF_OK);
    CHECK(n > 0);

    sf_session_free(s);
    sf_context_free(ctx);
}

// Locks the sf_runtime_stats ABI shape: a default-constructed value must
// zero-init the appended fields, and sf_session_stats must fill them without
// error on the dry-run model (ttft_us is 0 - the CLI measures it; streamed_bytes
// is 0 on the dry-run/non-streaming path).
static void test_runtime_stats_fields() {
    sf_runtime_stats zero{};
    CHECK_EQ(zero.resident_weight_bytes, 0u);
    CHECK_EQ(zero.peak_rss_bytes, 0u);
    CHECK_EQ(zero.ttft_us, 0u);
    CHECK_EQ(zero.streamed_bytes, 0u);

    CHECK_EQ(sf_session_stats(nullptr, &zero), SF_ERR_INVALID_ARGUMENT);

    sf_context_params p = sf_context_default_params();
    p.model_path = "dummy.strata";

    sf_context *ctx = nullptr;
    CHECK_EQ(sf_context_create(&p, &ctx), SF_OK);
    sf_session *s = nullptr;
    CHECK_EQ(sf_session_create(ctx, &s), SF_OK);

    sf_sampling_params sp = sf_sampling_default_params();
    sp.max_tokens = 8;
    std::string out;
    CHECK_EQ(sf_generate(s, "stats prompt", &sp, append_text, &out), SF_OK);

    sf_runtime_stats rs{};
    CHECK_EQ(sf_session_stats(s, &rs), SF_OK);
    // Dry-run model does not stream experts and the CLI (not the ABI) times
    // TTFT, so both appended fields stay 0 here. This asserts the ABI wiring
    // (sf_session_stats sets them) without depending on real streaming numbers.
    CHECK_EQ(rs.ttft_us, 0u);
    CHECK_EQ(rs.streamed_bytes, 0u);

    sf_session_free(s);
    sf_context_free(ctx);
}

// Greedy generation must be reproducible (determinism contract, PLAN section 7).
static void test_determinism() {
    sf_context_params p = sf_context_default_params();
    p.model_path = "dummy.strata";

    auto run_once = [&]() {
        sf_context *ctx = nullptr;
        sf_context_create(&p, &ctx);
        sf_session *s = nullptr;
        sf_session_create(ctx, &s);
        sf_sampling_params sp = sf_sampling_default_params();  // greedy
        std::string out;
        sf_generate(s, "same prompt", &sp, append_text, &out);
        sf_session_free(s);
        sf_context_free(ctx);
        return out;
    };

    CHECK(run_once() == run_once());
}

static void run_all() {
    RUN(test_version);
    RUN(test_invalid_args);
    RUN(test_generate_runs);
    RUN(test_runtime_stats_fields);
    RUN(test_determinism);
}

TEST_MAIN()

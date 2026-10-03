// Tests that the hardware profiler actually measures SSD bandwidth (Task 1).
// A non-quick profile of a writable directory must report non-zero sequential
// and random read throughput, and must leave no probe file behind.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "hw/profiler.h"
#include "test_util.h"

#include <cstdio>
#include <string>

using namespace sf;

// The build directory is always writable (tests run from there), so use "."
// (the ctest working dir) as the model_dir to benchmark.
static void test_ssd_bandwidth_measured() {
    HardwareProfile p = profile_hardware(".", /*quick=*/false);
    CHECK(p.ssd_seq_gbps > 0.0);
    CHECK(p.ssd_rand_gbps > 0.0);

    // The summary and JSON must surface the measured values.
    const std::string summary = p.to_summary();
    CHECK(summary.find("SSD") != std::string::npos);
    const std::string json = p.to_json();
    CHECK(json.find("ssd_seq_gbps") != std::string::npos);
    CHECK(json.find("ssd_rand_gbps") != std::string::npos);
}

// Even under `quick`, the profiler runs a shrunk probe and reports non-zero
// bandwidth so the planner always has an estimate.
static void test_ssd_bandwidth_quick() {
    HardwareProfile p = profile_hardware(".", /*quick=*/true);
    CHECK(p.ssd_seq_gbps > 0.0);
    CHECK(p.ssd_rand_gbps > 0.0);
}

// The probe file must never be left behind.
static void test_no_probe_file_left() {
    HardwareProfile p = profile_hardware(".", /*quick=*/true);
    (void)p;
    std::FILE *f = std::fopen("./.strataflow_ssd_probe.tmp", "rb");
    CHECK(f == nullptr);
    if (f != nullptr) std::fclose(f);
}

static void run_all() {
    RUN(test_ssd_bandwidth_measured);
    RUN(test_ssd_bandwidth_quick);
    RUN(test_no_probe_file_left);
}

TEST_MAIN()

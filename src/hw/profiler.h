// Hardware profiler: measures the machine so the planner can place weights.
// See docs/PLAN.md section 3.1.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sf {

struct GpuInfo {
    std::string name;
    uint64_t    vram_bytes = 0;
    double      vram_gbps  = 0.0;   // measured/estimated device bandwidth
    int         backend    = 0;     // sf_backend value
};

struct HardwareProfile {
    // CPU
    unsigned      logical_cores   = 0;
    unsigned      physical_cores  = 0;
    bool          has_avx2        = false;
    bool          has_avx512      = false;
    bool          has_neon        = false;

    // Memory
    uint64_t      ram_total_bytes = 0;
    uint64_t      ram_free_bytes  = 0;
    double        ram_gbps        = 0.0;   // measured host read bandwidth

    // Storage (where the model lives)
    double        ssd_seq_gbps    = 0.0;   // sequential read
    double        ssd_rand_gbps   = 0.0;   // random 4 MiB read, QD1

    std::vector<GpuInfo> gpus;

    std::string to_json() const;
    std::string to_summary() const;
};

// Profiles the current machine. `model_dir` (optional) is used to benchmark the
// disk the model actually sits on; empty uses the current working directory.
// `quick` skips the longer bandwidth micro-benchmarks (used in tests/CI).
HardwareProfile profile_hardware(const std::string &model_dir = {}, bool quick = false);

// Returns the directory component of a path: everything up to (but not
// including) the last separator ('/' on all platforms, also '\\' on Windows).
// A path with no separator (a bare filename) returns "" (the current working
// directory). Used to turn a model FILE path into the directory to SSD-probe.
std::string containing_dir(const std::string &path);

} // namespace sf

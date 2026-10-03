// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "hw/profiler.h"

#include "common/log.h"

#include <chrono>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

#if defined(STRATAFLOW_PLATFORM_linux)
#  include <fstream>
#  include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  define SF_ARCH_X86 1
#  if defined(__GNUC__)
#    include <cpuid.h>
#  endif
#endif

#if defined(__aarch64__) || defined(__arm__)
#  define SF_ARCH_ARM 1
#endif

namespace sf {
namespace {

void detect_isa(HardwareProfile &p) {
#if defined(SF_ARCH_X86) && defined(__GNUC__)
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        p.has_avx2   = (ebx & (1u << 5))  != 0;   // AVX2
        p.has_avx512 = (ebx & (1u << 16)) != 0;   // AVX-512F
    }
#elif defined(SF_ARCH_ARM)
    p.has_neon = true;  // baseline on aarch64
#endif
}

void detect_memory(HardwareProfile &p) {
#if defined(STRATAFLOW_PLATFORM_linux)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t value_kb = 0;
    std::string unit;
    while (meminfo >> key >> value_kb >> unit) {
        if (key == "MemTotal:")     p.ram_total_bytes = value_kb * 1024ull;
        else if (key == "MemAvailable:") p.ram_free_bytes = value_kb * 1024ull;
    }
#endif
    if (p.ram_free_bytes == 0) {
        p.ram_free_bytes = p.ram_total_bytes;  // best effort on other OSes (Phase 1)
    }
}

// A crude but honest host-memory read-bandwidth probe: sum over a large buffer.
double measure_ram_gbps() {
    constexpr size_t kBytes = 128ull * 1024 * 1024;  // 128 MiB
    std::vector<uint8_t> buf(kBytes, 1);
    uint64_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 4; ++rep) {
        uint64_t acc = 0;
        for (size_t i = 0; i < kBytes; i += 64) {  // one touch per cache line
            acc += buf[i];
        }
        sink += acc;
    }
    auto t1 = std::chrono::steady_clock::now();
    // Keep `sink` observable so the read loop is not optimized away (portable).
    static volatile uint64_t keep_alive;
    keep_alive = sink;
    (void)keep_alive;
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double gib  = (double(kBytes) * 4.0) / (1024.0 * 1024.0 * 1024.0);
    return secs > 0 ? gib / secs : 0.0;
}

} // namespace

HardwareProfile profile_hardware(const std::string &model_dir, bool quick) {
    (void)model_dir;  // SSD benchmark lands with the async-IO layer (Phase 3)
    HardwareProfile p;

    p.logical_cores = std::thread::hardware_concurrency();
    p.physical_cores = p.logical_cores;  // refined with topology parsing later
    detect_isa(p);
    detect_memory(p);

    if (!quick) {
        p.ram_gbps = measure_ram_gbps();
    }

    // GPU discovery is delegated to the ggml backends in Phase 1b; none yet.

    log_info("hardware profile: " + p.to_summary());
    return p;
}

std::string HardwareProfile::to_summary() const {
    std::ostringstream os;
    os << logical_cores << " cores";
    if (has_avx512) os << " +avx512";
    else if (has_avx2) os << " +avx2";
    else if (has_neon) os << " +neon";
    os << ", RAM "
       << (ram_total_bytes / (1024ull * 1024 * 1024)) << " GiB total";
    if (ram_gbps > 0) os << " (~" << int(ram_gbps) << " GiB/s read)";
    if (gpus.empty()) os << ", no GPU";
    else os << ", " << gpus.size() << " GPU(s)";
    return os.str();
}

std::string HardwareProfile::to_json() const {
    std::ostringstream os;
    os << "{\n"
       << "  \"logical_cores\": " << logical_cores << ",\n"
       << "  \"has_avx2\": "   << (has_avx2 ? "true" : "false") << ",\n"
       << "  \"has_avx512\": " << (has_avx512 ? "true" : "false") << ",\n"
       << "  \"has_neon\": "   << (has_neon ? "true" : "false") << ",\n"
       << "  \"ram_total_bytes\": " << ram_total_bytes << ",\n"
       << "  \"ram_free_bytes\": "  << ram_free_bytes << ",\n"
       << "  \"ram_gbps\": " << ram_gbps << ",\n"
       << "  \"gpus\": " << gpus.size() << "\n"
       << "}";
    return os.str();
}

} // namespace sf

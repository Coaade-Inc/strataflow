// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "hw/profiler.h"

#include "common/log.h"
#include "hw/gpu_discovery.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(STRATAFLOW_PLATFORM_linux)
#  include <fstream>
#  include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define SF_ARCH_X86 1
#  if defined(__GNUC__) || defined(__clang__)
#    include <cpuid.h>
#  elif defined(_MSC_VER)
#    include <intrin.h>
#  endif
#endif

#if defined(__aarch64__) || defined(__arm__)
#  define SF_ARCH_ARM 1
#endif

namespace sf {
namespace {

void detect_isa(HardwareProfile &p) {
#if defined(SF_ARCH_X86) && (defined(__GNUC__) || defined(__clang__))
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        p.has_avx2   = (ebx & (1u << 5))  != 0;   // AVX2
        p.has_avx512 = (ebx & (1u << 16)) != 0;   // AVX-512F
    }
#elif defined(SF_ARCH_X86) && defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, 7, 0);
    const unsigned ebx = static_cast<unsigned>(regs[1]);
    p.has_avx2   = (ebx & (1u << 5))  != 0;
    p.has_avx512 = (ebx & (1u << 16)) != 0;
#elif defined(SF_ARCH_ARM)
    p.has_neon = true;  // baseline on aarch64
#else
    (void)p;  // no ISA probe on this target
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

// Build a probe-file path inside `dir` (the directory the model lives on, so we
// benchmark the right disk). Empty `dir` means the current working directory.
std::string probe_path(const std::string &dir) {
    std::string base = dir;
    if (!base.empty()) {
        const char c = base.back();
        if (c != '/' && c != '\\') base.push_back('/');
    }
    base += ".strataflow_ssd_probe.tmp";
    return base;
}

// Open `path` for binary write, create+truncate. Returns nullptr on failure
// (e.g. a read-only model dir), so the caller can fall back.
FILE *open_probe_write(const std::string &path) {
#if defined(STRATAFLOW_PLATFORM_windows)
    FILE *f = nullptr;
    if (fopen_s(&f, path.c_str(), "wb") != 0) return nullptr;
    return f;
#else
    return std::fopen(path.c_str(), "wb");
#endif
}

FILE *open_probe_read(const std::string &path) {
#if defined(STRATAFLOW_PLATFORM_windows)
    FILE *f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") != 0) return nullptr;
    return f;
#else
    return std::fopen(path.c_str(), "rb");
#endif
}

// Measure sequential and random read throughput of the disk `model_dir` sits
// on, in GiB/s, by writing a bounded probe file, reading it back sequentially
// and in random 4 MiB chunks, then deleting it. Portable (buffered stdio),
// bounded, and self-cleaning: it never leaves the probe file behind and reports
// 0 for a tier it cannot write (e.g. a read-only directory, with no temp
// fallback available). `quick` shrinks the probe for CI.
//
// NOTE: this goes through the OS page cache (direct I/O is Task 4), so on a
// warm cache the numbers reflect cached-read bandwidth. That is acceptable for
// the planner's coarse tiering decision; Task 4's O_DIRECT path measures true
// device bandwidth.
void measure_ssd_bandwidth(const std::string &model_dir, bool quick,
                           double &seq_gbps, double &rand_gbps) {
    seq_gbps = 0.0;
    rand_gbps = 0.0;

    const size_t chunk = 4ull * 1024 * 1024;              // 4 MiB random chunk
    const size_t total = quick ? (16ull * 1024 * 1024)    // 16 MiB in quick
                               : (64ull * 1024 * 1024);   // 64 MiB otherwise
    const int    rand_reads = quick ? 4 : 16;

    const std::string path = probe_path(model_dir);

    // --- write the probe file -------------------------------------------
    FILE *wf = open_probe_write(path);
    if (wf == nullptr) {
        log_warn("profiler: cannot write SSD probe to '" + path +
                 "'; reporting 0 bandwidth.");
        return;
    }
    std::vector<uint8_t> block(chunk);
    for (size_t i = 0; i < chunk; ++i) {
        block[i] = static_cast<uint8_t>(i * 1103515245u + 12345u);
    }
    size_t written = 0;
    bool write_ok = true;
    while (written < total) {
        const size_t n = std::min(chunk, total - written);
        if (std::fwrite(block.data(), 1, n, wf) != n) { write_ok = false; break; }
        written += n;
    }
    std::fflush(wf);
    std::fclose(wf);
    if (!write_ok || written == 0) {
        std::remove(path.c_str());
        return;
    }

    // --- sequential read ------------------------------------------------
    FILE *rf = open_probe_read(path);
    if (rf == nullptr) {
        std::remove(path.c_str());
        return;
    }
    std::vector<uint8_t> buf(chunk);
    uint64_t sink = 0;
    {
        auto t0 = std::chrono::steady_clock::now();
        size_t got_total = 0;
        for (;;) {
            const size_t got = std::fread(buf.data(), 1, chunk, rf);
            if (got == 0) break;
            sink += buf[0];
            got_total += got;
        }
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double gib = double(got_total) / (1024.0 * 1024.0 * 1024.0);
        if (secs > 0) seq_gbps = gib / secs;
    }

    // --- random 4 MiB reads (QD1) --------------------------------------
    {
        const size_t n_chunks = written / chunk;
        std::mt19937_64 rng(0xC0FFEEu);
        auto t0 = std::chrono::steady_clock::now();
        uint64_t got_total = 0;
        for (int r = 0; r < rand_reads; ++r) {
            const uint64_t idx =
                n_chunks ? (rng() % n_chunks) : 0;
            const uint64_t off = idx * chunk;
#if defined(STRATAFLOW_PLATFORM_windows)
            if (_fseeki64(rf, static_cast<long long>(off), SEEK_SET) != 0) break;
#else
            if (std::fseek(rf, static_cast<long>(off), SEEK_SET) != 0) break;
#endif
            const size_t got = std::fread(buf.data(), 1, chunk, rf);
            if (got == 0) break;
            sink += buf[0];
            got_total += got;
        }
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const double gib = double(got_total) / (1024.0 * 1024.0 * 1024.0);
        if (secs > 0) rand_gbps = gib / secs;
    }

    std::fclose(rf);
    std::remove(path.c_str());

    // Keep `sink` observable so the read loops are not optimized away.
    static volatile uint64_t keep_alive;
    keep_alive = sink;
    (void)keep_alive;
}

} // namespace

std::string containing_dir(const std::string &path) {
    if (path.empty()) return {};
    // Scan for the last path separator. On Windows both '/' and '\\' separate
    // components; on POSIX only '/'. A bare filename (no separator) maps to the
    // current working directory ("").
    std::string::size_type sep = path.find_last_of('/');
#if defined(STRATAFLOW_PLATFORM_windows)
    const std::string::size_type bsep = path.find_last_of('\\');
    if (bsep != std::string::npos &&
        (sep == std::string::npos || bsep > sep)) {
        sep = bsep;
    }
#endif
    if (sep == std::string::npos) return {};
    return path.substr(0, sep);
}

HardwareProfile profile_hardware(const std::string &model_dir, bool quick) {
    HardwareProfile p;

    p.logical_cores = std::thread::hardware_concurrency();
    p.physical_cores = p.logical_cores;  // refined with topology parsing later
    detect_isa(p);
    detect_memory(p);

    if (!quick) {
        p.ram_gbps = measure_ram_gbps();
    }

    // Measure the disk the model lives on. Even under `quick` we run a shrunk
    // probe so the planner always has a non-zero SSD estimate; it is bounded
    // and self-cleaning (see measure_ssd_bandwidth).
    measure_ssd_bandwidth(model_dir, quick, p.ssd_seq_gbps, p.ssd_rand_gbps);

    // GPU discovery is delegated to the ggml backends (Phase 2). The thin
    // bridge in model/ggml_model.cpp enumerates GGML_BACKEND_DEVICE_TYPE_GPU
    // devices and fills p.gpus; on a CPU-only machine (CI) this finds none.
    enumerate_gpus(p.gpus);

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
    if (ssd_seq_gbps > 0 || ssd_rand_gbps > 0) {
        os << ", SSD " << ssd_seq_gbps << "/" << ssd_rand_gbps
           << " GiB/s seq/rand";
    }
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
       << "  \"ssd_seq_gbps\": " << ssd_seq_gbps << ",\n"
       << "  \"ssd_rand_gbps\": " << ssd_rand_gbps << ",\n"
       << "  \"gpus\": " << gpus.size() << "\n"
       << "}";
    return os.str();
}

} // namespace sf

// Byte-correctness tests for the native direct-I/O BlockFile backends
// (docs/PHASE3_PLAN.md Task 4). We write a deterministic temp file UNDER THE
// WORKSPACE (the test's current working directory inside build/, never /tmp —
// the sandbox has split /tmp mounts), then read it back both 4 KiB-aligned and
// deliberately UNALIGNED via read_at and read_async. The aligned/unaligned mix
// exercises the aligned bounce-buffer path that direct I/O (O_DIRECT /
// FILE_FLAG_NO_BUFFERING / F_NOCACHE) requires.
//
// We assert byte-exact correctness regardless of which backend is active:
// direct I/O may be rejected inside a CI sandbox and fall back to sync, and
// both outcomes are correct. We only assert backend_name() is non-empty, never
// a specific backend, and never throughput.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "test_util.h"
#include "tws/async_io.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <process.h>
#else
#  include <unistd.h>
#endif

using namespace sf;

namespace {

// Deterministic byte pattern: cheap, non-repeating within a page, reproducible.
unsigned char pattern_byte(uint64_t i) {
    return static_cast<unsigned char>((i * 2654435761ull) >> 24);
}

const size_t kFileSize = 5 * kIoAlignment + 777; // not a multiple of 4 KiB

// A temp path in the current working directory (build tree, inside the
// workspace). Unique per process so parallel ctest runs don't collide.
std::string temp_path() {
    return "sf_async_io_test_" + std::to_string(
#if defined(_WIN32)
                                     static_cast<unsigned long>(_getpid())
#else
                                     static_cast<unsigned long>(::getpid())
#endif
                                     ) +
           ".bin";
}

bool write_pattern_file(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    std::vector<unsigned char> buf(kFileSize);
    for (uint64_t i = 0; i < kFileSize; ++i) buf[i] = pattern_byte(i);
    size_t n = std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    return n == buf.size();
}

// Verify that `len` bytes read at `offset` match the known pattern.
void check_range(const BlockFile &bf, uint64_t offset, size_t len) {
    std::vector<unsigned char> got(len, 0xCC);
    int64_t n = bf.read_at(got.data(), len, offset);
    CHECK(n == static_cast<int64_t>(len));
    bool match = true;
    for (size_t i = 0; i < len; ++i) {
        if (got[i] != pattern_byte(offset + i)) { match = false; break; }
    }
    CHECK(match);
}

void check_range_async(const BlockFile &bf, uint64_t offset, size_t len) {
    std::vector<unsigned char> got(len, 0xCC);
    std::future<int64_t> fut = bf.read_async(got.data(), len, offset);
    int64_t n = fut.get();
    CHECK(n == static_cast<int64_t>(len));
    bool match = true;
    for (size_t i = 0; i < len; ++i) {
        if (got[i] != pattern_byte(offset + i)) { match = false; break; }
    }
    CHECK(match);
}

} // namespace

static void test_open_and_size() {
    const std::string path = temp_path();
    CHECK(write_pattern_file(path));

    BlockFile bf;
    CHECK(bf.open(path));
    CHECK(bf.is_open());
    CHECK_EQ(bf.size(), static_cast<uint64_t>(kFileSize));

    // backend_name()/active_backend_name() must report a non-empty active
    // backend. We do NOT assert WHICH backend: direct I/O may fall back to sync.
    CHECK(std::string(BlockFile::backend_name()).size() > 0);
    CHECK(std::string(bf.active_backend_name()).size() > 0);

    bf.close();
    CHECK(!bf.is_open());
    std::remove(path.c_str());
}

static void test_read_aligned() {
    const std::string path = temp_path();
    CHECK(write_pattern_file(path));

    BlockFile bf;
    CHECK(bf.open(path));

    // 4 KiB-aligned offsets and lengths.
    check_range(bf, 0, kIoAlignment);
    check_range(bf, kIoAlignment, kIoAlignment);
    check_range(bf, 2 * kIoAlignment, 2 * kIoAlignment);
    check_range(bf, 0, kIoAlignment); // repeat offset 0, full page

    bf.close();
    std::remove(path.c_str());
}

static void test_read_unaligned() {
    const std::string path = temp_path();
    CHECK(write_pattern_file(path));

    BlockFile bf;
    CHECK(bf.open(path));

    // Offsets that are NOT multiples of 4096 and lengths that are not either.
    // These stress the aligned bounce-buffer superset read + copy-out path.
    check_range(bf, 1, 100);
    check_range(bf, 123, 456);
    check_range(bf, kIoAlignment - 1, kIoAlignment + 3);
    check_range(bf, 2 * kIoAlignment + 7, 1000);
    check_range(bf, 17, 1);                 // single byte, odd offset
    check_range(bf, 4095, 4098);            // straddles two pages

    // Read right up to EOF with an unaligned tail.
    const uint64_t near_end_off = kFileSize - 300;
    check_range(bf, near_end_off, 300);

    bf.close();
    std::remove(path.c_str());
}

static void test_read_async_matches() {
    const std::string path = temp_path();
    CHECK(write_pattern_file(path));

    BlockFile bf;
    CHECK(bf.open(path));

    check_range_async(bf, 0, kIoAlignment);          // aligned
    check_range_async(bf, 55, 900);                  // unaligned
    check_range_async(bf, kIoAlignment + 13, 2050);  // unaligned, spanning pages
    check_range_async(bf, kFileSize - 128, 128);     // up to EOF

    bf.close();
    std::remove(path.c_str());
}

// Concurrent-read stress: multiple threads call read_at() on the SAME
// BlockFile handle at once. This is exactly the scenario the engine creates
// (the async prefetch worker and the compute thread both reach
// BlockFile::read_at via one shared handle). It runs on EVERY CI platform with
// no fixture/env dependency, so on Windows it exercises the positional,
// stateless, lock-free ReadFile path under contention: a shared-position
// regression would return wrong bytes (or hang) and fail here. Kept small so
// it adds negligible CI time (4 threads x 200 reads of short ranges).
static void test_concurrent_reads() {
    const std::string path = temp_path();
    CHECK(write_pattern_file(path));

    BlockFile bf;
    CHECK(bf.open(path));

    // A mix of aligned and unaligned (offset, len) ranges spanning the file,
    // reusing the offsets exercised by the single-threaded tests above.
    struct Range {
        uint64_t offset;
        size_t len;
    };
    const Range ranges[] = {
        {0, kIoAlignment},
        {1, 100},
        {123, 456},
        {kIoAlignment - 1, kIoAlignment + 3},
        {2 * kIoAlignment + 7, 1000},
        {17, 1},
        {4095, 4098},
        {kFileSize - 300, 300}, // near EOF
    };
    const size_t kNumRanges = sizeof(ranges) / sizeof(ranges[0]);

    const int kThreads = 4;
    const int kItersPerThread = 200;
    std::atomic<bool> ok{true};

    auto worker = [&](int tid) {
        std::vector<unsigned char> got;
        for (int it = 0; it < kItersPerThread; ++it) {
            const size_t idx =
                (static_cast<size_t>(tid) + static_cast<size_t>(it)) % kNumRanges;
            const Range &r = ranges[idx];
            got.assign(r.len, 0xCC);
            int64_t n = bf.read_at(got.data(), r.len, r.offset);
            if (n != static_cast<int64_t>(r.len)) {
                ok.store(false);
                return;
            }
            for (size_t i = 0; i < r.len; ++i) {
                if (got[i] != pattern_byte(r.offset + i)) {
                    ok.store(false);
                    return;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
    for (auto &th : threads) th.join();

    CHECK(ok.load());

    bf.close();
    std::remove(path.c_str());
}

// Fire several read_async() futures concurrently from the main thread, then
// collect and verify each. This confirms the std::async wrapper over the
// positional read_at is also correct when multiple reads overlap in flight.
static void test_concurrent_read_async() {
    const std::string path = temp_path();
    CHECK(write_pattern_file(path));

    BlockFile bf;
    CHECK(bf.open(path));

    struct Range {
        uint64_t offset;
        size_t len;
    };
    const Range ranges[] = {
        {0, kIoAlignment},
        {55, 900},
        {kIoAlignment + 13, 2050},
        {2 * kIoAlignment + 7, 1000},
        {kFileSize - 128, 128},
    };
    const size_t kNumRanges = sizeof(ranges) / sizeof(ranges[0]);

    std::vector<std::vector<unsigned char>> bufs(kNumRanges);
    std::vector<std::future<int64_t>> futs;
    futs.reserve(kNumRanges);
    for (size_t i = 0; i < kNumRanges; ++i) {
        bufs[i].assign(ranges[i].len, 0xCC);
        futs.push_back(bf.read_async(bufs[i].data(), ranges[i].len, ranges[i].offset));
    }

    for (size_t i = 0; i < kNumRanges; ++i) {
        int64_t n = futs[i].get();
        CHECK(n == static_cast<int64_t>(ranges[i].len));
        bool match = true;
        for (size_t j = 0; j < ranges[i].len; ++j) {
            if (bufs[i][j] != pattern_byte(ranges[i].offset + j)) {
                match = false;
                break;
            }
        }
        CHECK(match);
    }

    bf.close();
    std::remove(path.c_str());
}

static void test_read_closed_is_error() {
    BlockFile bf;
    unsigned char b = 0;
    CHECK(bf.read_at(&b, 1, 0) == -1);
    CHECK(!bf.is_open());
}

static void run_all() {
    RUN(test_open_and_size);
    RUN(test_read_aligned);
    RUN(test_read_unaligned);
    RUN(test_read_async_matches);
    RUN(test_concurrent_reads);
    RUN(test_concurrent_read_async);
    RUN(test_read_closed_is_error);
}

TEST_MAIN()

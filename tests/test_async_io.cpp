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

#include <cstdint>
#include <cstdio>
#include <string>
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
    RUN(test_read_closed_is_error);
}

TEST_MAIN()

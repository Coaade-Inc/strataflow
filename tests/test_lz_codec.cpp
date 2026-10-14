// Unit gate for the self-contained LZ codec (src/common/lz_codec). This is the
// FEAT-001 foundation for the "compress-what-streams" lever: the pack-time
// compressor and the read-time (concurrent) decompressor both build on these
// pure functions, so the codec must be LOSSLESS and FAST to decompress.
//
// Coverage:
//   (1) round-trip byte-identity for empty, tiny, highly-compressible (runs),
//       high-entropy (pseudo-random), and realistic blob-sized inputs;
//   (2) the incompressible-does-not-fit contract: when the compressed form
//       would not fit, lz_compress returns -1 (the caller stores uncompressed);
//   (3) lz_decompress REJECTS truncated/garbage input with -1 (no crash, no
//       overrun), with a canary byte past the output buffer verified untouched;
//   (4) a measured decompress throughput (GiB/s) on a multi-MiB buffer, printed
//       as a summary line (mirroring test_coalesced_expert_read) so FEAT-003's
//       docs can cite the real number.
//
// Pure library test, no fixtures: a plain sf_add_test, always runs in CI.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/lz_codec.h"
#include "test_util.h"

using namespace sf;

// Deterministic xorshift PRNG so the test is reproducible across platforms
// without depending on <random> engine details.
static uint64_t xorshift64(uint64_t &state) {
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}

// Round-trip a buffer and assert byte-identity. If the compressor returns -1
// (does not fit), that is the valid incompressible path and we assert the
// caller would simply store uncompressed; otherwise decompress must reproduce
// the exact input.
static void roundtrip_or_nofit(const std::vector<uint8_t> &input) {
    const size_t bound = lz_compress_bound(input.size());
    std::vector<uint8_t> comp(bound);
    const int64_t clen = lz_compress(
        input.empty() ? nullptr : input.data(), input.size(),
        comp.data(), comp.size());
    // With a worst-case bound destination, compression must always fit.
    CHECK(clen >= 0);
    if (clen < 0) return;

    std::vector<uint8_t> out(input.size());
    const int64_t dlen = lz_decompress(comp.data(),
                                       static_cast<size_t>(clen),
                                       out.empty() ? nullptr : out.data(),
                                       out.size());
    CHECK_EQ(dlen, static_cast<int64_t>(input.size()));
    if (dlen != static_cast<int64_t>(input.size())) return;
    if (!input.empty()) {
        CHECK_EQ(std::memcmp(out.data(), input.data(), input.size()), 0);
    }
}

static void test_empty() {
    std::vector<uint8_t> empty;
    roundtrip_or_nofit(empty);
}

static void test_small() {
    for (size_t n = 1; n <= 32; ++n) {
        std::vector<uint8_t> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
        }
        roundtrip_or_nofit(v);
    }
}

static void test_highly_compressible() {
    // Long zero run, then a long run of a single byte, then a repeating
    // pattern - all should compress well and round-trip exactly.
    std::vector<uint8_t> v(1u << 16, 0);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<uint8_t>((i / 97) & 1 ? 0xAB : 0x00);
    }
    roundtrip_or_nofit(v);

    std::vector<uint8_t> zeros(1u << 18, 0);
    roundtrip_or_nofit(zeros);

    std::vector<uint8_t> pattern(1u << 17);
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<uint8_t>("STRATAFLOW"[i % 10]);
    }
    roundtrip_or_nofit(pattern);
}

static void test_high_entropy() {
    // Pseudo-random (incompressible-ish) data: must still round-trip exactly.
    uint64_t st = 0x123456789abcdef0ull;
    std::vector<uint8_t> v(1u << 18);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<uint8_t>(xorshift64(st) & 0xFF);
    }
    roundtrip_or_nofit(v);
}

static void test_blob_sized() {
    // A realistic per-expert blob: a few MiB mixing compressible structure and
    // high-entropy regions (like K-quant data: mostly high entropy with some
    // repetition in block headers / zero padding).
    uint64_t st = 0xfeedfacecafebeefull;
    std::vector<uint8_t> v(6u * 1024u * 1024u);
    for (size_t i = 0; i < v.size(); ++i) {
        if ((i & 0x3F) < 8) {
            v[i] = 0x00;  // periodic structured/zero region
        } else {
            v[i] = static_cast<uint8_t>(xorshift64(st) & 0xFF);
        }
    }
    roundtrip_or_nofit(v);
}

static void test_does_not_fit_returns_minus_one() {
    // High-entropy data compressed into a too-small destination must return -1
    // (the caller then stores the input uncompressed with codec kNone).
    uint64_t st = 0x0f0f0f0f0f0f0f0full;
    std::vector<uint8_t> v(4096);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<uint8_t>(xorshift64(st) & 0xFF);
    }
    // Tiny destination that cannot hold the (near-incompressible) output.
    std::vector<uint8_t> tiny(16);
    const int64_t clen =
        lz_compress(v.data(), v.size(), tiny.data(), tiny.size());
    CHECK_EQ(clen, static_cast<int64_t>(-1));
}

static void test_rejects_malformed() {
    // Build a valid compressed buffer, then prove truncations and garbage are
    // rejected with -1 and never overrun a canary byte past the output.
    std::vector<uint8_t> input(4096);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint8_t>((i * 3 + (i >> 2)) & 0xFF);
    }
    std::vector<uint8_t> comp(lz_compress_bound(input.size()));
    const int64_t clen =
        lz_compress(input.data(), input.size(), comp.data(), comp.size());
    CHECK(clen > 0);
    if (clen <= 0) return;

    // Output buffer with a trailing canary we verify stays untouched.
    std::vector<uint8_t> out(input.size() + 1);
    const uint8_t canary = 0x7E;
    out.back() = canary;

    // (a) Truncate the stream at many lengths; every one must be -1 (or, for a
    //     prefix that happens to be a complete valid stream, decode to a size
    //     <= input - but with our header the only self-consistent length is the
    //     full stream, so shorter prefixes are rejected). No overrun either way.
    for (int64_t cut = 0; cut < clen; ++cut) {
        out.back() = canary;
        const int64_t d = lz_decompress(comp.data(), static_cast<size_t>(cut),
                                        out.data(), input.size());
        CHECK(d == -1);
        CHECK_EQ(out.back(), canary);
    }

    // (b) Header claiming a huge original length but no payload: reject.
    {
        std::vector<uint8_t> bad(4);
        bad[0] = 0xFF; bad[1] = 0xFF; bad[2] = 0xFF; bad[3] = 0x7F;
        out.back() = canary;
        const int64_t d = lz_decompress(bad.data(), bad.size(), out.data(),
                                        input.size());
        CHECK_EQ(d, static_cast<int64_t>(-1));
        CHECK_EQ(out.back(), canary);
    }

    // (c) Original length larger than dst_cap: reject.
    {
        std::vector<uint8_t> hdr(4);
        const uint32_t big = static_cast<uint32_t>(input.size() + 1000);
        std::memcpy(hdr.data(), &big, sizeof(big));
        out.back() = canary;
        const int64_t d = lz_decompress(hdr.data(), hdr.size(), out.data(),
                                        input.size());
        CHECK_EQ(d, static_cast<int64_t>(-1));
        CHECK_EQ(out.back(), canary);
    }

    // (d) Garbage body: valid header length but random following bytes. Must
    //     either reject with -1 or (improbably) decode, but never overrun.
    {
        std::vector<uint8_t> g(static_cast<size_t>(clen));
        std::memcpy(g.data(), comp.data(), 4);  // keep the valid header
        uint64_t st = 0xabcdef1234567890ull;
        for (size_t i = 4; i < g.size(); ++i) {
            g[i] = static_cast<uint8_t>(xorshift64(st) & 0xFF);
        }
        out.back() = canary;
        const int64_t d = lz_decompress(g.data(), g.size(), out.data(),
                                        input.size());
        CHECK(d == -1 || (d >= 0 && d <= static_cast<int64_t>(input.size())));
        CHECK_EQ(out.back(), canary);
    }
}

static void test_decompress_throughput() {
    // Multi-MiB buffer with realistic mixed content; measure decompress GiB/s.
    uint64_t st = 0x5151515151515151ull;
    const size_t n = 32u * 1024u * 1024u;  // 32 MiB
    std::vector<uint8_t> input(n);
    for (size_t i = 0; i < n; ++i) {
        if ((i & 0x7F) < 24) {
            input[i] = 0x00;  // structured region so there is real match work
        } else {
            input[i] = static_cast<uint8_t>(xorshift64(st) & 0xFF);
        }
    }

    std::vector<uint8_t> comp(lz_compress_bound(n));
    const int64_t clen =
        lz_compress(input.data(), n, comp.data(), comp.size());
    CHECK(clen > 0);
    if (clen <= 0) return;

    std::vector<uint8_t> out(n);
    // Warm-up + timed loop (several iterations to average out noise).
    const int iters = 8;
    int64_t dlen = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        dlen = lz_decompress(comp.data(), static_cast<size_t>(clen),
                             out.data(), out.size());
    }
    const auto t1 = std::chrono::steady_clock::now();
    CHECK_EQ(dlen, static_cast<int64_t>(n));
    CHECK_EQ(std::memcmp(out.data(), input.data(), n), 0);

    const double secs =
        std::chrono::duration<double>(t1 - t0).count();
    const double gib =
        (static_cast<double>(n) * iters) / (1024.0 * 1024.0 * 1024.0);
    const double gibps = secs > 0.0 ? gib / secs : 0.0;
    const double ratio =
        static_cast<double>(clen) / static_cast<double>(n);
    std::printf("[lz_codec: %zu MiB mixed, ratio %.3f, decompress %.2f GiB/s] ",
                n / (1024u * 1024u), ratio, gibps);
}

static void run_all() {
    RUN(test_empty);
    RUN(test_small);
    RUN(test_highly_compressible);
    RUN(test_high_entropy);
    RUN(test_blob_sized);
    RUN(test_does_not_fit_returns_minus_one);
    RUN(test_rejects_malformed);
    RUN(test_decompress_throughput);
}

TEST_MAIN()

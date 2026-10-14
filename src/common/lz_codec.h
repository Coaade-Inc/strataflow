// Self-contained, lossless, dependency-light LZ codec for streamed .strata
// blobs. The format is a byte-oriented LZ77 variant (LZ4-style sequences): a
// token byte splits into a 4-bit literal-length nibble and a 4-bit match-length
// nibble, each extensible by 0xFF continuation bytes, followed by the literal
// bytes and (for a match) a 2-byte little-endian back-reference offset. The
// window/offset is bounded to 16 bits (64 KiB), which is ample for the per-blob
// bundles StrataFlow streams. The codec is designed for FAST, correct decode
// (well above SSD bandwidth) rather than maximum ratio: the experts are already
// K-quantized (high entropy), so the win is reducing bytes read, not squeezing.
//
// All functions are PURE: they touch no shared or global mutable state and keep
// every scratch buffer caller-local, so the compute thread and the async
// prefetch worker may call them concurrently with their own per-call buffers.
//
// decompress(compress(x)) == x byte-exact for arbitrary binary input, and
// lz_decompress rejects malformed/truncated input with -1 (no overrun).
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace sf {

// On-disk codec identifier stored in the per-blob compression metadata. kNone
// means the bytes are stored verbatim (used when compression would not help or
// does not fit); kLz means they are lz_compress()'d.
enum class StrataCodec : uint8_t {
    kNone = 0,
    kLz = 1,
};

// Worst-case compressed size for src_len input bytes. A caller allocates a
// destination of at least this size before calling lz_compress. The bound
// accounts for the all-incompressible case (every byte a literal) plus the
// token/length-extension overhead and a small fixed header margin.
size_t lz_compress_bound(size_t src_len);

// Compress [src, src+src_len) into [dst, dst+dst_cap). Returns the number of
// bytes written (>= 0) on success, or -1 if the compressed form does not fit in
// dst_cap (the caller then stores the input UNCOMPRESSED with codec kNone).
// Pure: no shared state. src may be null only when src_len == 0.
int64_t lz_compress(const void *src, size_t src_len, void *dst,
                    size_t dst_cap);

// Decompress [src, src+src_len) into [dst, dst+dst_cap). Returns the number of
// decompressed bytes written (>= 0) on success, or -1 if the input is malformed
// or truncated or would overrun dst_cap. Pure: no shared state. Every literal
// and match copy is bounds-checked; a -1 return guarantees no out-of-bounds
// read or write occurred.
int64_t lz_decompress(const void *src, size_t src_len, void *dst,
                      size_t dst_cap);

}  // namespace sf

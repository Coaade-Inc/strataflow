// Implementation of the self-contained LZ codec declared in lz_codec.h.
//
// Format (all little-endian, byte-oriented, no alignment assumptions):
//
//   stream := header, sequence*
//   header := u32 original_length (little-endian, via memcpy)
//   sequence := token, [literal_length_ext], literals, [offset, match_ext]
//   token    := high nibble = literal length (0..15),
//               low  nibble = match length minus kMinMatch (0..15)
//
// If a nibble is 15 the true length is extended by reading continuation bytes
// until a byte < 0xFF (adding 0xFF each time, then the final byte). Literals
// are copied verbatim. A match is a 2-byte little-endian back-reference offset
// in [1, 65535] followed by a run of (kMinMatch + match_nibble[+ext]) bytes
// copied from offset bytes behind the current output position. The final
// sequence has literals and a match length nibble of 0 and NO offset (it is a
// trailing-literals-only run); we detect "last sequence" by exhausting src.
//
// This mirrors the LZ4 block layout closely enough to be trivially bit-exact
// while staying a single dependency-free file. The decoder bounds-checks every
// read from src and every write to dst, so malformed or truncated input yields
// -1 with no overrun.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "common/lz_codec.h"

#include <cstring>
#include <new>

namespace sf {
namespace {

// Minimum encodable match length. Matches shorter than this are emitted as
// literals (an LZ4 convention: short matches cost more than they save).
constexpr size_t kMinMatch = 4;

// Fixed stream header: a 4-byte little-endian original length. Lets the decoder
// validate the final output size and lets callers size dst exactly.
constexpr size_t kHeaderLen = 4;

// 16-bit bounded window: offsets are stored in two bytes, so a match can reach
// at most 65535 bytes back. Ample for the per-blob bundles we stream.
constexpr size_t kMaxOffset = 65535;

// Hash table for greedy match finding. 2^kHashLog buckets of the most recent
// position whose 4-byte sequence hashed there. Small, stack-free, per-call.
constexpr int kHashLog = 15;
constexpr size_t kHashSize = size_t(1) << kHashLog;

inline uint32_t read_u32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline void write_u32(uint8_t *p, uint32_t v) { std::memcpy(p, &v, sizeof(v)); }

inline uint32_t hash4(uint32_t seq) {
    // Fibonacci-ish multiplicative hash, take the top kHashLog bits.
    return (seq * 2654435761u) >> (32 - kHashLog);
}

}  // namespace

size_t lz_compress_bound(size_t src_len) {
    // Worst case: pure literals. LZ4-style literal-run overhead is one token
    // per 255-ish literals plus a length-extension byte roughly every 255
    // bytes. src_len + src_len/255 + 16 is a safe, standard bound; add the
    // fixed header.
    return kHeaderLen + src_len + (src_len / 255) + 16;
}

int64_t lz_compress(const void *src, size_t src_len, void *dst,
                    size_t dst_cap) {
    if (src_len == 0) {
        // Empty input: just the header with original_length == 0.
        if (dst_cap < kHeaderLen) return -1;
        write_u32(static_cast<uint8_t *>(dst), 0);
        return static_cast<int64_t>(kHeaderLen);
    }
    if (src == nullptr || dst == nullptr) return -1;
    // Original length must fit in the 32-bit header.
    if (src_len > 0xFFFFFFFFull) return -1;

    const uint8_t *const in = static_cast<const uint8_t *>(src);
    uint8_t *const out = static_cast<uint8_t *>(dst);
    uint8_t *op = out;
    uint8_t *const op_end = out + dst_cap;

    // Emit header.
    if (dst_cap < kHeaderLen) return -1;
    write_u32(op, static_cast<uint32_t>(src_len));
    op += kHeaderLen;

    // Per-call hash table (no shared state). Initialized so that "no match"
    // is distinguishable: store position+1, 0 means empty.
    // Heap-allocate to keep the stack small and stay thread-safe per call.
    uint32_t *table = new (std::nothrow) uint32_t[kHashSize];
    if (table == nullptr) return -1;
    std::memset(table, 0, kHashSize * sizeof(uint32_t));

    size_t ip = 0;              // current input position
    size_t anchor = 0;          // start of the pending literal run
    // Leave room so read_u32 at ip+ is always in-bounds for match search.
    const size_t limit = (src_len >= kMinMatch) ? (src_len - kMinMatch) : 0;

    auto emit_len = [&](size_t len) -> bool {
        // Emit length-extension bytes for a value already known to be >= 15.
        size_t remaining = len - 15;
        while (remaining >= 255) {
            if (op >= op_end) return false;
            *op++ = 255;
            remaining -= 255;
        }
        if (op >= op_end) return false;
        *op++ = static_cast<uint8_t>(remaining);
        return true;
    };

    auto emit_sequence = [&](size_t lit_len, size_t match_len,
                             size_t offset) -> bool {
        // match_len == 0 marks a trailing-literals-only sequence (no offset).
        const size_t m_nib = (match_len == 0)
                                 ? 0
                                 : (match_len - kMinMatch);
        const uint8_t lit_tok =
            static_cast<uint8_t>(lit_len >= 15 ? 15 : lit_len);
        const uint8_t m_tok = static_cast<uint8_t>(m_nib >= 15 ? 15 : m_nib);
        if (op >= op_end) return false;
        *op++ = static_cast<uint8_t>((lit_tok << 4) | m_tok);
        if (lit_len >= 15) {
            if (!emit_len(lit_len)) return false;
        }
        // Copy literals.
        if (lit_len > 0) {
            if (op + lit_len > op_end) return false;
            std::memcpy(op, in + anchor, lit_len);
            op += lit_len;
        }
        if (match_len != 0) {
            // 2-byte little-endian offset.
            if (op + 2 > op_end) return false;
            *op++ = static_cast<uint8_t>(offset & 0xFF);
            *op++ = static_cast<uint8_t>((offset >> 8) & 0xFF);
            if (m_nib >= 15) {
                if (!emit_len(m_nib)) return false;
            }
        }
        return true;
    };

    while (ip < limit) {
        const uint32_t seq = read_u32(in + ip);
        const uint32_t h = hash4(seq);
        const uint32_t cand = table[h];  // position+1 or 0
        table[h] = static_cast<uint32_t>(ip + 1);

        bool have_match = false;
        size_t mpos = 0;
        if (cand != 0) {
            mpos = static_cast<size_t>(cand - 1);
            const size_t off = ip - mpos;
            if (off >= 1 && off <= kMaxOffset && read_u32(in + mpos) == seq) {
                have_match = true;
            }
        }

        if (!have_match) {
            ++ip;
            continue;
        }

        // Extend the match forward as far as possible.
        size_t mlen = kMinMatch;
        while (ip + mlen < src_len && in[mpos + mlen] == in[ip + mlen]) {
            ++mlen;
        }
        const size_t offset = ip - mpos;
        const size_t lit_len = ip - anchor;

        if (!emit_sequence(lit_len, mlen, offset)) {
            delete[] table;
            return -1;
        }

        // Advance past the match. Insert hashes for a couple of interior
        // positions to improve subsequent matching (cheap, optional).
        ip += mlen;
        anchor = ip;
    }

    // Trailing literals (everything from anchor to end) as a final sequence
    // with match_len == 0 (no offset). Emit ONLY when there are leftover
    // literals: if the last match consumed the input exactly (anchor ==
    // src_len) the output is already complete and the output-driven decoder
    // stops without expecting a further token, so emitting an empty trailing
    // sequence here would leave an unconsumed byte and fail the ip==ip_end
    // check.
    const size_t tail = src_len - anchor;
    if (tail > 0) {
        if (!emit_sequence(tail, 0, 0)) {
            delete[] table;
            return -1;
        }
    }

    delete[] table;
    return static_cast<int64_t>(op - out);
}

int64_t lz_decompress(const void *src, size_t src_len, void *dst,
                      size_t dst_cap) {
    if (src == nullptr && src_len != 0) return -1;
    // Need at least the header.
    if (src_len < kHeaderLen) return -1;

    const uint8_t *const in = static_cast<const uint8_t *>(src);
    const uint8_t *ip = in;
    const uint8_t *const ip_end = in + src_len;

    const uint32_t orig_len = read_u32(ip);
    ip += kHeaderLen;

    if (orig_len > dst_cap) return -1;

    if (orig_len == 0) {
        // Must be exactly the header, nothing more.
        return (ip == ip_end) ? 0 : -1;
    }
    if (dst == nullptr) return -1;

    uint8_t *const out = static_cast<uint8_t *>(dst);
    uint8_t *op = out;
    uint8_t *const op_end = out + orig_len;

    // Read an extended length starting from a nibble value of 15.
    auto read_ext = [&](size_t base, size_t &out_len) -> bool {
        size_t len = base;
        for (;;) {
            if (ip >= ip_end) return false;
            const uint8_t b = *ip++;
            len += b;
            if (b != 255) break;
        }
        out_len = len;
        return true;
    };

    // Output-length-driven loop: we decode sequences until exactly orig_len
    // bytes have been produced. This makes truncation unambiguous - the final
    // sequence is literals-only (its token's match nibble is written but no
    // offset follows), and we recognize it because copying its literals fills
    // the output (op reaches op_end), after which NO offset is read.
    while (op < op_end) {
        if (ip >= ip_end) return -1;  // more output expected but input ended
        const uint8_t token = *ip++;
        size_t lit_len = static_cast<size_t>(token >> 4);
        size_t m_nib = static_cast<size_t>(token & 0x0F);

        if (lit_len == 15) {
            if (!read_ext(15, lit_len)) return -1;
        }

        // Copy literals with full bounds checking.
        if (lit_len > 0) {
            if (static_cast<size_t>(ip_end - ip) < lit_len) return -1;
            if (static_cast<size_t>(op_end - op) < lit_len) return -1;
            std::memcpy(op, ip, lit_len);
            ip += lit_len;
            op += lit_len;
        }

        // If the literals filled the output, this was the trailing
        // literals-only final sequence: no offset/match follows, stop.
        if (op == op_end) break;

        // Otherwise a match follows: 2-byte offset then (kMinMatch + m_nib).
        if (static_cast<size_t>(ip_end - ip) < 2) return -1;
        const size_t offset = static_cast<size_t>(ip[0]) |
                              (static_cast<size_t>(ip[1]) << 8);
        ip += 2;
        if (offset == 0) return -1;  // offset 0 is never valid

        size_t match_len = kMinMatch + m_nib;
        if (m_nib == 15) {
            if (!read_ext(15, m_nib)) return -1;
            match_len = kMinMatch + m_nib;
        }

        // The back-reference must stay within already-emitted output.
        const size_t produced = static_cast<size_t>(op - out);
        if (offset > produced) return -1;
        if (static_cast<size_t>(op_end - op) < match_len) return -1;

        // Copy the match. When the source and destination do not overlap
        // (offset >= match_len) a single memcpy is correct and fast. When they
        // overlap (offset < match_len, an RLE-style run) the copy MUST be
        // byte-by-byte so each written byte feeds the next read.
        const uint8_t *mref = op - offset;
        if (offset >= match_len) {
            std::memcpy(op, mref, match_len);
        } else {
            for (size_t i = 0; i < match_len; ++i) {
                op[i] = mref[i];
            }
        }
        op += match_len;
    }

    if (op != op_end) return -1;  // decoded size must match the header exactly
    if (ip != ip_end) return -1;  // trailing/garbage bytes after full output
    return static_cast<int64_t>(op - out);
}

}  // namespace sf

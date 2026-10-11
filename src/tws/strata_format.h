// The .strata packed on-disk format (v1): shared definitions used by the
// packer (tools/strata-pack) and the reader (src/tws/strata_file).
//
// See docs/TASK5_DESIGN.md section 2 for the authoritative format. In short a
// .strata file is:
//   [ 128-byte superblock ]
//   [ embedded GGUF metadata blob (verbatim header+KV+tensor-info, no_alloc) ]
//   [ expert index (array of ExpertIndexEntry) ]
//   [ trunk block     (4 KiB-aligned start, all non-expert tensors) ]
//   [ expert region   (4 KiB-aligned start, one 4 KiB-aligned blob per
//                      (layer,expert) holding gate+up+down bundled) ]
//
// Little-endian throughout, matching GGUF. The structs below are written and
// read as raw bytes; they are declared with fixed-width integers and are
// trivially copyable so a single read/write of sizeof() moves the whole record.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace sf {

// Fixed superblock size in bytes (the file always starts with exactly this
// many bytes; trailing bytes after the fields are reserved/zero).
constexpr uint32_t kStrataSuperblockSize = 128;

// Magic occupies the first 8 bytes of the file. Not NUL-terminated on disk.
constexpr char     kStrataMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', '0', '1'};
// v2 (FEAT-002 'stream the trunk') adds an optional per-trunk-tensor index so
// an individual trunk tensor can be located and streamed without loading the
// whole trunk block. The change is strictly additive: the trunk block bytes
// are byte-for-byte identical to v1; v2 only appends a trunk index region and
// carves two superblock fields out of the previously-reserved pad. A reader
// that understands v2 falls back to full-resident trunk when the trunk index
// is absent (trunk_index_count==0), so a v2 file without the index behaves
// exactly like v1.
constexpr uint32_t kStrataVersion  = 2;

// Superblock flag bits.
constexpr uint32_t kStrataFlagExpertsBundled = 1u << 0;  // gate+up+down bundled
constexpr uint32_t kStrataFlagIndexSorted    = 1u << 1;  // sorted {layer,expert}
constexpr uint32_t kStrataFlagTrunkIndexed   = 1u << 2;  // per-trunk-tensor index present

// The fixed 128-byte superblock at offset 0. All multi-byte fields are stored
// little-endian; on the little-endian targets we support (x86-64, arm64) the
// in-memory layout matches the on-disk layout, so a raw read/write is correct.
// `header_crc32` covers bytes [0, offsetof(header_crc32)); i.e. every field
// before it, which is everything that must be trusted before using an offset.
struct StrataSuperblock {
    char     magic[8];                 // "STRATA01"
    uint32_t version;                  // == kStrataVersion
    uint32_t flags;                    // kStrataFlag*
    uint64_t gguf_meta_offset;         // start of the embedded GGUF blob
    uint64_t gguf_meta_size;           // bytes
    uint64_t index_offset;             // start of the expert index
    uint64_t index_count;              // number of ExpertIndexEntry
    uint64_t trunk_offset;             // 4 KiB-aligned start of trunk block
    uint64_t trunk_size;               // bytes
    uint64_t expert_region_offset;     // 4 KiB-aligned start of expert region
    uint64_t expert_region_size;       // bytes
    // v2 trunk index (FEAT-002). Carved from the former reserved[32] pad, so
    // sizeof(StrataSuperblock) is unchanged at 128 bytes, and placed here (both
    // 8-byte aligned, before header_crc32) so the existing CRC backfill covers
    // them. On a v1 file (or a v2 file without a trunk index) both are 0 and
    // the engine falls back to a fully-resident trunk.
    uint64_t trunk_index_offset;       // start of the trunk index region, or 0
    uint64_t trunk_index_count;        // number of TrunkIndexEntry, or 0
    uint32_t n_layers;                 // convenience (redundant with GGUF)
    uint32_t n_experts;                // convenience (redundant with GGUF)
    uint32_t align;                    // the alignment used (== kIoAlignment)
    uint32_t header_crc32;             // CRC32 of bytes before this field
    uint8_t  reserved[16];             // pad to 128 bytes (was 32; 16 bytes
                                       // repurposed for the two fields above)
};

static_assert(sizeof(StrataSuperblock) == kStrataSuperblockSize,
              "StrataSuperblock must be exactly 128 bytes");

// Byte offset of header_crc32 within the superblock; the CRC covers [0, this).
constexpr size_t kStrataHeaderCrcOffset = offsetof(StrataSuperblock, header_crc32);

// One index record per (layer, expert). 48 bytes, 8-byte aligned. The three
// *_rel fields are byte offsets of each kind's slice WITHIN the blob, so the
// reader maps {layer,expert,kind} to an exact aligned sub-read without having
// to re-derive sizes. For a fused gate_up arch, gate_rel == the gate_up slice
// and up_rel == gate_rel (the fused tensor is a single slice).
struct ExpertIndexEntry {
    uint32_t layer;        // blk.N
    uint32_t expert;       // 0..n_experts-1
    uint64_t blob_offset;  // absolute file offset, 4 KiB-aligned
    uint64_t blob_length;  // total bytes of the bundle (incl. internal pad)
    uint64_t gate_rel;     // byte offset of the gate slice within the blob
    uint64_t up_rel;       // byte offset of the up slice within the blob
    uint64_t down_rel;     // byte offset of the down slice within the blob
};

static_assert(sizeof(ExpertIndexEntry) == 48,
              "ExpertIndexEntry must be exactly 48 bytes");

// Max trunk-tensor-name length (including the NUL terminator) stored in a
// TrunkIndexEntry. llama trunk tensor names are short - the longest is of the
// form "blk.NN.attn_output.weight" (~25 chars) or "token_embd.weight" - so 64
// bytes is comfortably large and keeps the record a round 80 bytes. The name
// is stored as a fixed-size, NUL-terminated (and NUL-padded) char array so the
// index is a flat array of fixed-size records that round-trips with a single
// read/write of count*sizeof(TrunkIndexEntry).
constexpr uint32_t kStrataTrunkNameMax = 64;

// One index record per TRUNK tensor (v2+, when kStrataFlagTrunkIndexed is set).
// 80 bytes, 8-byte aligned. `rel_offset` is the tensor's byte offset WITHIN the
// trunk block (so the absolute file offset is superblock().trunk_offset +
// rel_offset); `length` is ggml_nbytes of the tensor. Trunk tensors are packed
// verbatim and contiguously in GGUF tensor order, so these records simply name
// the slot each tensor already occupies - the trunk block bytes are unchanged
// vs v1.
struct TrunkIndexEntry {
    char     name[kStrataTrunkNameMax];  // GGUF tensor name, NUL-padded
    uint64_t rel_offset;                 // byte offset within the trunk block
    uint64_t length;                     // ggml_nbytes of the tensor
};

static_assert(sizeof(TrunkIndexEntry) == kStrataTrunkNameMax + 16,
              "TrunkIndexEntry must be name + two uint64 (80 bytes)");

// CRC32 (IEEE 802.3, reflected, poly 0xEDB88320) over `len` bytes of `data`.
// Standalone and header-only so both the packer and the reader share one
// implementation with no extra link dependency.
inline uint32_t strata_crc32(const void *data, size_t len) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = static_cast<uint32_t>(-(int32_t)(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

}  // namespace sf

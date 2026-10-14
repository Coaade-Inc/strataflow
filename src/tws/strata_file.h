// StrataReader: opens a .strata packed model (see docs/TASK5_DESIGN.md section
// 2 and 3) via BlockFile, validates the superblock, parses the expert index,
// and serves aligned expert-blob reads.
//
// This is the byte-provenance seam for Task 5: the streaming buffer type
// (src/tws/stream_buft.cpp) sources expert bytes from read_blob() when a
// .strata is loaded, and read_shape_from_gguf (src/model/ggml_model.cpp) parses
// metadata() (the embedded GGUF blob) so the whole load path works unchanged on
// a .strata. The reader never reinterprets weights; it moves verbatim bytes.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include "tws/async_io.h"
#include "tws/strata_format.h"
#include "tws/stream_buft.h"  // ExpertTensorKind

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sf {

// Returns true if `path` names a file whose first 8 bytes are the STRATA01
// magic. Cheap: opens, reads 8 bytes, closes. False on any I/O error or a
// non-.strata (e.g. a plain GGUF, whose magic is "GGUF").
bool is_strata_file(const std::string &path);

class StrataReader {
public:
    StrataReader() = default;
    ~StrataReader() = default;
    StrataReader(const StrataReader &) = delete;
    StrataReader &operator=(const StrataReader &) = delete;

    // Open and validate a .strata file. Returns false (and leaves the reader
    // unusable) if the file is missing, not a .strata, fails superblock
    // validation (magic/version/align/CRC), or the index cannot be read.
    bool open(const std::string &path);

    bool is_open() const { return open_; }

    const StrataSuperblock &superblock() const { return sb_; }

    // Pointer/length into the embedded GGUF metadata blob (header+KV+
    // tensor-info, verbatim). The bytes live in an owned buffer valid for the
    // reader's lifetime, so gguf_init_from_buffer can parse them directly.
    const uint8_t *metadata() const { return meta_.data(); }
    uint64_t       metadata_size() const { return meta_.size(); }

    // Number of index entries == n_layers * n_experts for a dense-index pack.
    uint64_t index_count() const { return index_.size(); }
    const std::vector<ExpertIndexEntry> &index() const { return index_; }

    // Look up the index entry for {layer,expert}. Returns nullptr if absent.
    const ExpertIndexEntry *find(uint32_t layer, uint32_t expert) const;

    // The absolute file offset and length of one {layer,expert,kind} slice
    // inside its bundle. Returns false if the key is unknown.
    bool slice_range(uint32_t layer, uint32_t expert, ExpertTensorKind kind,
                     uint64_t &offset, uint64_t &length) const;

    // Read exactly the {layer,expert,kind} slice into `dst` (must hold at least
    // the slice length). Returns the number of bytes read, or -1 on error. The
    // underlying BlockFile performs the aligned direct-I/O read.
    int64_t read_blob(uint32_t layer, uint32_t expert, ExpertTensorKind kind,
                      void *dst) const;

    // FEAT-002 COALESCED BUNDLE READ. Read the WHOLE (layer,expert) bundle -
    // gate|up|down contiguously - into `dst` (which must hold at least
    // blob_length bytes) with ONE underlying read_at over the bundle's
    // [blob_offset, blob_offset+blob_length) range, instead of the three
    // separate read_blob(kGate/kUp/kDown) calls (3 reads -> 1). The slices are
    // packed TIGHTLY inside the bundle in gate|up|down order (only the blob
    // START is 4 KiB-aligned, see tools/strata-pack), so the bytes `dst`
    // receives are BYTE-IDENTICAL to the three read_blob calls concatenated:
    // read_blob(kGate,dst) then read_blob(kUp,dst+gate_len) then
    // read_blob(kDown,dst+gate_len+up_len) produce exactly these same bytes in
    // these same positions. This changes HOW the bytes are read (one large
    // sequential read, higher effective bandwidth), never WHICH bytes.
    //
    // Returns the number of bytes read (== blob_length) on success, or -1 on a
    // lookup/read error OR on a VERBATIM FUSED-gate_up bundle (gate_rel ==
    // up_rel), whose two-slice on-disk layout does not match the engine's
    // three-slice slot; the caller must then fall back to the per-kind
    // read_blob path. A COMPRESSED fused bundle IS served here: the whole blob
    // is decoded once into dst (its decompressed layout is the same tightly-
    // packed gate|up|down the three read_blob calls would produce), so a fused
    // compressed expert is read and decoded exactly once rather than three
    // times. streamed_bytes() counts the bundle's (compressed, when stored
    // compressed) bytes exactly once (identical total to three reads).
    int64_t read_bundle(uint32_t layer, uint32_t expert, void *dst) const;

    // Count of underlying BlockFile::read_at calls issued by the EXPERT read
    // path (read_blob + read_bundle) since open(). read_blob increments it once
    // per slice (3 per bundle); read_bundle increments it once per bundle. Lets
    // a test prove the 3:1 coalescing (read_bundle is 1 call where three
    // read_blob calls would be 3). Trunk reads are not counted here (they go
    // through read_at / read_trunk_tensor). Atomic for the same reason as
    // streamed_bytes_: the async prefetch worker may call concurrently.
    uint64_t read_calls() const {
        return read_calls_.load(std::memory_order_relaxed);
    }

    // Read `len` bytes at the absolute file `offset` into `dst`. Used by the
    // engine (src/model/engine) to fill the resident trunk tensors from the
    // trunk block (whose start is superblock().trunk_offset). Returns bytes
    // read, or -1 on error.
    int64_t read_at(void *dst, uint64_t len, uint64_t offset) const;

    // --- v2 trunk index (FEAT-002 'stream the trunk') ----------------------
    // Number of per-trunk-tensor index records parsed from the file. 0 for a
    // v1 file or a v2 file without the kStrataFlagTrunkIndexed flag, in which
    // case the engine falls back to a fully-resident trunk.
    uint64_t trunk_index_count() const { return trunk_index_.size(); }

    // True when this file carries a usable per-trunk-tensor index (v2 + flag +
    // at least one entry). The engine gates trunk streaming on this.
    bool has_trunk_index() const { return !trunk_index_.empty(); }

    // Absolute file offset + length of a single trunk tensor's bytes, looked up
    // by GGUF tensor name. Returns false if the name is not in the trunk index.
    // offset = superblock().trunk_offset + entry.rel_offset.
    bool trunk_slice_range(const std::string &name, uint64_t &offset,
                           uint64_t &length) const;

    // Read exactly one trunk tensor's bytes (by GGUF name) into `dst` (which
    // must hold at least the tensor's length). Returns bytes read, or -1 if the
    // name is unknown or the read fails. Const + read-only like read_blob, so
    // it is safe to call from any thread, but this feature only calls it from
    // the compute thread (synchronous, before the segment that reads it).
    //
    // Trunk bytes are counted separately from streamed_bytes() (which stays
    // expert-only for the bench harness); see streamed_trunk_bytes().
    int64_t read_trunk_tensor(const std::string &name, void *dst) const;

    // Total TRUNK bytes streamed via read_trunk_tensor() since open(). Kept
    // DISTINCT from streamed_bytes() (experts) so the benchmark harness's
    // per-token expert-streaming number is unchanged; trunk streaming is a
    // separate, per-edge accounting.
    uint64_t streamed_trunk_bytes() const {
        return streamed_trunk_bytes_.load(std::memory_order_relaxed);
    }

    // Total EXPERT bytes streamed from disk via read_blob() since open(). This
    // counts only per-token expert-slice reads; the one-time trunk fill (which
    // goes through read_at) is intentionally excluded so this reflects
    // per-token expert streaming, not the trunk load. Used by the benchmark
    // harness to report bytes-streamed-per-token. See read_blob() in the .cpp.
    uint64_t streamed_bytes() const {
        return streamed_bytes_.load(std::memory_order_relaxed);
    }

private:
    uint64_t slice_rel(const ExpertIndexEntry &e, ExpertTensorKind kind) const;

    // The compression record for the index entry at position `index_pos`, or
    // nullptr when the file carries no compression index (v1/v2, or a v3 file
    // with no compressed blob) so the blob is stored verbatim.
    const ExpertBlobCompression *compression_at(size_t index_pos) const {
        if (index_pos >= compress_.size()) return nullptr;
        return &compress_[index_pos];
    }

    bool                            open_ = false;
    mutable BlockFile               file_;
    // Accumulated expert-slice bytes read via read_blob(). mutable because
    // read_blob() is const (it never mutates logical reader state) but still
    // needs to bump this counter. ATOMIC because the async prefetch worker
    // (src/model/engine/engine.cpp) now calls read_blob() concurrently with the
    // compute thread's own read_blob() in ensure_layer_experts_resident, so the
    // increment races; relaxed order is sufficient (it is a pure monotonic
    // byte-counter whose value is read only for benchmark reporting, with no
    // ordering dependency on other memory). The reported total is unchanged.
    mutable std::atomic<uint64_t>   streamed_bytes_{0};
    // Count of underlying read_at calls on the EXPERT path (read_blob +
    // read_bundle). mutable+atomic for the same reasons as streamed_bytes_
    // (const read path, concurrent async prefetch worker, benchmark reporting).
    mutable std::atomic<uint64_t>   read_calls_{0};
    // Accumulated TRUNK bytes read via read_trunk_tensor(). Separate counter so
    // streamed_bytes() keeps its expert-only meaning. mutable+atomic for the
    // same reasons as streamed_bytes_ (const read path, benchmark reporting).
    mutable std::atomic<uint64_t>   streamed_trunk_bytes_{0};
    StrataSuperblock                sb_{};
    std::vector<uint8_t>            meta_;
    std::vector<ExpertIndexEntry>   index_;
    // {layer,expert} -> index into index_. Key packs layer<<32 | expert.
    std::unordered_map<uint64_t, size_t> map_;
    // v2 trunk index: records in file order, plus name -> index into the vector
    // for O(1) lookup. Empty for a v1 file (full-resident fallback).
    std::vector<TrunkIndexEntry>    trunk_index_;
    std::unordered_map<std::string, size_t> trunk_map_;
    // v3 per-expert-blob compression index (FEAT-002 'compress what streams'),
    // aligned 1:1 with index_ by POSITION (compress_[i] describes index_[i]'s
    // on-disk blob). Empty for a v1/v2 file or a v3 file with no compressed
    // blob, in which case every blob is streamed verbatim (byte-identical to
    // the uncompressed path). read_blob/read_bundle decompress transparently
    // when compress_[i].codec != kNone, using PER-CALL-LOCAL scratch only.
    std::vector<ExpertBlobCompression> compress_;
};

}  // namespace sf

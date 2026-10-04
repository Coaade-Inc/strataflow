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

    // Read `len` bytes at the absolute file `offset` into `dst`. Used by the
    // engine (src/model/engine) to fill the resident trunk tensors from the
    // trunk block (whose start is superblock().trunk_offset). Returns bytes
    // read, or -1 on error.
    int64_t read_at(void *dst, uint64_t len, uint64_t offset) const;

private:
    uint64_t slice_rel(const ExpertIndexEntry &e, ExpertTensorKind kind) const;

    bool                            open_ = false;
    mutable BlockFile               file_;
    StrataSuperblock                sb_{};
    std::vector<uint8_t>            meta_;
    std::vector<ExpertIndexEntry>   index_;
    // {layer,expert} -> index into index_. Key packs layer<<32 | expert.
    std::unordered_map<uint64_t, size_t> map_;
};

}  // namespace sf

// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "tws/strata_file.h"

#include "common/log.h"

#include <cstring>

namespace sf {
namespace {

// Pack {layer,expert} into a single 64-bit key for the lookup map.
uint64_t pack_key(uint32_t layer, uint32_t expert) {
    return (static_cast<uint64_t>(layer) << 32) | static_cast<uint64_t>(expert);
}

}  // namespace

bool is_strata_file(const std::string &path) {
    BlockFile f;
    if (!f.open(path)) return false;
    char magic[sizeof(kStrataMagic)] = {};
    const int64_t n = f.read_at(magic, sizeof(magic), 0);
    f.close();
    if (n != static_cast<int64_t>(sizeof(magic))) return false;
    return std::memcmp(magic, kStrataMagic, sizeof(kStrataMagic)) == 0;
}

bool StrataReader::open(const std::string &path) {
    open_ = false;
    index_.clear();
    map_.clear();
    meta_.clear();

    if (!file_.open(path)) {
        log_warn("StrataReader: cannot open '" + path + "'");
        return false;
    }

    // --- Superblock ---------------------------------------------------------
    if (file_.size() < kStrataSuperblockSize) {
        log_warn("StrataReader: '" + path + "' too small for a superblock");
        return false;
    }
    if (file_.read_at(&sb_, sizeof(sb_), 0) !=
        static_cast<int64_t>(sizeof(sb_))) {
        log_warn("StrataReader: failed to read superblock from '" + path + "'");
        return false;
    }
    if (std::memcmp(sb_.magic, kStrataMagic, sizeof(kStrataMagic)) != 0) {
        log_warn("StrataReader: bad magic in '" + path + "'");
        return false;
    }
    if (sb_.version != kStrataVersion) {
        log_warn("StrataReader: unsupported version in '" + path + "'");
        return false;
    }
    if (sb_.align != static_cast<uint32_t>(kIoAlignment)) {
        log_warn("StrataReader: alignment mismatch in '" + path + "'");
        return false;
    }
    {
        // Validate header_crc32 over the bytes it covers.
        StrataSuperblock copy = sb_;
        const uint32_t stored = copy.header_crc32;
        const uint32_t computed = strata_crc32(&copy, kStrataHeaderCrcOffset);
        if (stored != computed) {
            log_warn("StrataReader: header CRC mismatch in '" + path + "'");
            return false;
        }
    }

    // --- Embedded GGUF metadata blob ---------------------------------------
    if (sb_.gguf_meta_size == 0 ||
        sb_.gguf_meta_offset + sb_.gguf_meta_size > file_.size()) {
        log_warn("StrataReader: invalid metadata region in '" + path + "'");
        return false;
    }
    meta_.resize(static_cast<size_t>(sb_.gguf_meta_size));
    if (file_.read_at(meta_.data(), meta_.size(), sb_.gguf_meta_offset) !=
        static_cast<int64_t>(meta_.size())) {
        log_warn("StrataReader: failed to read metadata from '" + path + "'");
        return false;
    }

    // --- Expert index -------------------------------------------------------
    const uint64_t index_bytes = sb_.index_count * sizeof(ExpertIndexEntry);
    if (sb_.index_offset + index_bytes > file_.size()) {
        log_warn("StrataReader: invalid index region in '" + path + "'");
        return false;
    }
    index_.resize(static_cast<size_t>(sb_.index_count));
    if (sb_.index_count > 0) {
        if (file_.read_at(index_.data(), static_cast<size_t>(index_bytes),
                          sb_.index_offset) !=
            static_cast<int64_t>(index_bytes)) {
            log_warn("StrataReader: failed to read index from '" + path + "'");
            index_.clear();
            return false;
        }
    }
    map_.reserve(index_.size());
    for (size_t i = 0; i < index_.size(); ++i) {
        map_[pack_key(index_[i].layer, index_[i].expert)] = i;
    }

    open_ = true;
    log_info("StrataReader: opened '" + path + "' (" +
             std::to_string(sb_.n_layers) + " layers, " +
             std::to_string(sb_.n_experts) + " experts, " +
             std::to_string(index_.size()) + " index entries, backend " +
             file_.active_backend_name() + ")");
    return true;
}

const ExpertIndexEntry *StrataReader::find(uint32_t layer,
                                           uint32_t expert) const {
    auto it = map_.find(pack_key(layer, expert));
    if (it == map_.end()) return nullptr;
    return &index_[it->second];
}

uint64_t StrataReader::slice_rel(const ExpertIndexEntry &e,
                                 ExpertTensorKind kind) const {
    switch (kind) {
        case ExpertTensorKind::kGate:   return e.gate_rel;
        case ExpertTensorKind::kUp:     return e.up_rel;
        case ExpertTensorKind::kDown:   return e.down_rel;
        case ExpertTensorKind::kGateUp: return e.gate_rel;  // fused == gate slot
    }
    return e.gate_rel;
}

bool StrataReader::slice_range(uint32_t layer, uint32_t expert,
                               ExpertTensorKind kind, uint64_t &offset,
                               uint64_t &length) const {
    const ExpertIndexEntry *e = find(layer, expert);
    if (e == nullptr) return false;

    // Slices are stored tightly packed in {gate, up, down} order inside the
    // bundle (only the blob START is 4 KiB-aligned, see tools/strata-pack).
    // A slice's length is therefore the gap to the next slice's rel offset,
    // and the last slice (down) runs to blob_length.
    const uint64_t rel = slice_rel(*e, kind);
    uint64_t end = e->blob_length;
    // The smallest *_rel strictly greater than `rel` bounds this slice.
    const uint64_t rels[3] = {e->gate_rel, e->up_rel, e->down_rel};
    for (uint64_t r : rels) {
        if (r > rel && r < end) end = r;
    }
    offset = e->blob_offset + rel;
    length = end - rel;
    return true;
}

int64_t StrataReader::read_blob(uint32_t layer, uint32_t expert,
                                ExpertTensorKind kind, void *dst) const {
    uint64_t offset = 0;
    uint64_t length = 0;
    if (!slice_range(layer, expert, kind, offset, length)) return -1;
    return file_.read_at(dst, static_cast<size_t>(length), offset);
}

}  // namespace sf

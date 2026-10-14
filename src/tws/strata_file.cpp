// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "tws/strata_file.h"

#include "common/log.h"
#include "common/lz_codec.h"

#include <cstring>
#include <vector>

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
    trunk_index_.clear();
    trunk_map_.clear();
    compress_.clear();

    if (!file_.open(path)) {
        log_warn("StrataReader: cannot open '" + path + "'");
        return false;
    }

    // --- Superblock ---------------------------------------------------------
    if (file_.size() < kStrataSuperblockSize) {
        log_warn("StrataReader: '" + path + "' too small for a superblock");
        return false;
    }
    // Read the raw 128 superblock bytes once. We parse version-aware from these
    // bytes (v1 and v2 differ in trailing-field offsets), then normalize into
    // the in-memory v2 struct sb_.
    uint8_t raw[kStrataSuperblockSize];
    if (file_.read_at(raw, sizeof(raw), 0) !=
        static_cast<int64_t>(sizeof(raw))) {
        log_warn("StrataReader: failed to read superblock from '" + path + "'");
        return false;
    }
    std::memcpy(&sb_, raw, sizeof(sb_));
    if (std::memcmp(sb_.magic, kStrataMagic, sizeof(kStrataMagic)) != 0) {
        log_warn("StrataReader: bad magic in '" + path + "'");
        return false;
    }
    // Read the version from its fixed offset first (it is identical in v1 and
    // v2: right after the 8-byte magic). The superblock FIELD LAYOUT differs
    // between v1 and v2 for everything after expert_region_size: v2 inserts
    // trunk_index_offset + trunk_index_count (two uint64 = 16 bytes) ahead of
    // n_layers, which shifts n_layers/n_experts/align/header_crc32 down by 16
    // bytes. So a v1 file CANNOT be interpreted with the v2 struct layout: we
    // must re-map the trailing fields from their v1 offsets. We accept v1 and
    // v2; v1 has no trunk index, so it reports trunk_index_count()==0 and the
    // engine falls back to a fully-resident trunk (byte-identical to before).
    uint32_t version = 0;
    std::memcpy(&version, raw + 8, sizeof(version));
    if (version != 1 && version != 2 && version != 3) {
        log_warn("StrataReader: unsupported version in '" + path + "'");
        return false;
    }

    // The three on-disk layouts differ ONLY in the trailing scalars/index
    // fields after expert_region_size (which ends at byte 80); everything
    // before that is byte-identical. Per-version trailing offsets (little-
    // endian), see src/tws/strata_format.h for the authoritative table:
    //   v1: n_layers@80 n_experts@84 align@88 header_crc32@92  (CRC covers 92)
    //   v2: trunk_index_offset@80 trunk_index_count@88
    //       n_layers@96 n_experts@100 align@104 header_crc32@108 (CRC covers 108)
    //   v3: trunk_index_offset@80 trunk_index_count@88
    //       compress_index_offset@96 compress_index_count@104
    //       n_layers@112 n_experts@116 align@120 header_crc32@124 (CRC covers 124)
    // The in-memory sb_ is the v3 struct; we re-map each version's on-disk
    // trailing fields into it and zero the fields a given version does not
    // carry, so the rest of the reader sees one consistent sb_.
    size_t crc_cover = 0;
    switch (version) {
        case 1: crc_cover = 92;  break;
        case 2: crc_cover = 108; break;
        default: crc_cover = 124; break;  // v3 == kStrataHeaderCrcOffset
    }
    {
        uint32_t stored = 0;
        std::memcpy(&stored, raw + crc_cover, sizeof(stored));
        const uint32_t computed = strata_crc32(raw, crc_cover);
        if (stored != computed) {
            log_warn("StrataReader: header CRC mismatch in '" + path + "'");
            return false;
        }
    }

    // Normalize the trailing fields from the version's on-disk offsets into the
    // v3 in-memory struct. The common prefix [0, 80) was already copied by the
    // std::memcpy(&sb_, raw, sizeof(sb_)) above.
    sb_.version = version;
    if (version == 1) {
        sb_.trunk_index_offset    = 0;
        sb_.trunk_index_count     = 0;
        sb_.compress_index_offset = 0;
        sb_.compress_index_count  = 0;
        std::memcpy(&sb_.n_layers,     raw + 80, sizeof(sb_.n_layers));
        std::memcpy(&sb_.n_experts,    raw + 84, sizeof(sb_.n_experts));
        std::memcpy(&sb_.align,        raw + 88, sizeof(sb_.align));
        std::memcpy(&sb_.header_crc32, raw + 92, sizeof(sb_.header_crc32));
    } else if (version == 2) {
        std::memcpy(&sb_.trunk_index_offset, raw + 80, sizeof(uint64_t));
        std::memcpy(&sb_.trunk_index_count,  raw + 88, sizeof(uint64_t));
        sb_.compress_index_offset = 0;
        sb_.compress_index_count  = 0;
        std::memcpy(&sb_.n_layers,     raw + 96,  sizeof(sb_.n_layers));
        std::memcpy(&sb_.n_experts,    raw + 100, sizeof(sb_.n_experts));
        std::memcpy(&sb_.align,        raw + 104, sizeof(sb_.align));
        std::memcpy(&sb_.header_crc32, raw + 108, sizeof(sb_.header_crc32));
    } else {
        // v3: the struct layout matches the on-disk layout exactly, so the
        // bulk memcpy above already placed every field. (Re-copy defensively
        // for clarity / in case the struct ever gains padding.)
        std::memcpy(&sb_.trunk_index_offset,    raw + 80,  sizeof(uint64_t));
        std::memcpy(&sb_.trunk_index_count,     raw + 88,  sizeof(uint64_t));
        std::memcpy(&sb_.compress_index_offset, raw + 96,  sizeof(uint64_t));
        std::memcpy(&sb_.compress_index_count,  raw + 104, sizeof(uint64_t));
        std::memcpy(&sb_.n_layers,     raw + 112, sizeof(sb_.n_layers));
        std::memcpy(&sb_.n_experts,    raw + 116, sizeof(sb_.n_experts));
        std::memcpy(&sb_.align,        raw + 120, sizeof(sb_.align));
        std::memcpy(&sb_.header_crc32, raw + 124, sizeof(sb_.header_crc32));
    }

    if (sb_.align != static_cast<uint32_t>(kIoAlignment)) {
        log_warn("StrataReader: alignment mismatch in '" + path + "'");
        return false;
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

    // --- v2 trunk index (optional, FEAT-002) -------------------------------
    // Present only when the file is v2+, the kStrataFlagTrunkIndexed flag is
    // set, and trunk_index_count>0. A v1 file leaves trunk_index_ empty, so the
    // engine falls back to a fully-resident trunk (byte-identical to before).
    if (sb_.version >= 2 && (sb_.flags & kStrataFlagTrunkIndexed) != 0 &&
        sb_.trunk_index_count > 0) {
        const uint64_t trunk_index_bytes =
            sb_.trunk_index_count * sizeof(TrunkIndexEntry);
        if (sb_.trunk_index_offset + trunk_index_bytes > file_.size()) {
            log_warn("StrataReader: invalid trunk index region in '" + path +
                     "'");
            return false;
        }
        trunk_index_.resize(static_cast<size_t>(sb_.trunk_index_count));
        if (file_.read_at(trunk_index_.data(),
                          static_cast<size_t>(trunk_index_bytes),
                          sb_.trunk_index_offset) !=
            static_cast<int64_t>(trunk_index_bytes)) {
            log_warn("StrataReader: failed to read trunk index from '" + path +
                     "'");
            trunk_index_.clear();
            return false;
        }
        trunk_map_.reserve(trunk_index_.size());
        for (size_t i = 0; i < trunk_index_.size(); ++i) {
            // The name is a fixed NUL-padded char array; strnlen bounds the
            // read so a non-terminated record cannot overrun the field.
            const TrunkIndexEntry &e = trunk_index_[i];
            const size_t nlen = ::strnlen(e.name, sizeof(e.name));
            trunk_map_[std::string(e.name, nlen)] = i;
        }
    }

    // --- v3 compression index (optional, FEAT-002 'compress what streams') --
    // Present only when the file is v3, the kStrataFlagExpertsCompressed flag is
    // set, and compress_index_count>0. The array is 1:1 with index_ by position.
    // A v1/v2 file (or a v3 file with no compressed blob) leaves compress_
    // empty, so every expert blob is streamed verbatim (byte-identical to the
    // pre-compression path).
    if (sb_.version >= 3 && (sb_.flags & kStrataFlagExpertsCompressed) != 0 &&
        sb_.compress_index_count > 0) {
        if (sb_.compress_index_count != sb_.index_count) {
            log_warn("StrataReader: compression index count != expert index "
                     "count in '" + path + "'");
            return false;
        }
        const uint64_t compress_index_bytes =
            sb_.compress_index_count * sizeof(ExpertBlobCompression);
        if (sb_.compress_index_offset + compress_index_bytes > file_.size()) {
            log_warn("StrataReader: invalid compression index region in '" +
                     path + "'");
            return false;
        }
        compress_.resize(static_cast<size_t>(sb_.compress_index_count));
        if (file_.read_at(compress_.data(),
                          static_cast<size_t>(compress_index_bytes),
                          sb_.compress_index_offset) !=
            static_cast<int64_t>(compress_index_bytes)) {
            log_warn("StrataReader: failed to read compression index from '" +
                     path + "'");
            compress_.clear();
            return false;
        }
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
    auto it = map_.find(pack_key(layer, expert));
    if (it == map_.end()) return -1;
    const ExpertIndexEntry &e = index_[it->second];

    uint64_t offset = 0;
    uint64_t length = 0;
    if (!slice_range(layer, expert, kind, offset, length)) return -1;

    const ExpertBlobCompression *ci = compression_at(it->second);
    if (ci != nullptr && ci->codec != static_cast<uint32_t>(StrataCodec::kNone)) {
        // COMPRESSED per-kind path (the fallback the engine uses only for a
        // fused layout; the common path is read_bundle). Read the whole
        // compressed blob into a PER-CALL-LOCAL scratch, decompress the whole
        // blob into another per-call buffer, then copy the requested kind's
        // [slice_rel, slice_rel+length) range into dst. No shared mutable
        // state, so concurrent compute+prefetch calls are race-free.
        const uint64_t slice_rel_off = slice_rel(e, kind);
        if (slice_rel_off + length > ci->uncompressed_length) return -1;
        if (ci->uncompressed_length != e.blob_length) return -1;

        read_calls_.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint8_t> comp(static_cast<size_t>(ci->compressed_length));
        const int64_t cn = file_.read_at(comp.data(), comp.size(),
                                         e.blob_offset);
        if (cn != static_cast<int64_t>(ci->compressed_length)) return -1;

        std::vector<uint8_t> raw(static_cast<size_t>(ci->uncompressed_length));
        const int64_t dn = lz_decompress(comp.data(), comp.size(), raw.data(),
                                         raw.size());
        if (dn != static_cast<int64_t>(ci->uncompressed_length)) return -1;

        std::memcpy(dst, raw.data() + slice_rel_off,
                    static_cast<size_t>(length));
        // Count the COMPRESSED bytes read, but only ONCE per blob so three
        // per-kind calls do not treble-count: attribute them to the gate (the
        // first slice, rel == 0). The up/down calls count 0 compressed bytes.
        if (slice_rel_off == 0) {
            streamed_bytes_.fetch_add(ci->compressed_length,
                                      std::memory_order_relaxed);
        }
        return static_cast<int64_t>(length);
    }

    // VERBATIM path (unchanged): one aligned read of the requested slice.
    // Count every underlying read_at on the expert path (one per slice here),
    // so a test can observe the 3:1 reduction read_bundle gives.
    read_calls_.fetch_add(1, std::memory_order_relaxed);
    const int64_t n = file_.read_at(dst, static_cast<size_t>(length), offset);
    // Count only successful expert-slice reads toward the per-token streamed
    // total. read_at() (the one-time trunk fill in Engine::load_strata_weights)
    // is intentionally NOT counted here, so streamed_bytes() reflects per-token
    // expert streaming rather than the trunk load.
    if (n > 0) {
        streamed_bytes_.fetch_add(static_cast<uint64_t>(n),
                                  std::memory_order_relaxed);
    }
    return n;
}

int64_t StrataReader::read_bundle(uint32_t layer, uint32_t expert,
                                  void *dst) const {
    const ExpertIndexEntry *e = find(layer, expert);
    if (e == nullptr) return -1;
    // A FUSED gate_up bundle (gate_rel == up_rel) has a two-slice layout
    // (gate_up fused, then down) that does NOT match the engine's three-slice
    // gate|up|down slot buffer. Refuse so the caller falls back to the per-kind
    // read_blob path (which handles the fused case). The non-fused path below
    // is the common MoE arch (Mixtral et al.).
    if (e->up_rel == e->gate_rel) return -1;
    // The three slices must be tightly packed in gate|up|down order starting at
    // the blob start and covering the whole blob_length, so that ONE read of
    // [blob_offset, blob_offset+blob_length) yields the exact same bytes as the
    // three read_blob calls concatenated. The packer guarantees this (gate_rel
    // == 0, up_rel == gate_len, down_rel == gate_len+up_len, blob_length ==
    // gate_len+up_len+down_len). Verify defensively; refuse (fall back) if any
    // assumption fails so a malformed/unexpected layout can never silently
    // corrupt the slot.
    if (e->gate_rel != 0 || e->up_rel <= e->gate_rel ||
        e->down_rel <= e->up_rel || e->down_rel >= e->blob_length) {
        return -1;
    }

    // COMPRESSED path (FEAT-002 'compress what streams'). Read the COMPRESSED
    // bytes from blob_offset into a PER-CALL-LOCAL scratch, decompress straight
    // into the caller's dst (capacity blob_length), and return blob_length (the
    // UNCOMPRESSED length) so the engine's `nb == gb+ub+db` size check passes
    // unchanged. streamed_bytes_ counts the COMPRESSED bytes actually read -
    // the WIN. No shared mutable reader state: all scratch is per-call local, so
    // the compute thread and the async prefetch worker may call concurrently.
    auto it = map_.find(pack_key(layer, expert));
    const size_t index_pos = (it != map_.end()) ? it->second : compress_.size();
    const ExpertBlobCompression *ci = compression_at(index_pos);
    if (ci != nullptr && ci->codec != static_cast<uint32_t>(StrataCodec::kNone)) {
        if (ci->uncompressed_length != e->blob_length) return -1;
        read_calls_.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint8_t> comp(static_cast<size_t>(ci->compressed_length));
        const int64_t cn = file_.read_at(comp.data(), comp.size(),
                                         e->blob_offset);
        if (cn != static_cast<int64_t>(ci->compressed_length)) return -1;
        const int64_t dn = lz_decompress(comp.data(), comp.size(), dst,
                                          static_cast<size_t>(e->blob_length));
        if (dn != static_cast<int64_t>(e->blob_length)) return -1;
        streamed_bytes_.fetch_add(ci->compressed_length,
                                  std::memory_order_relaxed);
        return static_cast<int64_t>(e->blob_length);
    }

    read_calls_.fetch_add(1, std::memory_order_relaxed);
    const int64_t n = file_.read_at(dst, static_cast<size_t>(e->blob_length),
                                    e->blob_offset);
    // Count the coalesced read toward streamed_bytes_ ONLY on a FULL-length
    // read (n == blob_length). The engine treats any short read as a failure
    // (nb != gb+ub+db) and falls back to the three read_blob calls, each of
    // which counts its own slice; counting a partial coalesced read here too
    // would double-count those partial bytes. Making this all-or-nothing keeps
    // streamed_bytes() byte-accurate on both the success and the fallback path
    // (header comment: "counts the bundle's bytes exactly once").
    if (n == static_cast<int64_t>(e->blob_length)) {
        streamed_bytes_.fetch_add(static_cast<uint64_t>(n),
                                  std::memory_order_relaxed);
    }
    return n;
}

int64_t StrataReader::read_at(void *dst, uint64_t len, uint64_t offset) const {
    return file_.read_at(dst, static_cast<size_t>(len), offset);
}

bool StrataReader::trunk_slice_range(const std::string &name, uint64_t &offset,
                                     uint64_t &length) const {
    auto it = trunk_map_.find(name);
    if (it == trunk_map_.end()) return false;
    const TrunkIndexEntry &e = trunk_index_[it->second];
    // The index stores the trunk-RELATIVE offset; the absolute file offset is
    // the trunk block start plus that relative offset.
    offset = sb_.trunk_offset + e.rel_offset;
    length = e.length;
    return true;
}

int64_t StrataReader::read_trunk_tensor(const std::string &name,
                                        void *dst) const {
    uint64_t offset = 0;
    uint64_t length = 0;
    if (!trunk_slice_range(name, offset, length)) return -1;
    const int64_t n = file_.read_at(dst, static_cast<size_t>(length), offset);
    // Count successful trunk reads toward the SEPARATE trunk-streaming total
    // (not streamed_bytes(), which stays expert-only for the bench harness).
    if (n > 0) {
        streamed_trunk_bytes_.fetch_add(static_cast<uint64_t>(n),
                                        std::memory_order_relaxed);
    }
    return n;
}

}  // namespace sf

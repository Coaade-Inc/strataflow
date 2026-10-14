// FEAT-002 'compress what streams' gate (lossless, opt-in on-disk compression
// of the EXPERT bundles that cross the SSD per token). The packer can store
// each (layer,expert) gate|up|down bundle LZ-compressed on disk; the
// StrataReader decompresses it TRANSPARENTLY at the read_bundle/read_blob seam
// so the engine's compute sees byte-identical expert bytes while FEWER
// (compressed) bytes are read from disk. The default (no --compress) stays
// byte-for-byte identical to the pre-compression packer.
//
// This gate asserts, through the REAL StrataReader and the REAL engine on the
// Mixtral-geometry legacy fixture packed TWICE (uncompressed and --compress lz):
//   (a) ORACLE GATE: the engine decode on the compressed .strata is BYTE-
//       IDENTICAL (greedy sequence identical, per-step logit max|delta|==0.0) to
//       the uncompressed .strata, and both match the libllama oracle on the
//       matched stacked GGUF within the soft quant bound. (Guarded on
//       STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF_STACKED, like the sibling real-path
//       tests; the reader/region half below already stands without it.)
//   (b) the compressed expert_region_size (and total on-disk file size) is <=
//       the uncompressed one, and STRICTLY < when the fixture compresses at all
//       (reported honestly via printf - already-quantized K-quant experts are
//       high-entropy, so the ratio is typically ~1.0 on the random-weight
//       fixture; the lever still proves lossless + opt-in + a strict win when
//       any blob compresses).
//   (c) read_bundle on the compressed reader yields DECOMPRESSED bytes that are
//       memcmp-identical to the uncompressed read_bundle bytes AND to the three
//       read_blob (gate/up/down) calls concatenated, while advancing
//       streamed_bytes() by NO MORE than the uncompressed reader (strictly less
//       when the blob actually compressed) - the WIN is fewer bytes read.
//   (d) at the codec level lz_decompress(lz_compress(blob)) == blob byte-exact
//       on a real expert blob.
//
// GUARDED on STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF; skips cleanly when unset.
// Generate + run locally:
//   build/release/bin/make-legacy-moe-gguf FIX.gguf FIX.stacked.gguf
//       1024 256 4 8 q4_k 4096 2 8
//   STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF=FIX.gguf
//     STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF_STACKED=FIX.stacked.gguf
//     ctest -R test_compressed_strata --output-on-failure
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/lz_codec.h"
#include "model/engine/engine.h"
#include "test_util.h"
#include "tws/strata_file.h"
#include "tws/strata_format.h"
#include "tws/strata_pack.h"
#include "tws/stream_buft.h"  // ExpertTensorKind

using namespace sf;

static int32_t argmax(const std::vector<float> &v) {
    int32_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] > best_v) {
            best_v = v[i];
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

struct DecodeRun {
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> step_logits;
    bool ok = false;
};

// Greedy-decode `prompt_ids` + n_generate tokens through a freshly-loaded
// engine on `engine_path`, mirroring test_target_ram_modes_real_path's
// decode_with_plan but on the DEFAULT full-resident levers (0,0,0) - the modes
// are irrelevant here; only the expert BYTE SOURCE (compressed vs verbatim
// .strata) differs between the two runs, and it must not change the output.
static DecodeRun decode_default(const char *engine_path,
                                const std::vector<int32_t> &prompt_ids,
                                int n_generate) {
    DecodeRun r;
    if (prompt_ids.empty()) return r;
    std::unique_ptr<engine::Engine> eng = engine::Engine::load(engine_path);
    if (eng == nullptr) return r;
    eng->reset_kv();

    int32_t pos = 0;
    std::vector<float> logits;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        if (!eng->forward(prompt_ids[i], pos, logits)) return r;
        ++pos;
    }
    for (int g = 0; g < n_generate; ++g) {
        if (logits.empty()) return r;
        r.step_logits.push_back(logits);
        const int32_t next = argmax(logits);
        r.tokens.push_back(next);
        if (!eng->forward(next, pos, logits)) return r;
        ++pos;
    }
    r.ok = true;
    return r;
}

static double max_step_delta(const std::vector<std::vector<float>> &a,
                             const std::vector<std::vector<float>> &b) {
    double m = 0.0;
    const size_t steps = a.size() < b.size() ? a.size() : b.size();
    for (size_t s = 0; s < steps; ++s) {
        const size_t n = a[s].size() < b[s].size() ? a[s].size() : b[s].size();
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(a[s][i]) -
                                       static_cast<double>(b[s][i]));
            if (d > m) m = d;
        }
    }
    return m;
}

static void test_compressed_strata() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF] ");
        return;
    }
    const std::string gguf_path = gguf;
    const std::string plain_path = gguf_path + ".plain.strata";
    const std::string lz_path = gguf_path + ".lz.strata";

    // Pack the SAME fixture twice: verbatim (codec kNone) and --compress lz.
    std::string err;
    const bool packed_plain =
        pack_gguf_to_strata(gguf_path, plain_path, &err, StrataCodec::kNone);
    CHECK(packed_plain);
    if (!packed_plain) {
        std::printf("[plain pack failed: %s] ", err.c_str());
        return;
    }
    const bool packed_lz =
        pack_gguf_to_strata(gguf_path, lz_path, &err, StrataCodec::kLz);
    CHECK(packed_lz);
    if (!packed_lz) {
        std::printf("[lz pack failed: %s] ", err.c_str());
        std::remove(plain_path.c_str());
        return;
    }

    StrataReader plain;
    StrataReader lz;
    CHECK(plain.open(plain_path));
    CHECK(lz.open(lz_path));
    if (!plain.is_open() || !lz.is_open()) {
        std::remove(plain_path.c_str());
        std::remove(lz_path.c_str());
        return;
    }

    const StrataSuperblock &psb = plain.superblock();
    const StrataSuperblock &lsb = lz.superblock();
    CHECK(psb.n_layers > 0u);
    CHECK(psb.n_experts > 0u);
    // Same geometry on both packs.
    CHECK_EQ(psb.n_layers, lsb.n_layers);
    CHECK_EQ(psb.n_experts, lsb.n_experts);
    CHECK_EQ(psb.index_count, lsb.index_count);

    // The plain pack must be byte-identical to the pre-compression packer: v2
    // layout, no compression flag/region.
    CHECK_EQ(psb.version, 2u);
    CHECK((psb.flags & kStrataFlagExpertsCompressed) == 0u);
    CHECK_EQ(psb.compress_index_count, static_cast<uint64_t>(0));

    const bool any_compressed =
        (lsb.flags & kStrataFlagExpertsCompressed) != 0u &&
        lsb.compress_index_count > 0;
    if (any_compressed) {
        // A compressed pack is v3 with a compression index 1:1 with the experts.
        CHECK_EQ(lsb.version, 3u);
        CHECK_EQ(lsb.compress_index_count, lsb.index_count);
    } else {
        // Incompressible fixture: the lz pack fell back to verbatim for EVERY
        // blob, so it is byte-identical to the plain pack (v2, no region).
        CHECK_EQ(lsb.version, 2u);
    }

    // (b) The compressed expert region (and total file) is never larger, and
    // strictly smaller when any blob compressed.
    const double ratio = lsb.expert_region_size > 0
        ? static_cast<double>(psb.expert_region_size) /
              static_cast<double>(lsb.expert_region_size)
        : 1.0;
    CHECK(lsb.expert_region_size <= psb.expert_region_size);
    if (any_compressed) {
        CHECK(lsb.expert_region_size < psb.expert_region_size);
    }

    // (c) + (d): walk layer 0's experts. For each, read the bundle both ways and
    // prove byte-identity; verify the three read_blob calls concatenated match;
    // compare streamed_bytes deltas; and round-trip the codec on the real blob.
    const uint32_t layer = 0;
    uint64_t plain_streamed_total = 0;
    uint64_t lz_streamed_total = 0;
    uint32_t checked = 0;
    for (uint32_t e = 0; e < psb.n_experts; ++e) {
        const ExpertIndexEntry *pe = plain.find(layer, e);
        const ExpertIndexEntry *le = lz.find(layer, e);
        CHECK(pe != nullptr);
        CHECK(le != nullptr);
        if (pe == nullptr || le == nullptr) continue;
        // The UNCOMPRESSED layout (blob_length + slice rels) is identical in
        // both packs; only the on-disk bytes differ.
        CHECK_EQ(pe->blob_length, le->blob_length);
        CHECK_EQ(pe->gate_rel, le->gate_rel);
        CHECK_EQ(pe->up_rel, le->up_rel);
        CHECK_EQ(pe->down_rel, le->down_rel);
        const uint64_t blen = pe->blob_length;
        CHECK(blen > 0u);

        // read_bundle on the plain reader.
        std::vector<uint8_t> plain_bundle(static_cast<size_t>(blen));
        const uint64_t p_s0 = plain.streamed_bytes();
        const int64_t pn = plain.read_bundle(layer, e, plain_bundle.data());
        const uint64_t p_streamed = plain.streamed_bytes() - p_s0;
        CHECK_EQ(pn, static_cast<int64_t>(blen));
        plain_streamed_total += p_streamed;

        // read_bundle on the lz reader: returns the UNCOMPRESSED length and
        // yields byte-identical bytes.
        std::vector<uint8_t> lz_bundle(static_cast<size_t>(blen));
        const uint64_t l_s0 = lz.streamed_bytes();
        const int64_t ln = lz.read_bundle(layer, e, lz_bundle.data());
        const uint64_t l_streamed = lz.streamed_bytes() - l_s0;
        CHECK_EQ(ln, static_cast<int64_t>(blen));
        lz_streamed_total += l_streamed;

        // (c) byte-identity between the two read_bundle results.
        CHECK_EQ(std::memcmp(plain_bundle.data(), lz_bundle.data(),
                             static_cast<size_t>(blen)),
                 0);

        // streamed_bytes: the lz reader reads NO MORE than the plain reader for
        // this blob (strictly fewer when the blob actually compressed).
        CHECK(l_streamed <= p_streamed);

        // The three read_blob (gate/up/down) calls concatenated must equal the
        // coalesced bundle, on the lz reader (exercises the compressed per-kind
        // decode seam when the blob is compressed).
        uint64_t goff = 0, glen = 0, uoff = 0, ulen = 0, doff = 0, dlen = 0;
        CHECK(lz.slice_range(layer, e, ExpertTensorKind::kGate, goff, glen));
        CHECK(lz.slice_range(layer, e, ExpertTensorKind::kUp, uoff, ulen));
        CHECK(lz.slice_range(layer, e, ExpertTensorKind::kDown, doff, dlen));
        CHECK_EQ(glen + ulen + dlen, blen);
        std::vector<uint8_t> via_blobs(static_cast<size_t>(blen));
        const int64_t bg = lz.read_blob(layer, e, ExpertTensorKind::kGate,
                                        via_blobs.data());
        const int64_t bu = lz.read_blob(layer, e, ExpertTensorKind::kUp,
                                        via_blobs.data() + glen);
        const int64_t bd = lz.read_blob(layer, e, ExpertTensorKind::kDown,
                                        via_blobs.data() + glen + ulen);
        CHECK_EQ(bg, static_cast<int64_t>(glen));
        CHECK_EQ(bu, static_cast<int64_t>(ulen));
        CHECK_EQ(bd, static_cast<int64_t>(dlen));
        CHECK_EQ(std::memcmp(via_blobs.data(), lz_bundle.data(),
                             static_cast<size_t>(blen)),
                 0);

        // (d) codec round-trip on the real blob.
        std::vector<uint8_t> comp(lz_compress_bound(static_cast<size_t>(blen)));
        const int64_t clen = lz_compress(lz_bundle.data(),
                                         static_cast<size_t>(blen), comp.data(),
                                         comp.size());
        if (clen >= 0) {
            std::vector<uint8_t> round(static_cast<size_t>(blen));
            const int64_t dn = lz_decompress(comp.data(),
                                             static_cast<size_t>(clen),
                                             round.data(), round.size());
            CHECK_EQ(dn, static_cast<int64_t>(blen));
            CHECK_EQ(std::memcmp(round.data(), lz_bundle.data(),
                                 static_cast<size_t>(blen)),
                     0);
        }
        ++checked;
    }
    CHECK_EQ(checked, psb.n_experts);
    // Over the whole layer the lz reader streams no more than the plain reader,
    // strictly fewer when the fixture compressed.
    CHECK(lz_streamed_total <= plain_streamed_total);
    if (any_compressed) {
        CHECK(lz_streamed_total < plain_streamed_total);
    }

    std::printf("[compress-what-streams: ratio=%.4f expert_region plain=%lluB "
                "lz=%lluB | layer0 streamed plain=%lluB lz=%lluB | "
                "compressed=%s] ",
                ratio,
                static_cast<unsigned long long>(psb.expert_region_size),
                static_cast<unsigned long long>(lsb.expert_region_size),
                static_cast<unsigned long long>(plain_streamed_total),
                static_cast<unsigned long long>(lz_streamed_total),
                any_compressed ? "yes" : "no (incompressible fixture)");

    // (a) ORACLE GATE: engine decode on the compressed .strata is byte-identical
    // to the uncompressed .strata and matches the libllama oracle. Needs the
    // matched stacked GGUF for the tokenizer + oracle; skip cleanly when unset
    // (the reader/region half above already stands).
    const char *stacked = std::getenv("STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF_STACKED");
    if (stacked == nullptr || stacked[0] == '\0') {
        std::printf("[decode half skipped: set "
                    "STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF_STACKED] ");
        std::remove(plain_path.c_str());
        std::remove(lz_path.c_str());
        return;
    }

    const std::string prompt = "The city of";
    const int n_generate = 6;
    std::vector<int32_t> prompt_ids;
    CHECK(engine::vocab_tokenize(stacked, prompt, prompt_ids));
    CHECK(!prompt_ids.empty());

    DecodeRun plain_run = decode_default(plain_path.c_str(), prompt_ids,
                                         n_generate);
    DecodeRun lz_run = decode_default(lz_path.c_str(), prompt_ids, n_generate);
    CHECK(plain_run.ok);
    CHECK(lz_run.ok);
    if (!plain_run.ok || !lz_run.ok) {
        std::remove(plain_path.c_str());
        std::remove(lz_path.c_str());
        return;
    }

    std::vector<int32_t> oracle_tokens;
    std::vector<std::vector<float>> oracle_step_logits;
    const bool oracle_ok = engine::run_oracle_sequence(
        stacked, prompt, n_generate, oracle_tokens, &oracle_step_logits);
    CHECK(oracle_ok);

    bool seq_match = plain_run.tokens.size() == lz_run.tokens.size() &&
                     plain_run.tokens.size() == oracle_tokens.size();
    for (size_t i = 0; i < plain_run.tokens.size() &&
                       i < lz_run.tokens.size() && i < oracle_tokens.size();
         ++i) {
        CHECK_EQ(plain_run.tokens[i], lz_run.tokens[i]);
        CHECK_EQ(plain_run.tokens[i], oracle_tokens[i]);
        if (plain_run.tokens[i] != lz_run.tokens[i] ||
            plain_run.tokens[i] != oracle_tokens[i]) {
            seq_match = false;
        }
    }

    const double max_pl = max_step_delta(plain_run.step_logits,
                                         lz_run.step_logits);
    const double max_po = max_step_delta(plain_run.step_logits,
                                         oracle_step_logits);
    std::printf("[oracle gate: seq match=%s plain-vs-lz|delta|=%.3e "
                "plain-vs-oracle|delta|=%.3e] ",
                seq_match ? "yes" : "no", max_pl, max_po);

    CHECK(seq_match);
    // HARD: the compressed .strata decodes BIT-IDENTICALLY to the uncompressed
    // one (lossless transport; the engine saw the same expert bytes).
    CHECK(max_pl == 0.0);
    // Soft quant-noise bound vs the libllama oracle (same 1e-1 bound the sibling
    // real-path tests use for this random-weight q4_k fixture).
    CHECK(max_po < 1e-1);

    std::remove(plain_path.c_str());
    std::remove(lz_path.c_str());
}

// Directly exercise the StrataReader DECOMPRESS SEAM on a .strata that is
// GUARANTEED to compress, independent of the (high-entropy, often
// incompressible) K-quant fixture. The already-quantized expert bytes in the
// real fixture frequently do not compress with the fast byte-level LZ, so the
// test above honestly reports ratio ~1.0 and cannot demonstrate the fewer-
// bytes-read win. This test fills one expert bundle with HIGHLY-compressible
// bytes, writes a minimal valid v3 .strata (one layer, one expert) with the LZ-
// compressed blob + a compression index, and drives the REAL StrataReader to
// prove: read_bundle returns the UNCOMPRESSED length with byte-identical bytes,
// read_blob (per-kind) decompresses the same bytes, and streamed_bytes() counts
// the strictly-smaller COMPRESSED size - the lever's win, proven end to end on
// the reader without needing compressible model weights.
static void test_compressed_seam_win() {
    // Build an uncompressed bundle: gate|up|down of compressible content.
    const size_t gate_len = 4096, up_len = 4096, down_len = 8192;
    const size_t blob_len = gate_len + up_len + down_len;
    std::vector<uint8_t> raw(blob_len);
    // Compressible but not trivial: long runs with periodic variation.
    for (size_t i = 0; i < blob_len; ++i) {
        raw[i] = static_cast<uint8_t>((i / 64) & 0x0F);
    }

    std::vector<uint8_t> comp(lz_compress_bound(blob_len));
    const int64_t clen = lz_compress(raw.data(), blob_len, comp.data(),
                                     comp.size());
    CHECK(clen > 0);
    CHECK(static_cast<uint64_t>(clen) < blob_len);  // genuinely compressible
    if (clen <= 0 || static_cast<uint64_t>(clen) >= blob_len) return;

    // Lay out a minimal v3 .strata by hand (same field offsets the packer and
    // reader agree on). Regions: [superblock 128][meta (empty)][index]
    // [blob (compressed)][compression index]. 4 KiB-align the blob start like
    // the packer. We do NOT embed GGUF metadata (gguf_meta_size==0 would fail
    // the reader's metadata check), so give it a 1-byte placeholder region the
    // reader will happily read; metadata content is irrelevant to the seam test
    // because we call the reader APIs (open+read_bundle/read_blob) directly, not
    // the engine. The reader requires gguf_meta_size>0, so use a tiny stub.
    const uint32_t kAlign = static_cast<uint32_t>(kIoAlignment);
    auto align_up_local = [&](uint64_t v) {
        return (v + kAlign - 1) & ~static_cast<uint64_t>(kAlign - 1);
    };

    const uint64_t meta_off = 128;
    const uint64_t meta_size = 16;  // stub, non-zero
    const uint64_t index_off = meta_off + meta_size;
    const uint64_t index_count = 1;
    const uint64_t index_bytes = index_count * sizeof(ExpertIndexEntry);
    const uint64_t blob_off = align_up_local(index_off + index_bytes);
    const uint64_t region_off = blob_off;  // single blob region start
    const uint64_t cindex_off =
        align_up_local(blob_off + static_cast<uint64_t>(clen));
    const uint64_t cindex_count = 1;
    const uint64_t cindex_bytes = cindex_count * sizeof(ExpertBlobCompression);
    const uint64_t file_size = cindex_off + cindex_bytes;

    std::vector<uint8_t> file(static_cast<size_t>(file_size), 0);

    // Expert index entry (UNCOMPRESSED layout).
    ExpertIndexEntry ent{};
    ent.layer = 0;
    ent.expert = 0;
    ent.blob_offset = blob_off;
    ent.blob_length = blob_len;
    ent.gate_rel = 0;
    ent.up_rel = gate_len;
    ent.down_rel = gate_len + up_len;
    std::memcpy(file.data() + index_off, &ent, sizeof(ent));

    // Compressed blob bytes on disk.
    std::memcpy(file.data() + blob_off, comp.data(), static_cast<size_t>(clen));

    // Compression index record.
    ExpertBlobCompression ci{};
    ci.codec = static_cast<uint32_t>(StrataCodec::kLz);
    ci.compressed_length = static_cast<uint64_t>(clen);
    ci.uncompressed_length = blob_len;
    std::memcpy(file.data() + cindex_off, &ci, sizeof(ci));

    // v3 superblock (write the on-disk v3 layout directly).
    std::memcpy(file.data() + 0, kStrataMagic, sizeof(kStrataMagic));
    const uint32_t version = 3;
    const uint32_t flags = kStrataFlagExpertsBundled | kStrataFlagIndexSorted |
                           kStrataFlagExpertsCompressed;
    std::memcpy(file.data() + 8, &version, 4);
    std::memcpy(file.data() + 12, &flags, 4);
    std::memcpy(file.data() + 16, &meta_off, 8);
    std::memcpy(file.data() + 24, &meta_size, 8);
    std::memcpy(file.data() + 32, &index_off, 8);
    std::memcpy(file.data() + 40, &index_count, 8);
    const uint64_t trunk_off = 0, trunk_size = 0;
    std::memcpy(file.data() + 48, &trunk_off, 8);
    std::memcpy(file.data() + 56, &trunk_size, 8);
    std::memcpy(file.data() + 64, &region_off, 8);
    const uint64_t region_size = cindex_off - region_off;
    std::memcpy(file.data() + 72, &region_size, 8);
    const uint64_t trunk_index_off = 0, trunk_index_count = 0;
    std::memcpy(file.data() + 80, &trunk_index_off, 8);
    std::memcpy(file.data() + 88, &trunk_index_count, 8);
    std::memcpy(file.data() + 96, &cindex_off, 8);
    std::memcpy(file.data() + 104, &cindex_count, 8);
    const uint32_t n_layers = 1, n_experts = 1;
    std::memcpy(file.data() + 112, &n_layers, 4);
    std::memcpy(file.data() + 116, &n_experts, 4);
    std::memcpy(file.data() + 120, &kAlign, 4);
    const uint32_t crc = strata_crc32(file.data(), 124);
    std::memcpy(file.data() + 124, &crc, 4);

    const std::string path = "./test_compressed_seam_win.strata";
    FILE *f = std::fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (f == nullptr) return;
    const size_t wrote = std::fwrite(file.data(), 1, file.size(), f);
    std::fclose(f);
    CHECK_EQ(wrote, file.size());

    StrataReader r;
    CHECK(r.open(path));
    if (!r.is_open()) {
        std::remove(path.c_str());
        return;
    }

    // read_bundle: returns the UNCOMPRESSED length, bytes decompress byte-exact,
    // and streamed_bytes counts the COMPRESSED (strictly smaller) size.
    std::vector<uint8_t> got(blob_len);
    const uint64_t s0 = r.streamed_bytes();
    const int64_t n = r.read_bundle(0, 0, got.data());
    const uint64_t streamed = r.streamed_bytes() - s0;
    CHECK_EQ(n, static_cast<int64_t>(blob_len));
    CHECK_EQ(std::memcmp(got.data(), raw.data(), blob_len), 0);
    CHECK_EQ(streamed, static_cast<uint64_t>(clen));
    CHECK(streamed < blob_len);  // THE WIN: fewer bytes crossed the SSD

    // read_blob per-kind: decompresses the whole blob and copies each slice;
    // concatenated they equal the uncompressed bundle, and streamed_bytes is
    // counted once (on the gate read) as the compressed size.
    std::vector<uint8_t> via_blobs(blob_len);
    const uint64_t s1 = r.streamed_bytes();
    const int64_t bg = r.read_blob(0, 0, ExpertTensorKind::kGate,
                                   via_blobs.data());
    const int64_t bu = r.read_blob(0, 0, ExpertTensorKind::kUp,
                                   via_blobs.data() + gate_len);
    const int64_t bd = r.read_blob(0, 0, ExpertTensorKind::kDown,
                                   via_blobs.data() + gate_len + up_len);
    const uint64_t streamed_blobs = r.streamed_bytes() - s1;
    CHECK_EQ(bg, static_cast<int64_t>(gate_len));
    CHECK_EQ(bu, static_cast<int64_t>(up_len));
    CHECK_EQ(bd, static_cast<int64_t>(down_len));
    CHECK_EQ(std::memcmp(via_blobs.data(), raw.data(), blob_len), 0);
    // Compressed bytes counted exactly once across the three per-kind reads.
    CHECK_EQ(streamed_blobs, static_cast<uint64_t>(clen));

    std::printf("[compressed seam win: blob=%zuB compressed=%lldB streamed=%lluB "
                "(read_bundle byte-exact, streamed==compressed<blob)] ",
                blob_len, static_cast<long long>(clen),
                static_cast<unsigned long long>(streamed));

    std::remove(path.c_str());
}

static void run_all() {
    RUN(test_compressed_strata);
    RUN(test_compressed_seam_win);
}

TEST_MAIN()

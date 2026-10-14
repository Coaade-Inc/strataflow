// FEAT-002 coalesced-expert-read gate (lossless batched I/O). The engine's
// per-token expert residency path used to issue THREE StrataReader::read_blob
// calls per (layer,expert) bundle - one each for the gate/up/down slices - even
// though those three slices are packed TIGHTLY and contiguously inside ONE
// 4 KiB-aligned bundle (blob_length) in gate|up|down order. StrataReader::
// read_bundle coalesces them into ONE underlying read_at over the whole bundle
// (3 reads -> 1), which raises effective bandwidth on the real I/O-bound box
// WITHOUT changing which bytes are read.
//
// This gate HARD-asserts through the REAL StrataReader (not a mock): it packs
// the Mixtral-geometry legacy fixture to a .strata, then for a layer's experts
// proves that
//   (1) read_bundle() returns BYTE-IDENTICAL bytes to the three read_blob
//       (kGate/kUp/kDown) calls concatenated (memcmp == 0),
//   (2) read_bundle issues exactly ONE underlying read_at (read_calls() delta
//       == 1) where the three read_blob calls issue THREE (delta == 3) - the
//       3:1 coalescing, and
//   (3) streamed_bytes() advances by the SAME total on both paths (the bundle's
//       bytes counted once, identical to the three slice reads summed).
//
// GUARDED on STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF (the same Mixtral-geometry
// fixture the real-path mode gate uses), so the suite stays green in CI where
// the fixture is not generated. Generate + run locally:
//   build/release/bin/make-legacy-moe-gguf FIX.gguf FIX.stacked.gguf
//       1024 256 32 8 q4_k 32768 2 8
//   STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF=FIX.gguf ctest -R test_coalesced_expert_read
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "test_util.h"
#include "tws/strata_file.h"
#include "tws/strata_format.h"
#include "tws/strata_pack.h"
#include "tws/stream_buft.h"  // ExpertTensorKind

using namespace sf;

static void test_read_bundle_byte_identical_and_coalesced() {
    const char *gguf = std::getenv("STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF");
    if (gguf == nullptr || gguf[0] == '\0') {
        std::printf("[skipped: set STRATAFLOW_TEST_MIXTRAL_GEOM_GGUF] ");
        return;
    }
    const std::string gguf_path = gguf;
    const std::string strata_path = gguf_path + ".coalesced.strata";

    std::string err;
    const bool packed = pack_gguf_to_strata(gguf_path, strata_path, &err);
    CHECK(packed);
    if (!packed) {
        std::printf("[pack failed: %s] ", err.c_str());
        return;
    }

    StrataReader reader;
    CHECK(reader.open(strata_path));
    if (!reader.is_open()) {
        std::remove(strata_path.c_str());
        return;
    }

    const StrataSuperblock &sb = reader.superblock();
    CHECK(sb.n_layers > 0u);
    CHECK(sb.n_experts > 0u);

    // Walk a layer's experts (layer 0), proving the coalesced read for each.
    const uint32_t layer = 0;
    uint32_t checked = 0;
    for (uint32_t e = 0; e < sb.n_experts; ++e) {
        const ExpertIndexEntry *ent = reader.find(layer, e);
        CHECK(ent != nullptr);
        if (ent == nullptr) continue;

        // Slice lengths from the real reader (gate|up|down, tightly packed).
        uint64_t goff = 0, glen = 0, uoff = 0, ulen = 0, doff = 0, dlen = 0;
        CHECK(reader.slice_range(layer, e, ExpertTensorKind::kGate, goff, glen));
        CHECK(reader.slice_range(layer, e, ExpertTensorKind::kUp, uoff, ulen));
        CHECK(reader.slice_range(layer, e, ExpertTensorKind::kDown, doff, dlen));
        const uint64_t total = glen + ulen + dlen;
        CHECK(total > 0u);
        // The three tightly-packed slices cover the whole bundle length.
        CHECK_EQ(total, ent->blob_length);

        // Path A: three per-kind read_blob calls into a gate|up|down buffer.
        std::vector<uint8_t> via_blobs(static_cast<size_t>(total));
        const uint64_t calls_before_blobs = reader.read_calls();
        const uint64_t streamed_before_blobs = reader.streamed_bytes();
        const int64_t ng = reader.read_blob(layer, e, ExpertTensorKind::kGate,
                                            via_blobs.data());
        const int64_t nu = reader.read_blob(layer, e, ExpertTensorKind::kUp,
                                            via_blobs.data() + glen);
        const int64_t nd = reader.read_blob(layer, e, ExpertTensorKind::kDown,
                                            via_blobs.data() + glen + ulen);
        CHECK_EQ(ng, static_cast<int64_t>(glen));
        CHECK_EQ(nu, static_cast<int64_t>(ulen));
        CHECK_EQ(nd, static_cast<int64_t>(dlen));
        const uint64_t blob_calls = reader.read_calls() - calls_before_blobs;
        const uint64_t blob_streamed =
            reader.streamed_bytes() - streamed_before_blobs;
        // Three per-kind reads == three underlying read_at calls.
        CHECK_EQ(blob_calls, static_cast<uint64_t>(3));
        CHECK_EQ(blob_streamed, total);

        // Path B: one coalesced read_bundle into a same-size buffer.
        std::vector<uint8_t> via_bundle(static_cast<size_t>(total));
        const uint64_t calls_before_bundle = reader.read_calls();
        const uint64_t streamed_before_bundle = reader.streamed_bytes();
        const int64_t nb = reader.read_bundle(layer, e, via_bundle.data());
        CHECK_EQ(nb, static_cast<int64_t>(total));
        const uint64_t bundle_calls = reader.read_calls() - calls_before_bundle;
        const uint64_t bundle_streamed =
            reader.streamed_bytes() - streamed_before_bundle;
        // ONE coalesced read == ONE underlying read_at (the 3:1 reduction).
        CHECK_EQ(bundle_calls, static_cast<uint64_t>(1));
        // Same bytes streamed total as the three per-kind reads.
        CHECK_EQ(bundle_streamed, total);

        // (1) BYTE-IDENTITY: the coalesced read yields exactly the same bytes
        // as the three read_blob calls concatenated.
        CHECK_EQ(std::memcmp(via_blobs.data(), via_bundle.data(),
                             static_cast<size_t>(total)),
                 0);
        ++checked;
    }
    CHECK(checked == sb.n_experts);

    std::printf("[coalesced read: layer %u, %u experts: read_bundle == "
                "3x read_blob byte-identical, 1 read_at vs 3 (3:1)] ",
                layer, checked);

    std::remove(strata_path.c_str());
}

static void run_all() {
    RUN(test_read_bundle_byte_identical_and_coalesced);
}

TEST_MAIN()

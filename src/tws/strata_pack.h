// strata-pack core: convert a GGUF checkpoint into the packed .strata format
// (docs/TASK5_DESIGN.md sections 2 and 5, docs/ENGINE_CORE_DESIGN.md section 5).
//
// The packer copies weight bytes VERBATIM: it never requantizes or reshapes, so
// a .strata decodes byte-identically to its source GGUF (the headline EC-4
// gate, docs/TASK5_DESIGN.md section 6.4). The implementation lives in the
// library so both the strata-pack CLI tool and the decode test can call it
// without shelling out.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <string>

#include "common/lz_codec.h"  // StrataCodec

namespace sf {

// Pack the GGUF at `in_path` into a .strata at `out_path`. Writes the 128-byte
// superblock, the verbatim embedded GGUF metadata blob, the expert index, the
// 4 KiB-aligned trunk block (all non-expert tensors), and the expert region
// (one 4 KiB-aligned blob per (layer,expert) bundling gate+up+down). Returns
// true on success; on failure returns false and, when `err` is non-null, sets
// it to a human-readable reason.
//
// `codec` ('compress what streams', FEAT-002) selects OPTIONAL lossless
// per-expert-blob compression. StrataCodec::kNone (the default) writes the
// expert bytes VERBATIM and the output is byte-for-byte identical to the
// pre-compression packer. StrataCodec::kLz compresses each expert bundle with
// the in-tree LZ codec, storing the compressed bytes on disk and recording the
// per-blob metadata in a new v3 region; any blob that does not compress (or
// does not help) is stored uncompressed with a per-blob kNone fallback, so the
// format stays robust on high-entropy (already-quantized) experts. In all cases
// a .strata decodes BYTE-IDENTICALLY to its source GGUF (the EC-4 gate): the
// reader decompresses transparently and the engine sees the same bytes.
bool pack_gguf_to_strata(const std::string &in_path,
                         const std::string &out_path, std::string *err,
                         StrataCodec codec);

// Convenience overload preserving the pre-compression 3-arg signature (codec
// defaults to kNone, i.e. the verbatim pack). Keeps every existing caller
// working unchanged.
inline bool pack_gguf_to_strata(const std::string &in_path,
                                const std::string &out_path, std::string *err) {
    return pack_gguf_to_strata(in_path, out_path, err, StrataCodec::kNone);
}

} // namespace sf

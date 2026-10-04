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

namespace sf {

// Pack the GGUF at `in_path` into a .strata at `out_path`. Writes the 128-byte
// superblock, the verbatim embedded GGUF metadata blob, the expert index, the
// 4 KiB-aligned trunk block (all non-expert tensors), and the expert region
// (one 4 KiB-aligned blob per (layer,expert) bundling gate+up+down). Returns
// true on success; on failure returns false and, when `err` is non-null, sets
// it to a human-readable reason.
bool pack_gguf_to_strata(const std::string &in_path,
                         const std::string &out_path, std::string *err);

} // namespace sf

// strata-pack: convert a GGUF checkpoint into the packed .strata format.
//
// See docs/TASK5_DESIGN.md sections 2 and 5, docs/ENGINE_CORE_DESIGN.md
// section 5. The packing logic lives in the library (src/tws/strata_pack.cpp)
// so the decode test can call it directly; this tool is a thin CLI wrapper.
//
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "tws/strata_pack.h"

#include <cstdio>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                     "Usage: %s <in.gguf> <out.strata>\n"
                     "Convert a GGUF checkpoint into the packed .strata "
                     "format (verbatim weight bytes, aligned expert blobs).\n",
                     argv[0]);
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];

    std::string err;
    if (!sf::pack_gguf_to_strata(in_path, out_path, &err)) {
        std::fprintf(stderr, "strata-pack: %s\n", err.c_str());
        return 1;
    }
    std::printf("strata-pack: wrote '%s'\n", out_path.c_str());
    return 0;
}

// strata-pack: convert a GGUF checkpoint into the packed .strata format.
//
// See docs/TASK5_DESIGN.md sections 2 and 5, docs/ENGINE_CORE_DESIGN.md
// section 5. The packing logic lives in the library (src/tws/strata_pack.cpp)
// so the decode test can call it directly; this tool is a thin CLI wrapper.
//
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "common/lz_codec.h"  // StrataCodec
#include "tws/strata_pack.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char *prog) {
    std::fprintf(stderr,
                 "Usage: %s [--compress <codec>] <in.gguf> <out.strata>\n"
                 "Convert a GGUF checkpoint into the packed .strata format "
                 "(verbatim weight bytes, aligned expert blobs).\n"
                 "\n"
                 "  --compress <codec>  OPTIONAL lossless per-expert-blob "
                 "compression.\n"
                 "                      'lz'   = compress each expert bundle "
                 "with the in-tree LZ codec\n"
                 "                      'none' = store verbatim (the default). "
                 "An uncompressed pack is\n"
                 "                      byte-for-byte identical to the "
                 "pre-compression packer.\n"
                 "\n"
                 "--compress may appear before or after the positional "
                 "arguments.\n",
                 prog);
}

int main(int argc, char **argv) {
    sf::StrataCodec codec = sf::StrataCodec::kNone;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--compress") {
            if (i + 1 >= argc) {
                std::fprintf(stderr,
                             "strata-pack: --compress requires a codec name "
                             "('lz' or 'none')\n");
                return 2;
            }
            const std::string name = argv[++i];
            if (name == "lz") {
                codec = sf::StrataCodec::kLz;
            } else if (name == "none") {
                codec = sf::StrataCodec::kNone;
            } else {
                std::fprintf(stderr,
                             "strata-pack: unknown codec '%s' (expected 'lz' "
                             "or 'none')\n",
                             name.c_str());
                return 2;
            }
        } else if (arg.rfind("--compress=", 0) == 0) {
            const std::string name = arg.substr(std::strlen("--compress="));
            if (name == "lz") {
                codec = sf::StrataCodec::kLz;
            } else if (name == "none") {
                codec = sf::StrataCodec::kNone;
            } else {
                std::fprintf(stderr,
                             "strata-pack: unknown codec '%s' (expected 'lz' "
                             "or 'none')\n",
                             name.c_str());
                return 2;
            }
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 2) {
        usage(argv[0]);
        return 2;
    }
    const std::string in_path = positional[0];
    const std::string out_path = positional[1];

    std::string err;
    if (!sf::pack_gguf_to_strata(in_path, out_path, &err, codec)) {
        std::fprintf(stderr, "strata-pack: %s\n", err.c_str());
        return 1;
    }
    std::printf("strata-pack: wrote '%s'%s\n", out_path.c_str(),
                codec == sf::StrataCodec::kLz ? " (--compress lz)" : "");
    return 0;
}

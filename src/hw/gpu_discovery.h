// GPU discovery bridge.
//
// Enumerating GPUs needs the ggml backend device API (ggml-backend.h), which
// only compiles cleanly in the single translation unit that owns the llama.cpp
// headers (src/model/ggml_model.cpp). The hardware profiler lives in hw/ and
// deliberately does NOT pull in the llama/ggml headers, so it calls this thin
// bridge instead. The dependency direction stays clean: hw/ depends on a plain
// declaration here; the implementation lives next to the backend.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include "hw/profiler.h"

#include <vector>

namespace sf {

// Appends every GGML_BACKEND_DEVICE_TYPE_GPU device the vendored ggml backends
// report to `out` (name, vram_bytes, backend). CPU/accelerator devices are
// skipped. On a CPU-only machine this is a no-op and `out` is left empty; it
// never throws or errors. Implemented in src/model/ggml_model.cpp.
void enumerate_gpus(std::vector<GpuInfo> &out);

} // namespace sf

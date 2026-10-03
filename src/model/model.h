// Model: owns weights, tokenizer, and the forward pass. This is the primary
// seam where llama.cpp/ggml plugs in during Phase 1b. See docs/PLAN.md 3.x.
//
// Phase 1 provides a deterministic "dry-run" model: it reports a realistic
// ModelShape and produces reproducible token ids, so the engine, CLI, server,
// and tests all run end to end before any real weights exist. When ggml is
// vendored, GgmlModel implements this same interface and the dry-run model
// stays as a test fixture.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "plan/planner.h"
#include "strataflow/strataflow.h"

#include <memory>
#include <string>
#include <vector>

namespace sf {

class Model {
public:
    virtual ~Model() = default;

    virtual const ModelShape &shape() const = 0;

    // Tokenize a prompt into ids (naive whitespace scheme in the dry-run model).
    virtual std::vector<int32_t> tokenize(const std::string &text) const = 0;
    virtual std::string detokenize(int32_t token) const = 0;

    // Produce the next token id given the previous one. Deterministic in the
    // dry-run model so greedy output is reproducible (the determinism contract
    // from docs/PLAN.md section 7).
    virtual int32_t forward(int32_t last_token) = 0;

    // Token id that signals end of generation.
    virtual int32_t eos_token() const = 0;
};

// Factory: inspects `path` and returns the right implementation.
//   * An existing *.gguf file is loaded by the vendored llama.cpp backend
//     (GgmlModel). On any load failure it falls back to the dry-run model.
//   * Anything else (missing file, *.strata, unknown extension) falls back to
//     the deterministic dry-run model with a logged warning.
// Never crashes on a bad path: always yields a usable model and SF_OK.
sf_status load_model(const std::string &path, std::unique_ptr<Model> &out);

// Phase 1b llama.cpp backend. Loads a GGUF file via the llama.cpp C API and
// implements the Model interface with real tokenize/detokenize and a real
// greedy single-token forward pass. Defined in model/ggml_model.cpp.
//
// Returns SF_OK and sets `out` on success. On any failure (file missing,
// unreadable GGUF, context allocation failure) it returns an error status and
// leaves `out` untouched -- load_model() then substitutes the dry-run model.
sf_status load_ggml_model(const std::string &path, std::unique_ptr<Model> &out);

} // namespace sf

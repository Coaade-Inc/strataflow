// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "model/model.h"

#include "common/log.h"

#include <cstdint>
#include <sstream>

namespace sf {
namespace {

constexpr uint64_t kGiB = 1024ull * 1024 * 1024;

// Deterministic 64-bit mixer (splitmix64) for reproducible dry-run tokens.
uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// A stand-in MoE model shaped like a trillion-scale Kimi-class checkpoint, so
// the planner and TWS exercise realistic numbers. No real weights are loaded.
class DryRunModel final : public Model {
public:
    explicit DryRunModel(const std::string &path) {
        shape_.name          = "dry-run-moe";
        shape_.n_layers      = 61;
        shape_.is_moe        = true;
        shape_.n_experts     = 256;
        shape_.n_experts_used = 8;
        // Representative sizes (not tied to real weights): ~1 GiB trunk,
        // ~24 MiB per expert at low-bit, ~0.4 TiB total.
        shape_.trunk_bytes   = 1 * kGiB;
        shape_.expert_bytes  = 24ull * 1024 * 1024;
        shape_.total_bytes   = shape_.trunk_bytes +
            uint64_t(shape_.n_experts) * shape_.n_layers * shape_.expert_bytes;
        (void)path;
    }

    const ModelShape &shape() const override { return shape_; }

    std::vector<int32_t> tokenize(const std::string &text) const override {
        std::vector<int32_t> ids;
        std::istringstream ss(text);
        std::string word;
        while (ss >> word) {
            uint64_t h = 0;
            for (char c : word) h = h * 131 + static_cast<unsigned char>(c);
            ids.push_back(static_cast<int32_t>(h % 32000));
        }
        if (ids.empty()) ids.push_back(1);
        return ids;
    }

    std::string detokenize(int32_t token) const override {
        return "<" + std::to_string(token) + ">";
    }

    int32_t forward(int32_t last_token) override {
        state_ = mix(state_ ^ static_cast<uint64_t>(last_token));
        ++generated_;
        if (generated_ >= 16) return eos_;  // bounded dry-run output
        return static_cast<int32_t>(state_ % 32000);
    }

    int32_t eos_token() const override { return eos_; }

private:
    ModelShape shape_;
    uint64_t   state_ = 0x1234567;
    int        generated_ = 0;
    int32_t    eos_ = 2;
};

} // namespace

sf_status load_model(const std::string &path, std::unique_ptr<Model> &out) {
    // Phase 1b will branch on the file type here and construct a GgmlModel for
    // .gguf / .strata inputs. For now everything maps to the dry-run model.
    log_warn("load_model: no ggml backend vendored yet; using dry-run model for '" +
             path + "'. Real weights require the Phase 1b backend.");
    out = std::make_unique<DryRunModel>(path);
    return SF_OK;
}

} // namespace sf

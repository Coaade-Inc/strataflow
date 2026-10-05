// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "model/model.h"

#include "common/log.h"
#include "plan/planner.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>

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

// Case-insensitive check for a trailing extension (e.g. ".gguf").
bool ends_with_ci(const std::string &s, const std::string &suffix) {
    if (s.size() < suffix.size()) return false;
    for (size_t i = 0; i < suffix.size(); ++i) {
        const char a = static_cast<char>(
            std::tolower(static_cast<unsigned char>(s[s.size() - suffix.size() + i])));
        const char b = static_cast<char>(
            std::tolower(static_cast<unsigned char>(suffix[i])));
        if (a != b) return false;
    }
    return true;
}

} // namespace

sf_status load_model(const std::string &path, const HardwareProfile &hw,
                     uint64_t vram_budget, uint64_t ram_budget,
                     std::unique_ptr<Model> &out, PlacementPlan *out_plan,
                     uint64_t cache_budget) {
    // Real weights: an existing *.gguf file goes to the llama.cpp backend. Any
    // failure there (unreadable file, bad GGUF) degrades to the dry-run model
    // rather than failing the whole context -- load_model never crashes on a
    // bad path (docs/PLAN.md "degrade smoothly").
    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec) && !ec;

    // Real weights: a *.gguf goes to the llama.cpp backend; a *.strata goes to
    // our own engine loader (EC-4). load_ggml_model auto-detects the .strata
    // magic and never hands it to libllama (the TASK5_BLOCKER root cause). Any
    // failure degrades to the dry-run model rather than failing the context.
    if (exists && (ends_with_ci(path, ".gguf") ||
                   ends_with_ci(path, ".strata"))) {
        sf_status st =
            load_ggml_model(path, hw, vram_budget, ram_budget, out, out_plan,
                            cache_budget);
        if (st == SF_OK && out) {
            return SF_OK;
        }
        log_warn("load_model: model load failed for '" + path +
                 "'; falling back to dry-run model.");
    } else {
        // Missing file, or a format we don't load yet (.strata is Phase 2+).
        log_warn("load_model: no real backend for '" + path +
                 "' (missing file or unsupported format); using dry-run model.");
    }

    out = std::make_unique<DryRunModel>(path);
    // The dry-run model has no real tensors to place, but the planner still
    // runs over its representative shape so the C API can describe a plan.
    if (out_plan != nullptr) {
        *out_plan = plan_placement(hw, out->shape(), vram_budget, ram_budget,
                                   cache_budget);
    }
    return SF_OK;
}

} // namespace sf

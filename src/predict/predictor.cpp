// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "predict/predictor.h"

#include <algorithm>
#include <numeric>

namespace sf {

ExpertPredictor::ExpertPredictor(uint32_t n_layers, uint32_t n_experts)
    : n_layers_(n_layers), n_experts_(n_experts),
      freq_(n_layers, std::vector<uint64_t>(n_experts, 0)) {}

void ExpertPredictor::observe(const ExpertId &id) {
    if (id.layer < n_layers_ && id.expert < n_experts_) {
        ++freq_[id.layer][id.expert];
    }
    if (std::find(last_prediction_.begin(), last_prediction_.end(), id.expert) !=
        last_prediction_.end()) {
        ++confirmed_;
    }
}

std::vector<uint32_t> ExpertPredictor::predict(uint32_t layer, uint32_t k) const {
    std::vector<uint32_t> out;
    if (layer >= n_layers_ || k == 0) return out;

    const auto &f = freq_[layer];
    std::vector<uint32_t> idx(f.size());
    std::iota(idx.begin(), idx.end(), 0u);

    using Diff = std::vector<uint32_t>::difference_type;
    const Diff take = static_cast<Diff>(std::min<size_t>(k, idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + take, idx.end(),
                      [&f](uint32_t a, uint32_t b) { return f[a] > f[b]; });
    out.assign(idx.begin(), idx.begin() + take);

    predicted_ += out.size();
    last_prediction_ = out;
    return out;
}

double ExpertPredictor::accuracy() const {
    return predicted_ ? double(confirmed_) / double(predicted_) : 0.0;
}

std::vector<uint64_t> ExpertPredictor::layer_activity() const {
    std::vector<uint64_t> out(n_layers_, 0);
    for (uint32_t l = 0; l < n_layers_; ++l) {
        uint64_t sum = 0;
        for (uint64_t c : freq_[l]) sum += c;
        out[l] = sum;
    }
    return out;
}

} // namespace sf

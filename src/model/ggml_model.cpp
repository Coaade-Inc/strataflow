// GgmlModel: the Phase 1b llama.cpp-backed implementation of sf::Model.
//
// It loads a GGUF checkpoint through the vendored llama.cpp C API, tokenizes
// and detokenizes with the model's real vocabulary, and runs a real greedy
// (argmax) single-token forward pass. KV state is carried across forward()
// calls by owning the llama_context and tracking the sequence position
// internally, matching the one-token-per-call contract the Scheduler relies
// on (see src/sched/scheduler.cpp and docs/PLAN.md section 3.x).
//
// Only this translation unit touches llama.cpp; the rest of the engine sees
// just the sf::Model interface, so the backend stays swappable.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "model/model.h"

#include "common/log.h"

#include <llama.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace sf {
namespace {

// llama_backend_init() must run exactly once per process before any model is
// loaded. std::call_once keeps that thread-safe without a global dtor race on
// backend teardown (the backend lives for the process lifetime, which is fine
// for a long-running inference server).
void ensure_backend() {
    static std::once_flag once;
    std::call_once(once, [] { llama_backend_init(); });
}

// Route llama.cpp's internal logging into our logger. Keeps the CLI/test
// output tidy and under our log-level control.
void sf_llama_log_cb(ggml_log_level level, const char *text, void * /*user*/) {
    if (text == nullptr) return;
    std::string msg(text);
    // llama.cpp lines usually carry their own trailing newline; trim it so our
    // logger owns the formatting.
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
    }
    if (msg.empty()) return;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: log_error("llama: " + msg); break;
        case GGML_LOG_LEVEL_WARN:  log_warn("llama: " + msg);  break;
        case GGML_LOG_LEVEL_INFO:  log_debug("llama: " + msg); break;
        default:                    log_debug("llama: " + msg); break;
    }
}

class GgmlModel final : public Model {
public:
    // Takes ownership of a loaded model + context. Private; use create().
    GgmlModel(llama_model *model, llama_context *ctx, const ModelShape &shape,
              llama_token eos)
        : model_(model), ctx_(ctx), vocab_(llama_model_get_vocab(model)),
          shape_(shape), eos_(eos) {}

    GgmlModel(const GgmlModel &) = delete;
    GgmlModel &operator=(const GgmlModel &) = delete;

    ~GgmlModel() override {
        if (ctx_ != nullptr) llama_free(ctx_);
        if (model_ != nullptr) llama_model_free(model_);
    }

    const ModelShape &shape() const override { return shape_; }

    std::vector<int32_t> tokenize(const std::string &text) const override {
        std::vector<int32_t> ids;
        // llama's tokenizer can throw (e.g. std::out_of_range) on a malformed
        // or incomplete vocabulary. This method is reachable from the C ABI,
        // which must not let C++ exceptions escape, so contain them here and
        // degrade to a single BOS token instead of crashing the process.
        try {
            // First call with a small buffer returns the (negative) required size.
            const int32_t n_max = static_cast<int32_t>(text.size()) + 8;
            std::vector<llama_token> toks(static_cast<size_t>(n_max));
            int32_t n = llama_tokenize(vocab_, text.c_str(),
                                       static_cast<int32_t>(text.size()),
                                       toks.data(), n_max,
                                       /*add_special=*/true,
                                       /*parse_special=*/true);
            if (n < 0) {
                // Buffer was too small; -n is the real count. Retry exactly once.
                toks.resize(static_cast<size_t>(-n));
                n = llama_tokenize(vocab_, text.c_str(),
                                   static_cast<int32_t>(text.size()),
                                   toks.data(), -n,
                                   /*add_special=*/true,
                                   /*parse_special=*/true);
            }
            if (n > 0) {
                ids.assign(toks.begin(), toks.begin() + n);
            }
        } catch (const std::exception &e) {
            log_error(std::string("GgmlModel::tokenize: ") + e.what());
            ids.clear();
        }
        if (ids.empty()) {
            // Keep the engine's "always have a seed token" invariant.
            ids.push_back(llama_vocab_bos(vocab_));
        }
        return ids;
    }

    std::string detokenize(int32_t token) const override {
        char buf[256];
        int32_t n = llama_token_to_piece(vocab_, token, buf,
                                         static_cast<int32_t>(sizeof(buf)),
                                         /*lstrip=*/0, /*special=*/false);
        if (n < 0) {
            std::vector<char> big(static_cast<size_t>(-n));
            n = llama_token_to_piece(vocab_, token, big.data(),
                                     static_cast<int32_t>(big.size()),
                                     /*lstrip=*/0, /*special=*/false);
            if (n <= 0) return std::string();
            return std::string(big.data(), static_cast<size_t>(n));
        }
        if (n == 0) return std::string();
        return std::string(buf, static_cast<size_t>(n));
    }

    // Decode `last_token` at the current position, read its logits, and return
    // the greedy (argmax) next token. KV state advances by one each call.
    int32_t forward(int32_t last_token) override {
        llama_token tok = last_token;
        llama_batch batch = llama_batch_get_one(&tok, 1);

        const int32_t rc = llama_decode(ctx_, batch);
        if (rc != 0) {
            log_error("GgmlModel::forward: llama_decode failed (rc=" +
                      std::to_string(rc) + ")");
            return eos_;
        }
        ++n_past_;

        const float *logits = llama_get_logits_ith(ctx_, -1);
        if (logits == nullptr) {
            log_error("GgmlModel::forward: null logits");
            return eos_;
        }

        const int32_t n_vocab = llama_vocab_n_tokens(vocab_);
        int32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best = i;
            }
        }
        return best;
    }

    int32_t eos_token() const override { return eos_; }

private:
    llama_model         *model_ = nullptr;
    llama_context       *ctx_   = nullptr;
    const llama_vocab   *vocab_ = nullptr;
    ModelShape           shape_{};
    llama_token          eos_    = 0;
    int32_t              n_past_ = 0;
};

// Read a GGUF uint32 metadata value by key; returns `fallback` if absent or
// unparsable. llama.cpp exposes metadata only as strings here.
uint32_t meta_u32(llama_model *model, const char *key, uint32_t fallback) {
    char buf[128];
    int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n <= 0) return fallback;
    char *end = nullptr;
    unsigned long v = std::strtoul(buf, &end, 10);
    if (end == buf) return fallback;
    return static_cast<uint32_t>(v);
}

// Build the planner's ModelShape from the loaded model's real metadata. MoE
// expert counts are read from the "<arch>.expert_count" / ".expert_used_count"
// GGUF keys when present (dense models report 0). Byte sizes come straight
// from the on-disk tensor totals.
ModelShape make_shape(llama_model *model) {
    ModelShape s;

    char name[256];
    int32_t nn = llama_model_meta_val_str(model, "general.name", name,
                                          sizeof(name));
    if (nn <= 0) {
        char desc[256];
        if (llama_model_desc(model, desc, sizeof(desc)) > 0) {
            s.name = desc;
        } else {
            s.name = "gguf-model";
        }
    } else {
        s.name = name;
    }

    s.n_layers = static_cast<uint32_t>(llama_model_n_layer(model));

    // Architecture prefix for the MoE metadata keys (e.g. "llama", "qwen2moe").
    char arch[64];
    std::string arch_str;
    if (llama_model_meta_val_str(model, "general.architecture", arch,
                                 sizeof(arch)) > 0) {
        arch_str = arch;
    }
    uint32_t n_experts = 0;
    uint32_t n_experts_used = 0;
    if (!arch_str.empty()) {
        n_experts = meta_u32(model, (arch_str + ".expert_count").c_str(), 0);
        n_experts_used =
            meta_u32(model, (arch_str + ".expert_used_count").c_str(), 0);
    }
    s.is_moe         = n_experts > 0;
    s.n_experts      = n_experts;
    s.n_experts_used = n_experts_used;

    s.total_bytes = llama_model_size(model);
    if (s.is_moe && s.n_experts > 0 && s.n_layers > 0) {
        // Rough split: assume expert weights dominate and are evenly spread
        // across layers and experts. Good enough for the planner's budgeting;
        // precise per-tensor accounting lands with the .strata format.
        const uint64_t experts_total =
            static_cast<uint64_t>(s.total_bytes) * 7 / 10;  // ~70% in experts
        s.expert_bytes = experts_total /
            (static_cast<uint64_t>(s.n_experts) * s.n_layers);
        s.trunk_bytes = s.total_bytes - experts_total;
    } else {
        s.trunk_bytes  = s.total_bytes;
        s.expert_bytes = 0;
    }
    return s;
}

} // namespace

sf_status load_ggml_model(const std::string &path,
                          std::unique_ptr<Model> &out) {
    ensure_backend();
    llama_log_set(sf_llama_log_cb, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  // CPU-only for this phase.

    llama_model *model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        log_warn("load_ggml_model: failed to load GGUF '" + path + "'");
        return SF_ERR_MODEL_LOAD;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 0;  // use the model's trained context length.

    llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        log_error("load_ggml_model: failed to create context for '" + path + "'");
        llama_model_free(model);
        return SF_ERR_MODEL_LOAD;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token eos = llama_vocab_eos(vocab);
    if (eos < 0) eos = 0;

    ModelShape shape = make_shape(model);
    log_info("GgmlModel loaded '" + shape.name + "': " +
             std::to_string(shape.n_layers) + " layers, " +
             (shape.is_moe ? ("MoE " + std::to_string(shape.n_experts) + "x" +
                              std::to_string(shape.n_experts_used))
                           : std::string("dense")) +
             ", " + std::to_string(shape.total_bytes) + " bytes");

    out = std::unique_ptr<Model>(new GgmlModel(model, ctx, shape, eos));
    return SF_OK;
}

} // namespace sf

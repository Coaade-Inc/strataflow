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
#include "hw/gpu_discovery.h"
#include "plan/llama_placement.h"
#include "plan/planner.h"

#include <ggml-backend.h>
#include <gguf.h>
#include <llama.h>

#include <array>
#include <cctype>
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

// Map a ggml backend registry name to the public sf_backend enum. The reg name
// is the backend family ("CUDA", "Vulkan", "Metal", "ROCm"/"HIP", "CPU", ...).
sf_backend backend_from_reg_name(const char *reg_name) {
    if (reg_name == nullptr) return SF_BACKEND_CPU;
    std::string n(reg_name);
    for (char &c : n) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (n.find("cuda") != std::string::npos) return SF_BACKEND_CUDA;
    if (n.find("vulkan") != std::string::npos) return SF_BACKEND_VULKAN;
    if (n.find("metal") != std::string::npos) return SF_BACKEND_METAL;
    if (n.find("hip") != std::string::npos || n.find("rocm") != std::string::npos)
        return SF_BACKEND_HIP;
    return SF_BACKEND_CPU;
}

// Read the few GGUF metadata values the planner needs (layer count, MoE expert
// counts, total weight bytes) WITHOUT building any tensors. We use the light
// gguf_* reader with no_alloc=true: it parses the header/kv/tensor-info only,
// so this is cheap compared to the full llama_model_load_from_file tensor read.
//
// Tradeoff: this duplicates a little of make_shape()'s metadata logic, but it
// lets us compute the placement plan and set llama_model_params BEFORE the
// heavy load, avoiding a load-twice (which would double I/O for GPU configs).
// After the real load we still call make_shape(model) for the authoritative
// shape; the two agree, this pre-pass only drives the load params.
bool read_shape_from_gguf(const std::string &path, ModelShape &s) {
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context *gc = gguf_init_from_file(path.c_str(), gp);
    if (gc == nullptr) return false;

    auto get_u32 = [&](const std::string &key, uint32_t fallback) -> uint32_t {
        int64_t id = gguf_find_key(gc, key.c_str());
        if (id < 0) return fallback;
        // GGUF integer metadata is commonly stored as u32; be tolerant of the
        // signed variant too.
        return gguf_get_val_u32(gc, id);
    };

    std::string arch;
    {
        int64_t id = gguf_find_key(gc, "general.architecture");
        if (id >= 0) {
            const char *a = gguf_get_val_str(gc, id);
            if (a != nullptr) arch = a;
        }
    }
    {
        int64_t id = gguf_find_key(gc, "general.name");
        if (id >= 0) {
            const char *nm = gguf_get_val_str(gc, id);
            if (nm != nullptr) s.name = nm;
        }
    }
    if (s.name.empty()) s.name = arch.empty() ? "gguf-model" : arch;

    if (!arch.empty()) {
        s.n_layers = get_u32(arch + ".block_count", 0);
        s.n_experts = get_u32(arch + ".expert_count", 0);
        s.n_experts_used = get_u32(arch + ".expert_used_count", 0);
    }
    s.is_moe = s.n_experts > 0;

    // Total weight bytes: sum the on-disk tensor sizes. gguf exposes each
    // tensor's byte size directly, so no tensor data is read.
    uint64_t total = 0;
    const int64_t n_tensors = gguf_get_n_tensors(gc);
    for (int64_t i = 0; i < n_tensors; ++i) {
        total += gguf_get_tensor_size(gc, i);
    }
    s.total_bytes = total;

    if (s.is_moe && s.n_experts > 0 && s.n_layers > 0) {
        const uint64_t experts_total = s.total_bytes * 7 / 10;  // ~70% experts
        s.expert_bytes =
            experts_total / (static_cast<uint64_t>(s.n_experts) * s.n_layers);
        s.trunk_bytes = s.total_bytes - experts_total;
    } else {
        s.trunk_bytes = s.total_bytes;
        s.expert_bytes = 0;
    }

    gguf_free(gc);
    return true;
}

} // namespace

// GPU discovery bridge (declared in hw/gpu_discovery.h). Enumerates ggml
// backend devices and records every GPU-type device. CPU/accelerator devices
// are skipped. No-op on a CPU-only machine; never throws.
void enumerate_gpus(std::vector<GpuInfo> &out) {
    ensure_backend();
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) continue;
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;  // only count dedicated GPUs, not CPU/ACCEL/IGPU/META
        }
        GpuInfo g;
        const char *desc = ggml_backend_dev_description(dev);
        g.name = desc != nullptr ? desc : "gpu";
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        g.vram_bytes = static_cast<uint64_t>(total_bytes);
        g.backend = backend_from_reg_name(
            ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
        out.push_back(std::move(g));
    }
}

sf_status load_ggml_model(const std::string &path, const HardwareProfile &hw,
                          uint64_t vram_budget, uint64_t ram_budget,
                          std::unique_ptr<Model> &out, PlacementPlan *out_plan) {
    ensure_backend();
    llama_log_set(sf_llama_log_cb, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  // CPU default; overridden below from the plan.

    // Phase 2: plan placement from a cheap metadata-only pre-pass so the load
    // params reflect the plan BEFORE the heavy tensor read. If the pre-pass
    // fails for any reason we fall back to a plain CPU load (identical to the
    // Phase 1b behaviour), which is always safe.
    ModelShape pre_shape;
    LlamaPlacement place;
    PlacementPlan plan;
    // The CPU/host buffer type for expert offload; must outlive the load call.
    std::array<llama_model_tensor_buft_override, 2> overrides{};
    if (read_shape_from_gguf(path, pre_shape)) {
        plan = plan_placement(hw, pre_shape, vram_budget, ram_budget);
        place = derive_llama_placement(plan, pre_shape, hw.gpus);

        mp.n_gpu_layers = place.n_gpu_layers;

        if (place.offload_experts_to_cpu) {
            // Resolve "experts -> CPU" into a concrete host buffer type. This
            // is the llama.cpp equivalent of --n-cpu-moe / -ot "exps=CPU":
            // route the routed-expert FFN tensors to the CPU device's default
            // buffer type via a NULL-terminated pattern override array.
            ggml_backend_buffer_type_t cpu_buft = nullptr;
            ggml_backend_dev_t cpu_dev =
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu_dev != nullptr) {
                cpu_buft = ggml_backend_dev_buffer_type(cpu_dev);
            }
            if (cpu_buft == nullptr) {
                cpu_buft = ggml_backend_cpu_buffer_type();
            }
            overrides[0].pattern = place.expert_override_pattern.c_str();
            overrides[0].buft = cpu_buft;
            overrides[1].pattern = nullptr;  // NULL-terminates the array
            overrides[1].buft = nullptr;
            mp.tensor_buft_overrides = overrides.data();
            log_info("GgmlModel: offloading MoE experts to CPU buffer (pattern '" +
                     place.expert_override_pattern + "')");
        }
    } else {
        log_warn("load_ggml_model: metadata pre-pass failed for '" + path +
                 "'; loading CPU-only.");
    }

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

    // Report the plan that was applied. Prefer re-planning from the loaded
    // model's authoritative shape so the summary matches reality exactly; it
    // agrees with the pre-pass plan used for the load params.
    if (out_plan != nullptr) {
        *out_plan = plan_placement(hw, shape, vram_budget, ram_budget);
    }

    out = std::unique_ptr<Model>(new GgmlModel(model, ctx, shape, eos));
    return SF_OK;
}

} // namespace sf

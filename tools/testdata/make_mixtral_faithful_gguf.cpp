// Generate a Mixtral-FAITHFUL llama-arch MoE GGUF for root-causing the packer's
// "0 experts" on the real model. Mirrors the real Mixtral-8x7B tensor naming,
// stacking, quant family (K-quant), and (optionally) the real hidden dims, so
// the packer/loader classification path sees exactly what the real model
// presents. Configurable so a small-but-faithful fixture can live in CI.
//
// Usage:
//   make-mixtral-faithful-gguf <out.gguf> [n_embd] [n_ff] [n_layer] [n_expert] [qtype]
//   defaults: 256 512 2 8 q4_k  (CI-small)
//   real Mixtral dims:  make-... out.gguf 4096 14336 1 8 q4_k
//
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
std::mt19937 g_rng(0);

void fill_f32(std::vector<float> &buf, int64_t n) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    buf.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) buf[static_cast<size_t>(i)] = dist(g_rng) * 0.02f;
}

void add_quant_tensor(gguf_context *gc, ggml_context *mc, const std::string &name,
                      ggml_type qtype, int n_dims, const int64_t *ne) {
    int64_t n_elem = 1;
    for (int d = 0; d < n_dims; ++d) n_elem *= ne[d];
    const int64_t n_per_row = ne[0];
    const int64_t nrows = n_elem / n_per_row;
    std::vector<float> src;
    fill_f32(src, n_elem);
    ggml_tensor *t = ggml_new_tensor(mc, qtype, n_dims, ne);
    ggml_set_name(t, name.c_str());
    const size_t qbytes = ggml_quantize_chunk(qtype, src.data(), t->data, 0, nrows,
                                              n_per_row, nullptr);
    if (qbytes != ggml_nbytes(t)) {
        std::fprintf(stderr, "quant size mismatch for %s\n", name.c_str());
        std::exit(1);
    }
    gguf_add_tensor(gc, t);
}

void add_f32_tensor(gguf_context *gc, ggml_context *mc, const std::string &name,
                    int n_dims, const int64_t *ne, bool ones) {
    ggml_tensor *t = ggml_new_tensor(mc, GGML_TYPE_F32, n_dims, ne);
    ggml_set_name(t, name.c_str());
    int64_t n_elem = 1;
    for (int d = 0; d < n_dims; ++d) n_elem *= ne[d];
    float *dst = static_cast<float *>(t->data);
    if (ones) {
        for (int64_t i = 0; i < n_elem; ++i) dst[i] = 1.0f;
    } else {
        std::vector<float> src;
        fill_f32(src, n_elem);
        std::memcpy(dst, src.data(), src.size() * sizeof(float));
    }
    gguf_add_tensor(gc, t);
}
}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out.gguf> [n_embd n_ff n_layer n_expert qtype]\n", argv[0]);
        return 2;
    }
    const std::string out = argv[1];
    const int64_t n_embd   = argc > 2 ? std::atoll(argv[2]) : 256;
    const int64_t n_ff     = argc > 3 ? std::atoll(argv[3]) : 512;
    const int64_t n_layer  = argc > 4 ? std::atoll(argv[4]) : 2;
    const int64_t n_expert = argc > 5 ? std::atoll(argv[5]) : 8;
    const std::string qarg = argc > 6 ? argv[6] : "q4_k";
    const int64_t n_head = 32 <= n_embd / 8 ? 32 : 4;
    const int64_t head_dim = n_embd / n_head;
    const int64_t n_expert_used = 2;

    ggml_type qtype = GGML_TYPE_Q4_K;
    int32_t ftype = 15;
    if (qarg == "q3_k") { qtype = GGML_TYPE_Q3_K; ftype = 12; }
    else if (qarg == "q6_k") { qtype = GGML_TYPE_Q6_K; ftype = 18; }
    else if (qarg == "q4_k") { qtype = GGML_TYPE_Q4_K; ftype = 15; }

    std::vector<std::string> toks;
    std::vector<int32_t> ttypes;
    toks.push_back("<unk>"); toks.push_back("<s>"); toks.push_back("</s>");
    for (int i = 0; i < 3; ++i) ttypes.push_back(3);
    for (int i = 0; i < 256; ++i) {
        char b[16]; std::snprintf(b, sizeof(b), "<0x%02X>", i);
        toks.emplace_back(b); ttypes.push_back(6);
    }
    const char *normal[] = {"\xe2\x96\x81" "hello", "\xe2\x96\x81" "world",
                            "\xe2\x96\x81" "the", "\xe2\x96\x81" "a", "\xe2\x96\x81" "of"};
    for (const char *n : normal) { toks.emplace_back(n); ttypes.push_back(1); }
    const int64_t n_vocab = static_cast<int64_t>(toks.size());

    // Memory: generously size for quantized bytes + overhead.
    size_t mem = 512u * 1024u * 1024u;
    size_t need = static_cast<size_t>(n_embd) * static_cast<size_t>(n_ff) *
                  static_cast<size_t>(n_expert) * static_cast<size_t>(n_layer) * 4 * 4;
    if (need > mem) mem = need;
    ggml_init_params ip{}; ip.mem_size = mem; ip.no_alloc = false;
    ggml_context *mc = ggml_init(ip);
    gguf_context *gc = gguf_init_empty();

    gguf_set_val_str(gc, "general.architecture", "llama");
    gguf_set_val_str(gc, "general.name", "mixtral-faithful");
    gguf_set_val_u32(gc, "llama.context_length", 256);
    gguf_set_val_u32(gc, "llama.embedding_length", (uint32_t)n_embd);
    gguf_set_val_u32(gc, "llama.block_count", (uint32_t)n_layer);
    gguf_set_val_u32(gc, "llama.feed_forward_length", (uint32_t)n_ff);
    gguf_set_val_u32(gc, "llama.attention.head_count", (uint32_t)n_head);
    gguf_set_val_u32(gc, "llama.attention.head_count_kv", (uint32_t)n_head);
    gguf_set_val_u32(gc, "llama.rope.dimension_count", (uint32_t)head_dim);
    gguf_set_val_f32(gc, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gc, "llama.expert_count", (uint32_t)n_expert);
    gguf_set_val_u32(gc, "llama.expert_used_count", (uint32_t)n_expert_used);
    gguf_set_val_u32(gc, "general.file_type", (uint32_t)ftype);
    gguf_set_val_str(gc, "tokenizer.ggml.model", "llama");
    std::vector<const char *> tp; for (auto &s : toks) tp.push_back(s.c_str());
    gguf_set_arr_str(gc, "tokenizer.ggml.tokens", tp.data(), tp.size());
    std::vector<float> scores((size_t)n_vocab, 0.0f);
    gguf_set_arr_data(gc, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(gc, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, ttypes.data(), ttypes.size());
    gguf_set_val_u32(gc, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(gc, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(gc, "tokenizer.ggml.unknown_token_id", 0);

    { const int64_t ne[2] = {n_embd, n_vocab}; add_quant_tensor(gc, mc, "token_embd.weight", qtype, 2, ne); }
    { const int64_t ne1[1] = {n_embd}; add_f32_tensor(gc, mc, "output_norm.weight", 1, ne1, true);
      const int64_t ne2[2] = {n_embd, n_vocab}; add_quant_tensor(gc, mc, "output.weight", qtype, 2, ne2); }

    for (int64_t l = 0; l < n_layer; ++l) {
        char p[32]; std::snprintf(p, sizeof(p), "blk.%lld.", (long long)l);
        const std::string pre = p;
        const int64_t ne_norm[1] = {n_embd};
        add_f32_tensor(gc, mc, pre + "attn_norm.weight", 1, ne_norm, true);
        const int64_t ne_attn[2] = {n_embd, n_embd};
        add_quant_tensor(gc, mc, pre + "attn_q.weight", qtype, 2, ne_attn);
        add_quant_tensor(gc, mc, pre + "attn_k.weight", qtype, 2, ne_attn);
        add_quant_tensor(gc, mc, pre + "attn_v.weight", qtype, 2, ne_attn);
        add_quant_tensor(gc, mc, pre + "attn_output.weight", qtype, 2, ne_attn);
        add_f32_tensor(gc, mc, pre + "ffn_norm.weight", 1, ne_norm, true);
        const int64_t ne_gate_inp[2] = {n_embd, n_expert};
        add_f32_tensor(gc, mc, pre + "ffn_gate_inp.weight", 2, ne_gate_inp, false);
        const int64_t ne_gate[3] = {n_embd, n_ff, n_expert};
        const int64_t ne_down[3] = {n_ff, n_embd, n_expert};
        const int64_t ne_up[3]   = {n_embd, n_ff, n_expert};
        add_quant_tensor(gc, mc, pre + "ffn_gate_exps.weight", qtype, 3, ne_gate);
        add_quant_tensor(gc, mc, pre + "ffn_down_exps.weight", qtype, 3, ne_down);
        add_quant_tensor(gc, mc, pre + "ffn_up_exps.weight", qtype, 3, ne_up);
    }

    if (!gguf_write_to_file(gc, out.c_str(), false)) {
        std::fprintf(stderr, "write failed\n"); return 1;
    }
    std::printf("wrote %s: n_embd=%lld n_ff=%lld n_layer=%lld n_expert=%lld %s\n",
                out.c_str(), (long long)n_embd, (long long)n_ff, (long long)n_layer,
                (long long)n_expert, qarg.c_str());
    gguf_free(gc); ggml_free(mc); ggml_quantize_free();
    return 0;
}

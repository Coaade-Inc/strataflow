// Generate a tiny K-quant (Q4_K or Q6_K) llama-arch MoE GGUF so the engine's
// quantized-weight path can be validated against the libllama oracle without a
// multi-GB download. This is the C++ sibling of make_tiny_quant_moe_gguf.py:
// the Python gguf library cannot quantize to K-quants, so we quantize here with
// the vendored ggml (ggml_quantize_chunk), which has full K-quant support.
//
// K-quant superblock size is 256, so every quantized tensor's innermost dim
// must be a multiple of 256. The tiny Q8_0 fixture uses n_embd=32/n_ff=64,
// which is NOT valid for K-quant, so this tool uses K-quant-friendly dims
// (n_embd=256, n_ff=512) while mirroring the Q8_0 fixture's structure/vocab.
//
// Usage:
//   make-kquant-moe-gguf <out.gguf> [q4_k|q6_k]
//   STRATAFLOW_TEST_KQUANT_MOE_GGUF=<out.gguf> ctest -R test_engine_oracle
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

// llama file_type enum values (from llama.h); set so the oracle reports the
// right quant family. We only emit the two K-quant types this tool supports.
constexpr int32_t kFtypeQ4KM = 15;  // LLAMA_FTYPE_MOSTLY_Q4_K_M
constexpr int32_t kFtypeQ6K = 18;   // LLAMA_FTYPE_MOSTLY_Q6_K

// K-quant-friendly dims: inner dims (n_embd, n_ff) are multiples of 256.
constexpr int64_t kNEmbd = 256;
constexpr int64_t kNHead = 4;
constexpr int64_t kNLayer = 2;
constexpr int64_t kNFf = 512;
constexpr int64_t kNExpert = 8;
constexpr int64_t kNExpertUsed = 2;

// One global RNG so the fixture is deterministic across runs/compilers.
std::mt19937 g_rng(0);

void fill_f32(std::vector<float> &buf, int64_t n) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    buf.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) buf[static_cast<size_t>(i)] = dist(g_rng) * 0.02f;
}

// Add a quantized matmul weight tensor. `ne` is the ggml ne-shape (ne[0] is the
// innermost/row dim and must be a multiple of 256 for K-quant). The F32 source
// is generated here and quantized row-by-row with ggml_quantize_chunk.
void add_quant_tensor(gguf_context *gc, ggml_context *mc, const char *name,
                      ggml_type qtype, int n_dims, const int64_t *ne) {
    int64_t n_elem = 1;
    for (int d = 0; d < n_dims; ++d) n_elem *= ne[d];
    const int64_t n_per_row = ne[0];
    const int64_t nrows = n_elem / n_per_row;

    std::vector<float> src;
    fill_f32(src, n_elem);

    ggml_tensor *t = ggml_new_tensor(mc, qtype, n_dims, ne);
    ggml_set_name(t, name);
    const size_t qbytes = ggml_quantize_chunk(qtype, src.data(), t->data,
                                              /*start=*/0, nrows, n_per_row,
                                              /*imatrix=*/nullptr);
    if (qbytes != ggml_nbytes(t)) {
        std::fprintf(stderr,
                     "make-kquant-moe-gguf: quant size mismatch for %s "
                     "(%zu vs %zu)\n",
                     name, qbytes, ggml_nbytes(t));
        std::exit(1);
    }
    gguf_add_tensor(gc, t);
}

// Add an F32 tensor (norms, router). Data is generated here when `ones` is
// false; norms use all-ones to match the Python fixtures.
void add_f32_tensor(gguf_context *gc, ggml_context *mc, const char *name,
                    int n_dims, const int64_t *ne, bool ones) {
    ggml_tensor *t = ggml_new_tensor(mc, GGML_TYPE_F32, n_dims, ne);
    ggml_set_name(t, name);
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
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr,
                     "Usage: %s <out.gguf> [q4_k|q6_k]\n"
                     "Generate a tiny K-quant llama-arch MoE GGUF (random "
                     "weights) for oracle-gated engine tests.\n",
                     argv[0]);
        return 2;
    }
    const std::string out = argv[1];
    const std::string type_arg = (argc == 3) ? argv[2] : "q4_k";

    ggml_type qtype;
    int32_t ftype;
    const char *type_name;
    if (type_arg == "q4_k" || type_arg == "Q4_K") {
        qtype = GGML_TYPE_Q4_K;
        ftype = kFtypeQ4KM;
        type_name = "Q4_K";
    } else if (type_arg == "q6_k" || type_arg == "Q6_K") {
        qtype = GGML_TYPE_Q6_K;
        ftype = kFtypeQ6K;
        type_name = "Q6_K";
    } else {
        std::fprintf(stderr, "make-kquant-moe-gguf: unknown type '%s' "
                             "(expected q4_k or q6_k)\n", type_arg.c_str());
        return 2;
    }

    const int64_t head_dim = kNEmbd / kNHead;

    // Vocab mirrors make_tiny_quant_moe_gguf.py: 3 control + 256 byte-fallback
    // + a few normal pieces, so the SPM tokenizer can resegment any input.
    std::vector<std::string> toks;
    std::vector<int32_t> ttypes;
    toks.push_back("<unk>");
    toks.push_back("<s>");
    toks.push_back("</s>");
    for (int i = 0; i < 3; ++i) ttypes.push_back(3);  // CONTROL
    for (int i = 0; i < 256; ++i) {
        char b[16];
        std::snprintf(b, sizeof(b), "<0x%02X>", i);
        toks.emplace_back(b);
        ttypes.push_back(6);  // BYTE
    }
    // The U+2581 lower-one-eighth-block prefix (SPM word boundary) is written
    // as explicit bytes split from the following ASCII so the hex escape does
    // not swallow the next character.
    const char *normal[] = {"\xe2\x96\x81" "hello", "\xe2\x96\x81" "world",
                            "\xe2\x96\x81" "the", "\xe2\x96\x81" "a",
                            "\xe2\x96\x81" "of"};
    for (const char *n : normal) {
        toks.emplace_back(n);
        ttypes.push_back(1);  // NORMAL
    }
    const int64_t n_vocab = static_cast<int64_t>(toks.size());

    // One no_alloc=false context holds every tensor's data while we quantize
    // and write. Size generously: it must cover the F32 norms/router plus the
    // quantized weight bytes plus ggml tensor overhead.
    const size_t mem = 256u * 1024u * 1024u;
    ggml_init_params ip{};
    ip.mem_size = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context *mc = ggml_init(ip);
    if (mc == nullptr) {
        std::fprintf(stderr, "make-kquant-moe-gguf: ggml_init failed\n");
        return 1;
    }

    gguf_context *gc = gguf_init_empty();

    gguf_set_val_str(gc, "general.architecture", "llama");
    gguf_set_val_str(gc, "general.name", "tiny-kquant-moe");
    gguf_set_val_u32(gc, "llama.context_length", 256);
    gguf_set_val_u32(gc, "llama.embedding_length", static_cast<uint32_t>(kNEmbd));
    gguf_set_val_u32(gc, "llama.block_count", static_cast<uint32_t>(kNLayer));
    gguf_set_val_u32(gc, "llama.feed_forward_length", static_cast<uint32_t>(kNFf));
    gguf_set_val_u32(gc, "llama.attention.head_count", static_cast<uint32_t>(kNHead));
    gguf_set_val_u32(gc, "llama.attention.head_count_kv", static_cast<uint32_t>(kNHead));
    gguf_set_val_u32(gc, "llama.rope.dimension_count", static_cast<uint32_t>(head_dim));
    gguf_set_val_f32(gc, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gc, "llama.expert_count", static_cast<uint32_t>(kNExpert));
    gguf_set_val_u32(gc, "llama.expert_used_count", static_cast<uint32_t>(kNExpertUsed));
    gguf_set_val_u32(gc, "general.file_type", static_cast<uint32_t>(ftype));

    gguf_set_val_str(gc, "tokenizer.ggml.model", "llama");
    std::vector<const char *> tok_ptrs;
    tok_ptrs.reserve(toks.size());
    for (const std::string &s : toks) tok_ptrs.push_back(s.c_str());
    gguf_set_arr_str(gc, "tokenizer.ggml.tokens", tok_ptrs.data(), tok_ptrs.size());
    std::vector<float> scores(static_cast<size_t>(n_vocab), 0.0f);
    gguf_set_arr_data(gc, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32,
                      scores.data(), scores.size());
    gguf_set_arr_data(gc, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                      ttypes.data(), ttypes.size());
    gguf_set_val_u32(gc, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(gc, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(gc, "tokenizer.ggml.unknown_token_id", 0);

    // token_embd and output quantized (exercises quantized get_rows / lm_head).
    {
        const int64_t ne[2] = {kNEmbd, n_vocab};
        add_quant_tensor(gc, mc, "token_embd.weight", qtype, 2, ne);
    }
    {
        const int64_t ne1[1] = {kNEmbd};
        add_f32_tensor(gc, mc, "output_norm.weight", 1, ne1, /*ones=*/true);
        const int64_t ne2[2] = {kNEmbd, n_vocab};
        add_quant_tensor(gc, mc, "output.weight", qtype, 2, ne2);
    }

    for (int64_t l = 0; l < kNLayer; ++l) {
        char p[32];
        std::snprintf(p, sizeof(p), "blk.%lld.", static_cast<long long>(l));
        const std::string pre = p;

        const int64_t ne_norm[1] = {kNEmbd};
        add_f32_tensor(gc, mc, (pre + "attn_norm.weight").c_str(), 1, ne_norm, true);

        const int64_t ne_attn[2] = {kNEmbd, kNEmbd};
        add_quant_tensor(gc, mc, (pre + "attn_q.weight").c_str(), qtype, 2, ne_attn);
        add_quant_tensor(gc, mc, (pre + "attn_k.weight").c_str(), qtype, 2, ne_attn);
        add_quant_tensor(gc, mc, (pre + "attn_v.weight").c_str(), qtype, 2, ne_attn);
        add_quant_tensor(gc, mc, (pre + "attn_output.weight").c_str(), qtype, 2, ne_attn);

        add_f32_tensor(gc, mc, (pre + "ffn_norm.weight").c_str(), 1, ne_norm, true);

        // Router stays F32 (small; keeps gating numerics clean).
        const int64_t ne_gate_inp[2] = {kNEmbd, kNExpert};
        add_f32_tensor(gc, mc, (pre + "ffn_gate_inp.weight").c_str(), 2,
                       ne_gate_inp, /*ones=*/false);

        // Stacked expert tensors: ggml ne is [inner, outer, n_expert]. Inner
        // dims are kNEmbd or kNFf, both multiples of 256, so each per-expert
        // slice is a whole number of K-quant blocks.
        const int64_t ne_gate[3] = {kNEmbd, kNFf, kNExpert};
        const int64_t ne_down[3] = {kNFf, kNEmbd, kNExpert};
        const int64_t ne_up[3] = {kNEmbd, kNFf, kNExpert};
        add_quant_tensor(gc, mc, (pre + "ffn_gate_exps.weight").c_str(), qtype, 3, ne_gate);
        add_quant_tensor(gc, mc, (pre + "ffn_down_exps.weight").c_str(), qtype, 3, ne_down);
        add_quant_tensor(gc, mc, (pre + "ffn_up_exps.weight").c_str(), qtype, 3, ne_up);
    }

    if (!gguf_write_to_file(gc, out.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "make-kquant-moe-gguf: failed to write '%s'\n",
                     out.c_str());
        gguf_free(gc);
        ggml_free(mc);
        ggml_quantize_free();
        return 1;
    }

    std::printf("wrote %s: vocab %lld, %lld layers, %lld experts (top-%lld), "
                "%s weights (n_embd=%lld n_ff=%lld)\n",
                out.c_str(), static_cast<long long>(n_vocab),
                static_cast<long long>(kNLayer),
                static_cast<long long>(kNExpert),
                static_cast<long long>(kNExpertUsed), type_name,
                static_cast<long long>(kNEmbd), static_cast<long long>(kNFf));

    gguf_free(gc);
    ggml_free(mc);
    ggml_quantize_free();
    return 0;
}

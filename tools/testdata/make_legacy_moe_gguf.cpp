// Generate a llama-arch MoE GGUF using the LEGACY, UN-STACKED, PER-EXPERT
// routed-expert FFN tensor naming that the real TheBloke Mixtral-8x7B GGUFs
// use: one 2-D tensor per (layer, kind, expert), named exactly
//   blk.N.ffn_gate.E.weight / blk.N.ffn_down.E.weight / blk.N.ffn_up.E.weight
// (3 kinds x n_expert experts per layer). This is the layout StrataFlow does
// NOT yet recognize (it only knows the newer STACKED blk.N.ffn_{gate,down,up}
// _exps.weight 3-D form). The existing make-mixtral-faithful-gguf emits ONLY
// the stacked form, which is why it never reproduced the real-model bug; this
// tool reproduces it offline.
//
// The tool ALSO emits a matched STACKED-equivalent GGUF (same model, same
// trunk/metadata/tokenizer) whose per-expert nb02 slice is BYTE-IDENTICAL to
// the legacy standalone tensor for the same (layer, kind, expert). Byte
// identity is achieved by quantizing every per-expert tensor from the SAME
// deterministic F32 source (fixed seed, fixed generation order) and then, for
// the stacked tensor, concatenating those exact per-expert quantized byte
// blocks (K-quant quantizes per row-block, so a stacked tensor is a pure
// concatenation of its per-expert bytes). llama's runtime loader CAN read the
// stacked GGUF (it serves as the oracle target) but THROWS on the legacy one
// (confirmed against vendored llama.cpp b11379: legacy->stacked happens only in
// the Python convert script, never at GGUF load).
//
// Usage:
//   make-legacy-moe-gguf <out_legacy.gguf> [out_stacked.gguf]
//                        [n_embd n_ff n_layer n_expert qtype]
//   defaults: n_embd=256 n_ff=512 n_layer=2 n_expert=8 qtype=q4_k  (CI-small)
//   If <out_stacked.gguf> is omitted it defaults to <out_legacy>.stacked.gguf.
//   qtype in {q3_k, q4_k, q6_k, f32}.
//
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier:
// LicenseRef-Coaade-Source-Available-1.0
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

// One global RNG so the fixture is deterministic across runs/compilers. Both
// the legacy and stacked outputs draw from the SAME sequence in the SAME order,
// which is what makes their per-expert quantized bytes bit-identical.
std::mt19937 g_rng(0);

void reset_rng() {
    g_rng.seed(0);
}

void fill_f32(std::vector<float> &buf, int64_t n) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    buf.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) buf[static_cast<size_t>(i)] = dist(g_rng) * 0.02f;
}

// A fully materialized tensor (quantized or F32) ready to hand to gguf. We build
// the raw bytes once from the deterministic F32 source so the legacy standalone
// tensors and the stacked concatenation share identical byte blocks.
struct BuiltTensor {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    int n_dims = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    std::vector<uint8_t> bytes;
};

// Quantize (or copy, for F32) `n_elem` floats with row length `n_per_row` into a
// freshly built byte buffer, exactly like make_kquant_moe_gguf.cpp /
// make_mixtral_faithful_gguf.cpp (ggml_quantize_chunk, row-by-row).
std::vector<uint8_t> quantize_rows(ggml_type qtype, const float *src, int64_t n_elem,
                                   int64_t n_per_row) {
    const int64_t nrows = n_elem / n_per_row;
    const size_t nbytes = ggml_row_size(qtype, n_per_row) * static_cast<size_t>(nrows);
    std::vector<uint8_t> out(nbytes);
    if (qtype == GGML_TYPE_F32) {
        std::memcpy(out.data(), src, static_cast<size_t>(n_elem) * sizeof(float));
        return out;
    }
    const size_t written =
        ggml_quantize_chunk(qtype, src, out.data(), 0, nrows, n_per_row, nullptr);
    if (written != nbytes) {
        std::fprintf(stderr,
                     "make-legacy-moe-gguf: quant size mismatch "
                     "(%zu vs %zu)\n",
                     written, nbytes);
        std::exit(1);
    }
    return out;
}

// Build one quantized 2-D standalone tensor from a fresh deterministic F32
// draw. `ne0` is the innermost/row dim (must be a multiple of 256 for K-quant),
// `ne1` the outer dim. Returns the built bytes so the caller can both register
// it as a legacy tensor AND append it to the matching stacked tensor.
std::vector<uint8_t> build_expert_2d(ggml_type qtype, int64_t ne0, int64_t ne1) {
    std::vector<float> src;
    fill_f32(src, ne0 * ne1);
    return quantize_rows(qtype, src.data(), ne0 * ne1, ne0);
}

BuiltTensor make_quant_trunk(ggml_type qtype, const std::string &name, int n_dims,
                             const int64_t *ne) {
    int64_t n_elem = 1;
    for (int d = 0; d < n_dims; ++d) n_elem *= ne[d];
    std::vector<float> src;
    fill_f32(src, n_elem);
    BuiltTensor bt;
    bt.name = name;
    bt.type = qtype;
    bt.n_dims = n_dims;
    for (int d = 0; d < n_dims; ++d) bt.ne[d] = ne[d];
    bt.bytes = quantize_rows(qtype, src.data(), n_elem, ne[0]);
    return bt;
}

BuiltTensor make_f32_trunk(const std::string &name, int n_dims, const int64_t *ne,
                           bool ones) {
    int64_t n_elem = 1;
    for (int d = 0; d < n_dims; ++d) n_elem *= ne[d];
    BuiltTensor bt;
    bt.name = name;
    bt.type = GGML_TYPE_F32;
    bt.n_dims = n_dims;
    for (int d = 0; d < n_dims; ++d) bt.ne[d] = ne[d];
    bt.bytes.resize(static_cast<size_t>(n_elem) * sizeof(float));
    float *dst = reinterpret_cast<float *>(bt.bytes.data());
    if (ones) {
        for (int64_t i = 0; i < n_elem; ++i) dst[i] = 1.0f;
    } else {
        std::vector<float> src;
        fill_f32(src, n_elem);
        std::memcpy(dst, src.data(), src.size() * sizeof(float));
    }
    return bt;
}

// Register a pre-built tensor into a (gguf, ggml) pair: a ggml tensor is created
// with the right type/shape, the pre-built bytes are copied into its data, and
// it is added to the gguf context. This keeps the bytes identical across the two
// output files (we build once, register the same bytes into each gguf).
void add_built(gguf_context *gc, ggml_context *mc, const BuiltTensor &bt) {
    ggml_tensor *t = ggml_new_tensor(mc, bt.type, bt.n_dims, bt.ne);
    ggml_set_name(t, bt.name.c_str());
    if (bt.bytes.size() != ggml_nbytes(t)) {
        std::fprintf(stderr,
                     "make-legacy-moe-gguf: byte size mismatch for %s "
                     "(%zu vs %zu)\n",
                     bt.name.c_str(), bt.bytes.size(), ggml_nbytes(t));
        std::exit(1);
    }
    std::memcpy(t->data, bt.bytes.data(), bt.bytes.size());
    gguf_add_tensor(gc, t);
}

// Populate every shared metadata key + the tokenizer identically on both the
// legacy and stacked gguf contexts, so the ONLY difference between the two files
// is the expert tensor naming/stacking.
void set_common_meta(gguf_context *gc, int64_t n_embd, int64_t n_ff, int64_t n_layer,
                     int64_t n_expert, int64_t n_head, int64_t head_dim,
                     int64_t n_expert_used, int32_t ftype,
                     const std::vector<std::string> &toks,
                     const std::vector<int32_t> &ttypes) {
    const int64_t n_vocab = static_cast<int64_t>(toks.size());
    gguf_set_val_str(gc, "general.architecture", "llama");
    gguf_set_val_str(gc, "general.name", "legacy-per-expert-moe");
    gguf_set_val_u32(gc, "llama.context_length", 256);
    gguf_set_val_u32(gc, "llama.embedding_length", static_cast<uint32_t>(n_embd));
    gguf_set_val_u32(gc, "llama.block_count", static_cast<uint32_t>(n_layer));
    gguf_set_val_u32(gc, "llama.feed_forward_length", static_cast<uint32_t>(n_ff));
    gguf_set_val_u32(gc, "llama.attention.head_count", static_cast<uint32_t>(n_head));
    gguf_set_val_u32(gc, "llama.attention.head_count_kv", static_cast<uint32_t>(n_head));
    gguf_set_val_u32(gc, "llama.rope.dimension_count", static_cast<uint32_t>(head_dim));
    gguf_set_val_f32(gc, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gc, "llama.expert_count", static_cast<uint32_t>(n_expert));
    gguf_set_val_u32(gc, "llama.expert_used_count", static_cast<uint32_t>(n_expert_used));
    gguf_set_val_u32(gc, "general.file_type", static_cast<uint32_t>(ftype));

    gguf_set_val_str(gc, "tokenizer.ggml.model", "llama");
    std::vector<const char *> tp;
    tp.reserve(toks.size());
    for (const std::string &s : toks) tp.push_back(s.c_str());
    gguf_set_arr_str(gc, "tokenizer.ggml.tokens", tp.data(), tp.size());
    std::vector<float> scores(static_cast<size_t>(n_vocab), 0.0f);
    gguf_set_arr_data(gc, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(),
                      scores.size());
    gguf_set_arr_data(gc, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, ttypes.data(),
                      ttypes.size());
    gguf_set_val_u32(gc, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(gc, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_u32(gc, "tokenizer.ggml.unknown_token_id", 0);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <out_legacy.gguf> [out_stacked.gguf] "
                     "[n_embd n_ff n_layer n_expert qtype]\n",
                     argv[0]);
        return 2;
    }
    const std::string out_legacy = argv[1];
    // argv[2] is the optional stacked output path, recognized only when it
    // looks like a .gguf path (otherwise it is the first numeric arg). When
    // omitted the stacked path defaults to <out_legacy>.stacked.gguf.
    const bool have_stacked_arg =
        argc > 2 && std::string(argv[2]).find(".gguf") != std::string::npos;
    const std::string out_stacked =
        have_stacked_arg ? std::string(argv[2]) : out_legacy + ".stacked.gguf";
    const int argbase = have_stacked_arg ? 3 : 2;
    const int64_t n_embd = argc > argbase + 0 ? std::atoll(argv[argbase + 0]) : 256;
    const int64_t n_ff = argc > argbase + 1 ? std::atoll(argv[argbase + 1]) : 512;
    const int64_t n_layer = argc > argbase + 2 ? std::atoll(argv[argbase + 2]) : 2;
    const int64_t n_expert = argc > argbase + 3 ? std::atoll(argv[argbase + 3]) : 8;
    const std::string qarg = argc > argbase + 4 ? argv[argbase + 4] : "q4_k";

    const int64_t n_head = (32 <= n_embd / 8) ? 32 : 4;
    const int64_t head_dim = n_embd / n_head;
    const int64_t n_expert_used = 2;

    ggml_type qtype = GGML_TYPE_Q4_K;
    int32_t ftype = 15;  // LLAMA_FTYPE_MOSTLY_Q4_K_M
    if (qarg == "q3_k") {
        qtype = GGML_TYPE_Q3_K;
        ftype = 12;
    } else if (qarg == "q6_k") {
        qtype = GGML_TYPE_Q6_K;
        ftype = 18;
    } else if (qarg == "q4_k") {
        qtype = GGML_TYPE_Q4_K;
        ftype = 15;
    } else if (qarg == "f32") {
        qtype = GGML_TYPE_F32;
        ftype = 0;
    } else {
        std::fprintf(stderr,
                     "make-legacy-moe-gguf: unknown qtype '%s' "
                     "(expected q3_k/q4_k/q6_k/f32)\n",
                     qarg.c_str());
        return 2;
    }

    // Vocab mirrors make_mixtral_faithful_gguf.cpp: 3 control + 256 byte-fallback
    // + a few normal pieces so the SPM tokenizer can resegment any input.
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
    const char *normal[] = {
        "\xe2\x96\x81"
        "hello",
        "\xe2\x96\x81"
        "world",
        "\xe2\x96\x81"
        "the",
        "\xe2\x96\x81"
        "a",
        "\xe2\x96\x81"
        "of"};
    for (const char *n : normal) {
        toks.emplace_back(n);
        ttypes.push_back(1);
    }
    const int64_t n_vocab = static_cast<int64_t>(toks.size());

    // Generously sized data contexts (one per output file) for the quantized
    // bytes + ggml tensor overhead.
    size_t mem = 512u * 1024u * 1024u;
    const size_t need = static_cast<size_t>(n_embd) * static_cast<size_t>(n_ff) *
                        static_cast<size_t>(n_expert) * static_cast<size_t>(n_layer) *
                        4u * 4u;
    if (need > mem) mem = need;
    ggml_init_params ip{};
    ip.mem_size = mem;
    ip.no_alloc = false;
    ggml_context *mc_legacy = ggml_init(ip);
    ggml_context *mc_stacked = ggml_init(ip);
    gguf_context *gc_legacy = gguf_init_empty();
    gguf_context *gc_stacked = gguf_init_empty();
    if (mc_legacy == nullptr || mc_stacked == nullptr) {
        std::fprintf(stderr, "make-legacy-moe-gguf: ggml_init failed\n");
        return 1;
    }

    set_common_meta(gc_legacy, n_embd, n_ff, n_layer, n_expert, n_head, head_dim,
                    n_expert_used, ftype, toks, ttypes);
    set_common_meta(gc_stacked, n_embd, n_ff, n_layer, n_expert, n_head, head_dim,
                    n_expert_used, ftype, toks, ttypes);

    // IMPORTANT: build every tensor in a FIXED order from the SAME RNG so the
    // legacy and stacked files draw identical floats. We reset the RNG and walk
    // the exact same sequence for each file; because the per-expert draws happen
    // in the same (layer, kind, expert) order, the quantized bytes match.
    //
    // Shared trunk tensors (identical in both files): built once from a reset
    // RNG, then registered into both gguf contexts.
    reset_rng();
    std::vector<BuiltTensor> trunk;
    {
        const int64_t ne[2] = {n_embd, n_vocab};
        trunk.push_back(make_quant_trunk(qtype, "token_embd.weight", 2, ne));
    }
    {
        const int64_t ne1[1] = {n_embd};
        trunk.push_back(make_f32_trunk("output_norm.weight", 1, ne1, true));
        const int64_t ne2[2] = {n_embd, n_vocab};
        trunk.push_back(make_quant_trunk(qtype, "output.weight", 2, ne2));
    }
    for (int64_t l = 0; l < n_layer; ++l) {
        char p[32];
        std::snprintf(p, sizeof(p), "blk.%lld.", static_cast<long long>(l));
        const std::string pre = p;
        const int64_t ne_norm[1] = {n_embd};
        trunk.push_back(make_f32_trunk(pre + "attn_norm.weight", 1, ne_norm, true));
        const int64_t ne_attn[2] = {n_embd, n_embd};
        trunk.push_back(make_quant_trunk(qtype, pre + "attn_q.weight", 2, ne_attn));
        trunk.push_back(make_quant_trunk(qtype, pre + "attn_k.weight", 2, ne_attn));
        trunk.push_back(make_quant_trunk(qtype, pre + "attn_v.weight", 2, ne_attn));
        trunk.push_back(make_quant_trunk(qtype, pre + "attn_output.weight", 2, ne_attn));
        trunk.push_back(make_f32_trunk(pre + "ffn_norm.weight", 1, ne_norm, true));
        const int64_t ne_gate_inp[2] = {n_embd, n_expert};
        trunk.push_back(
            make_f32_trunk(pre + "ffn_gate_inp.weight", 2, ne_gate_inp, false));
    }
    for (const BuiltTensor &bt : trunk) {
        add_built(gc_legacy, mc_legacy, bt);
        add_built(gc_stacked, mc_stacked, bt);
    }

    // Expert tensors. For each (layer, kind, expert) we build the per-expert 2-D
    // bytes ONCE (deterministic draw) and (1) register it as a legacy standalone
    // tensor, (2) append its bytes to the matching stacked 3-D tensor. Walking
    // the kinds/experts in a fixed order from the (continued) RNG stream keeps
    // the stacked expert-e nb02 slice byte-identical to the legacy tensor.
    //
    // Shapes: gate/up rows are [n_embd, n_ff] (ne0=n_embd), down is [n_ff,
    // n_embd] (ne0=n_ff). Stacked ne adds n_expert as the 3rd dim.
    struct Kind {
        const char *legacy;
        const char *stacked;
        int64_t ne0;
        int64_t ne1;
    };
    const Kind kinds[3] = {
        {"ffn_gate", "ffn_gate_exps", n_embd, n_ff},
        {"ffn_down", "ffn_down_exps", n_ff, n_embd},
        {"ffn_up", "ffn_up_exps", n_embd, n_ff},
    };

    for (int64_t l = 0; l < n_layer; ++l) {
        char p[32];
        std::snprintf(p, sizeof(p), "blk.%lld.", static_cast<long long>(l));
        const std::string pre = p;
        for (const Kind &k : kinds) {
            // Stacked tensor: ne = {ne0, ne1, n_expert}; its bytes are the
            // concatenation of the per-expert blocks, in expert order.
            BuiltTensor stacked;
            stacked.name = pre + k.stacked + ".weight";
            stacked.type = qtype;
            stacked.n_dims = 3;
            stacked.ne[0] = k.ne0;
            stacked.ne[1] = k.ne1;
            stacked.ne[2] = n_expert;
            const size_t per_expert_bytes =
                ggml_row_size(qtype, k.ne0) * static_cast<size_t>(k.ne1);
            stacked.bytes.reserve(per_expert_bytes * static_cast<size_t>(n_expert));

            for (int64_t e = 0; e < n_expert; ++e) {
                std::vector<uint8_t> ebytes = build_expert_2d(qtype, k.ne0, k.ne1);
                if (ebytes.size() != per_expert_bytes) {
                    std::fprintf(stderr,
                                 "make-legacy-moe-gguf: expert byte "
                                 "size mismatch\n");
                    return 1;
                }
                // Legacy standalone 2-D tensor: blk.N.ffn_<kind>.<e>.weight.
                char en[64];
                std::snprintf(en, sizeof(en), "%s%s.%lld.weight", pre.c_str(), k.legacy,
                              static_cast<long long>(e));
                BuiltTensor legacy;
                legacy.name = en;
                legacy.type = qtype;
                legacy.n_dims = 2;
                legacy.ne[0] = k.ne0;
                legacy.ne[1] = k.ne1;
                legacy.bytes = ebytes;  // copy before moving into stacked
                add_built(gc_legacy, mc_legacy, legacy);

                stacked.bytes.insert(stacked.bytes.end(), ebytes.begin(), ebytes.end());
            }
            add_built(gc_stacked, mc_stacked, stacked);
        }
    }

    if (!gguf_write_to_file(gc_legacy, out_legacy.c_str(), false)) {
        std::fprintf(stderr, "make-legacy-moe-gguf: write failed for %s\n",
                     out_legacy.c_str());
        return 1;
    }
    if (!gguf_write_to_file(gc_stacked, out_stacked.c_str(), false)) {
        std::fprintf(stderr, "make-legacy-moe-gguf: write failed for %s\n",
                     out_stacked.c_str());
        return 1;
    }

    std::printf(
        "wrote legacy=%s stacked=%s: n_embd=%lld n_ff=%lld n_layer=%lld "
        "n_expert=%lld (top-%lld) %s\n",
        out_legacy.c_str(), out_stacked.c_str(), (long long)n_embd, (long long)n_ff,
        (long long)n_layer, (long long)n_expert, (long long)n_expert_used, qarg.c_str());

    gguf_free(gc_legacy);
    gguf_free(gc_stacked);
    ggml_free(mc_legacy);
    ggml_free(mc_stacked);
    ggml_quantize_free();
    return 0;
}

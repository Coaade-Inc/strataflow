// SPIKE -- do not ship. Confirms the tiny MoE GGUF loads and decodes in plain
// llama.cpp (CPU), establishing the baseline token sequence the streaming
// variant must reproduce byte-identically.
#include <llama.h>
#include <cstdio>
#include <vector>
#include <limits>
#include <cstring>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tiny_moe.gguf";
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(path, mp);
    if (!model) { printf("FAIL load\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("FAIL ctx\n"); return 1; }

    // tokenize a prompt
    const char* prompt = "hello world";
    std::vector<llama_token> toks(32);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           toks.data(), (int)toks.size(), true, true);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt, 11, toks.data(), -n, true, true); }
    toks.resize(n);

    llama_batch batch = llama_batch_get_one(toks.data(), (int)toks.size());
    if (llama_decode(ctx, batch) != 0) { printf("FAIL decode prompt\n"); return 1; }

    printf("baseline tokens:");
    int32_t nv = llama_vocab_n_tokens(vocab);
    llama_token last = toks.back();
    for (int step = 0; step < 8; ++step) {
        float* logits = llama_get_logits_ith(ctx, -1);
        int32_t best = 0; float bl = -std::numeric_limits<float>::infinity();
        for (int32_t i = 0; i < nv; ++i) if (logits[i] > bl) { bl = logits[i]; best = i; }
        printf(" %d", best);
        last = best;
        llama_batch b2 = llama_batch_get_one(&last, 1);
        if (llama_decode(ctx, b2) != 0) { printf(" FAIL decode\n"); return 1; }
    }
    printf("\nOK\n");
    llama_free(ctx); llama_model_free(model);
    return 0;
}

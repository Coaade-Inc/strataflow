// StrataFlow CLI.
// Phase 1: load a model (dry-run), print the hardware/placement plan, and
// stream a generation. This is the smoke test for the whole engine wiring.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "strataflow/strataflow.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void print_usage(const char *argv0) {
    std::printf(
        "StrataFlow %s\n"
        "Usage: %s --model <path> [options] [-- <prompt>]\n"
        "\n"
        "Options:\n"
        "  --model PATH       model file (.gguf / .strata). Required.\n"
        "  --prompt TEXT      prompt text (or pass after --)\n"
        "  --vram-gb X        VRAM budget in GiB (0 = auto)\n"
        "  --ram-gb X         RAM budget in GiB (0 = auto)\n"
        "  --max-tokens N     generation cap (default 64)\n"
        "  --plan             print the placement plan and exit\n"
        "  --version          print version and exit\n"
        "  -h, --help         this help\n",
        sf_version(), argv0);
}

int on_token(const char *text, void *user_data) {
    (void)user_data;
    std::fputs(text, stdout);
    std::fflush(stdout);
    return 0;
}

uint64_t gib(double x) { return static_cast<uint64_t>(x * 1024.0 * 1024.0 * 1024.0); }

} // namespace

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt = "hello from strataflow";
    double vram_gb = 0, ram_gb = 0;
    int max_tokens = 64;
    bool plan_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "--version")      { std::printf("%s\n", sf_version()); return 0; }
        else if (a == "--model")        { model_path = next("--model"); }
        else if (a == "--prompt")       { prompt = next("--prompt"); }
        else if (a == "--vram-gb")      { vram_gb = std::atof(next("--vram-gb")); }
        else if (a == "--ram-gb")       { ram_gb = std::atof(next("--ram-gb")); }
        else if (a == "--max-tokens")   { max_tokens = std::atoi(next("--max-tokens")); }
        else if (a == "--plan")         { plan_only = true; }
        else if (a == "--")             { // rest is the prompt
            prompt.clear();
            for (int j = i + 1; j < argc; ++j) {
                if (!prompt.empty()) prompt += ' ';
                prompt += argv[j];
            }
            break;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "error: --model is required\n\n");
        print_usage(argv[0]);
        return 2;
    }

    sf_context_params params = sf_context_default_params();
    params.model_path  = model_path.c_str();
    params.vram_budget = gib(vram_gb);
    params.ram_budget  = gib(ram_gb);

    sf_context *ctx = nullptr;
    sf_status st = sf_context_create(&params, &ctx);
    if (st != SF_OK) {
        std::fprintf(stderr, "error: context create failed: %s\n", sf_status_str(st));
        return 1;
    }

    char plan[512];
    if (sf_describe_plan(ctx, plan, sizeof(plan)) == SF_OK) {
        std::printf("plan: %s\n", plan);
    }
    if (plan_only) { sf_context_free(ctx); return 0; }

    sf_session *session = nullptr;
    st = sf_session_create(ctx, &session);
    if (st != SF_OK) {
        std::fprintf(stderr, "error: session create failed: %s\n", sf_status_str(st));
        sf_context_free(ctx);
        return 1;
    }

    sf_sampling_params sampling = sf_sampling_default_params();
    sampling.max_tokens = max_tokens;

    std::printf("prompt: %s\noutput: ", prompt.c_str());
    std::fflush(stdout);
    st = sf_generate(session, prompt.c_str(), &sampling, on_token, nullptr);
    std::printf("\n");
    if (st != SF_OK) {
        std::fprintf(stderr, "error: generate failed: %s\n", sf_status_str(st));
    }

    sf_session_free(session);
    sf_context_free(ctx);
    return st == SF_OK ? 0 : 1;
}

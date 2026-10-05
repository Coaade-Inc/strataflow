// StrataFlow CLI.
// Phase 1: load a model (dry-run), print the hardware/placement plan, and
// stream a generation. This is the smoke test for the whole engine wiring.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "strataflow/strataflow.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// Carried through sf_generate's user_data so the token callback can measure
// time-to-first-token (TTFT): wall-clock from just before sf_generate to the
// FIRST produced token. No engine math is involved, so this is oracle-safe.
struct TokenTiming {
    std::chrono::steady_clock::time_point start;  // set just before sf_generate
    bool     seen_first = false;                  // first token recorded yet?
    uint64_t ttft_us    = 0;                       // measured TTFT, microseconds
};

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
        "  --cache-gb X       cap resident expert RAM to X GiB (0 = auto from\n"
        "                     free RAM). Caps the auto slot choice.\n"
        "  --expert-slots N   bound streamed experts to N bundles (0 = auto;\n"
        "                     a value below the model's expert count forces\n"
        "                     bounded streaming - run a big model in little RAM).\n"
        "                     Precedence: --expert-slots > --cache-gb > auto.\n"
        "  --max-tokens N     generation cap (default 64)\n"
        "  --plan             print the placement plan and exit\n"
        "  --version          print version and exit\n"
        "  -h, --help         this help\n",
        sf_version(), argv0);
}

int on_token(const char *text, void *user_data) {
    auto *t = static_cast<TokenTiming *>(user_data);
    if (t != nullptr && !t->seen_first) {
        const auto now = std::chrono::steady_clock::now();
        t->ttft_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(now - t->start)
                .count());
        t->seen_first = true;
    }
    std::fputs(text, stdout);
    std::fflush(stdout);
    return 0;
}

uint64_t gib(double x) { return static_cast<uint64_t>(x * 1024.0 * 1024.0 * 1024.0); }

} // namespace

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt = "hello from strataflow";
    double vram_gb = 0, ram_gb = 0, cache_gb = 0;
    uint32_t expert_slots = 0;
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
        else if (a == "--cache-gb")     { cache_gb = std::atof(next("--cache-gb")); }
        else if (a == "--expert-slots") { expert_slots = (uint32_t)std::strtoul(next("--expert-slots"), nullptr, 10); }
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
    params.model_path   = model_path.c_str();
    params.vram_budget  = gib(vram_gb);
    params.ram_budget   = gib(ram_gb);
    params.expert_slots = expert_slots;
    params.cache_budget = gib(cache_gb);  // 0 = auto; caps the auto slot choice

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
    TokenTiming timing;
    timing.start = std::chrono::steady_clock::now();
    st = sf_generate(session, prompt.c_str(), &sampling, on_token, &timing);
    std::printf("\n");
    if (st != SF_OK) {
        std::fprintf(stderr, "error: generate failed: %s\n", sf_status_str(st));
    }

    // Report the resident weight budget and peak RAM so the bounded-memory
    // claim is observable without an external tool (no /usr/bin/time needed).
    sf_runtime_stats rs;
    if (sf_session_stats(session, &rs) == SF_OK) {
        const double mib = 1024.0 * 1024.0;
        std::printf("stats: resident model weights = %.1f MiB",
                    static_cast<double>(rs.resident_weight_bytes) / mib);
        if (rs.peak_rss_bytes > 0) {
            std::printf(", peak RSS = %.1f MiB",
                        static_cast<double>(rs.peak_rss_bytes) / mib);
        }
        // Additive metrics on the SAME line (the existing wording above is
        // unchanged so the Python parser keeps working). TTFT comes from the
        // CLI wall-clock (only if a token was produced); streamed comes from
        // the engine ground truth (only on the streaming .strata path).
        if (timing.seen_first) {
            std::printf(", TTFT = %.1f ms",
                        static_cast<double>(timing.ttft_us) / 1000.0);
        }
        if (rs.streamed_bytes > 0) {
            // Human-readable MiB (unchanged wording) AND the exact uint64 byte
            // count so a harness can compute bytes/token without the 1-decimal
            // MiB rounding error. The raw field is additive: the existing
            // 'streamed = W MiB' field above is kept verbatim.
            std::printf(", streamed = %.1f MiB",
                        static_cast<double>(rs.streamed_bytes) / mib);
            std::printf(", streamed_bytes = %llu",
                        static_cast<unsigned long long>(rs.streamed_bytes));
        }
        std::printf("\n");
    }

    sf_session_free(session);
    sf_context_free(ctx);
    return st == SF_OK ? 0 : 1;
}

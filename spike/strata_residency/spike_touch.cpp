// SPIKE -- do not ship. Phase 3 Task 5 pivotal-question feasibility probe.
//
// Question: during a single llama_decode of a MoE graph, does the CPU kernel
// read ONLY the routed (top-k) experts' byte regions of each stacked expert
// tensor, or does something touch all n_expert regions?
//
// Method: own the expert tensors with a custom ggml buffer type backed by a
// page-aligned mmap region. Fill it with the real weights. Then mprotect every
// per-expert slice (src0->data + e*nb02 .. +nb02) to PROT_NONE and install a
// SIGSEGV handler that, on first touch of a slice, records {tensor,expert},
// re-enables the page (PROT_READ) and resumes. After each decode we print the
// distinct experts whose bytes were actually read. For a top-2 model only ~2
// experts per (layer,kind) stacked tensor should ever fault in.
//
// If confirmed: .strata CAN stream only top-k experts' bytes per token (I/O
// win). The RAM-residency question (addresses must stay valid) is separate and
// handled in the design doc.
#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>
#include "ggml-backend-impl.h"

#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

struct ExpertSlice { const ggml_tensor* t; int expert; uint8_t* base; size_t len; };

static std::vector<ExpertSlice> g_slices;              // one per (tensor,expert)
static std::set<std::pair<std::string,int>> g_touched; // distinct faults this window
static long g_pagesz = 4096;

// region management
static uint8_t* g_region = nullptr;
static size_t   g_region_sz = 0;

struct Range { uint64_t off; uint64_t len; };
static std::unordered_map<const ggml_tensor*, Range> g_range;
static uint64_t g_cursor = 0;

static void handler(int, siginfo_t* si, void*) {
    uint8_t* addr = (uint8_t*)si->si_addr;
    for (auto& s : g_slices) {
        if (addr >= s.base && addr < s.base + s.len) {
            // record and re-enable the whole slice
            g_touched.insert({ s.t->name, s.expert });
            uintptr_t pstart = (uintptr_t)s.base & ~(uintptr_t)(g_pagesz-1);
            uintptr_t pend   = ((uintptr_t)(s.base + s.len) + g_pagesz-1) & ~(uintptr_t)(g_pagesz-1);
            mprotect((void*)pstart, pend-pstart, PROT_READ | PROT_WRITE);
            return;
        }
    }
    // not ours: report where relative to the region, then fatal
    char buf[256];
    long rel = g_region ? (long)(addr - g_region) : -1;
    int nn = snprintf(buf, sizeof buf,
        "UNHANDLED SEGV addr=%p region=%p rel=%ld region_sz=%zu\n",
        (void*)addr, (void*)g_region, rel, g_region_sz);
    (void)!write(2, buf, nn);
    _exit(137);
}

static void protect_all_slices() {
    for (auto& s : g_slices) {
        uintptr_t pstart = (uintptr_t)s.base & ~(uintptr_t)(g_pagesz-1);
        uintptr_t pend   = ((uintptr_t)(s.base + s.len) + g_pagesz-1) & ~(uintptr_t)(g_pagesz-1);
        mprotect((void*)pstart, pend-pstart, PROT_NONE);
    }
}

static const char* bt_name(ggml_backend_buffer_type_t) { return "strata-touch"; }
static size_t bt_alignment(ggml_backend_buffer_type_t) { return g_pagesz; }
static size_t bt_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor* t) { return ggml_nbytes(t); }
static bool   bt_is_host(ggml_backend_buffer_type_t) { return true; }
static void   buf_free(ggml_backend_buffer_t) {}
static void*  buf_get_base(ggml_backend_buffer_t) { return g_region; }
static enum ggml_status buf_init_tensor(ggml_backend_buffer_t, ggml_tensor* t) {
    uint64_t off = (uint8_t*)t->data - g_region;
    g_range[t] = { off, (uint64_t)ggml_nbytes(t) };
    // record per-expert slices if this is a 3-D stacked expert tensor
    if (t->ne[2] > 1) {
        size_t nb02 = t->nb[2];
        for (int e = 0; e < t->ne[2]; ++e) {
            g_slices.push_back({ t, e, (uint8_t*)t->data + e*nb02, nb02 });
        }
    }
    return GGML_STATUS_SUCCESS;
}
static void buf_set_tensor(ggml_backend_buffer_t, ggml_tensor* t, const void* data, size_t offset, size_t size) {
    memcpy((uint8_t*)t->data + offset, data, size);
}
static void buf_get_tensor(ggml_backend_buffer_t, const ggml_tensor* t, void* data, size_t offset, size_t size) {
    memcpy(data, (const uint8_t*)t->data + offset, size);
}
static void buf_clear(ggml_backend_buffer_t, uint8_t) {}
static ggml_backend_buffer_t bt_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    g_region_sz = (size + g_pagesz-1) & ~(size_t)(g_pagesz-1);
    g_region = (uint8_t*)mmap(nullptr, g_region_sz, PROT_READ|PROT_WRITE,
                              MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    static ggml_backend_buffer_i iface = {};
    iface.free_buffer = buf_free; iface.get_base = buf_get_base;
    iface.init_tensor = buf_init_tensor; iface.set_tensor = buf_set_tensor;
    iface.get_tensor = buf_get_tensor; iface.clear = buf_clear;
    return ggml_backend_buffer_init(buft, iface, nullptr, size);
}
static ggml_backend_buffer_type g_buft = {
    { bt_name, bt_alloc_buffer, nullptr, bt_alignment, nullptr, bt_alloc_size, nullptr, bt_is_host },
    nullptr, nullptr
};

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/projects/sandbox/moe.gguf";
    g_pagesz = sysconf(_SC_PAGESIZE);

    struct sigaction sa{}; sa.sa_sigaction = handler; sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask); sigaction(SIGSEGV, &sa, nullptr);

    g_buft.device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    static llama_model_tensor_buft_override ov[2];
    ov[0].pattern = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    ov[0].buft = &g_buft;
    ov[1].pattern = nullptr; ov[1].buft = nullptr;
    mp.tensor_buft_overrides = ov;

    llama_model* model = llama_model_load_from_file(path, mp);
    if (!model) { printf("FAIL load\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params(); cp.n_ctx = 64;
    cp.n_threads = 1; cp.n_threads_batch = 1;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("FAIL ctx\n"); return 1; }

    printf("expert stacked tensors: %zu slices (per-expert) across all stacked tensors\n", g_slices.size());

    const char* prompt = "hello world";
    std::vector<llama_token> toks(32);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt), toks.data(), (int)toks.size(), true, true);
    toks.resize(n);

    // prompt decode: leave pages readable (prefill touches many experts)
    llama_batch batch = llama_batch_get_one(toks.data(), (int)toks.size());
    if (llama_decode(ctx, batch) != 0) { printf("FAIL decode prompt\n"); return 1; }

    int32_t nv = llama_vocab_n_tokens(vocab);
    llama_token last = toks.back();
    for (int step = 0; step < 4; ++step) {
        float* logits = llama_get_logits_ith(ctx, -1);
        int32_t best = 0; float bl = -std::numeric_limits<float>::infinity();
        for (int32_t i = 0; i < nv; ++i) if (logits[i] > bl) { bl = logits[i]; best = i; }
        last = best;

        // Arm the probe: poison every per-expert slice, decode ONE token, see
        // which slices fault in.
        g_touched.clear();
        protect_all_slices();
        llama_batch b2 = llama_batch_get_one(&last, 1);
        int rc = llama_decode(ctx, b2);
        // re-enable everything for the next arming
        for (auto& s : g_slices) {
            uintptr_t ps = (uintptr_t)s.base & ~(uintptr_t)(g_pagesz-1);
            uintptr_t pe = ((uintptr_t)(s.base+s.len)+g_pagesz-1)&~(uintptr_t)(g_pagesz-1);
            mprotect((void*)ps, pe-ps, PROT_READ|PROT_WRITE);
        }
        if (rc != 0) { printf("FAIL decode step %d\n", step); return 1; }
        printf("token=%d  distinct expert-slices touched=%zu : ", best, g_touched.size());
        for (auto& p : g_touched) printf("%s[e%d] ", p.first.c_str(), p.second);
        printf("\n");
    }
    llama_free(ctx); llama_model_free(model);
    return 0;
}

# Task 5 blocker: how does llama.cpp load the trunk from a `.strata`?

Status: the WIP implementation on branch `phase-3/task-5-strata-format`
(commit `2dbfb21`) builds clean and all 11 prior tests pass, but a packed
`.strata` **fails the byte-identical gate**: it loads as the dry-run fallback and
produces different tokens.

## Root cause (architectural, not a typo)

`load_ggml_model()` reads the model shape from the `.strata` embedded metadata
correctly, but then still calls:

```
llama_model_load_from_file(path /* the .strata */, mp);
```

llama.cpp's GGUF loader rejects it:

```
gguf_init_from_reader: invalid magic characters: 'STRA', expected 'GGUF'
```

so the whole load fails and falls back to the dry-run model.

The design (`docs/TASK5_DESIGN.md` section 2.3) assumed llama could parse the
embedded GGUF metadata prefix. But llama's loader needs a file it recognizes as
GGUF **with the trunk tensor DATA laid out at `data_offset`**. The `.strata`
packer embeds only `[0, data_offset)` (header + KV + tensor-info, no tensor
data) and puts the trunk tensor data in a separate aligned region. Nothing in
the load path gives llama a GGUF it can actually read the trunk weights from.

## Options found in the vendored llama.cpp (tag b11379)

`include/llama.h` exposes three loaders:

1. `llama_model_load_from_file(const char* path, params)` - current call; needs
   a real GGUF file.
2. `llama_model_load_from_file_ptr(FILE* file, params)` - load from an open
   FILE*. Could point at a reconstructed in-memory/temp GGUF.
3. `llama_model_load_from_splits(const char** paths, size_t n, params)` - load
   from multiple GGUF parts.

## Candidate resolutions (pick one; decide before resuming)

- **A. `.strata` is a sidecar, source GGUF stays authoritative.** Keep loading
  the original `.gguf` with llama (trunk loads normally; experts are routed to
  `strata_stream_buft` as today), and use the `.strata` ONLY as the aligned
  per-expert byte source for streaming reads. Simplest, lowest-risk, keeps
  byte-identical by construction. Cost: two files (`.gguf` + `.strata`), not the
  single-file story. Good v1.
- **B. Pack trunk as a complete standalone GGUF inside `.strata`.** Make the
  embedded region a FULL valid GGUF (metadata + trunk tensor data at its
  `data_offset`), then load it via `llama_model_load_from_file_ptr` on a FILE*
  positioned at `gguf_meta_offset`, or extract it to a temp GGUF. Single file,
  but the embedded GGUF must omit/zero the expert tensors (they stream) while
  keeping llama's tensor-info happy - needs care so llama does not try to read
  expert data from the embedded GGUF.
- **C. Two GGUFs via splits.** Pack trunk as one GGUF and experts as another,
  load via `llama_model_load_from_splits`. Fights the custom aligned layout;
  least attractive.

## Recommendation

Start with **A** (sidecar): it delivers the aligned-streaming I/O win and the
memory dial with the least risk and an unbreakable byte-identical guarantee,
and it is a small change from the current WIP (stop handing llama the `.strata`;
hand it the source `.gguf`, keep the `StrataReader` as the expert byte source).
Revisit **B** (true single-file) once A is green and benchmarked.

## Also missing

`tests/test_strata_pack.cpp` (the byte-identical `.strata`-vs-GGUF decode gate)
was never added - its absence is why this gap reached a "done" report. Add it
first when resuming so the fix is test-driven.

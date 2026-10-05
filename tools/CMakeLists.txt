# StrataFlow command-line tools.
# Phase 1: `strataflow` CLI. Phase 3 Task 5 / engine-core EC-4: `strata-pack`
# (GGUF -> .strata). The OpenAI-compatible server, strata-bench and
# strata-calibrate are added in later phases.

add_executable(strataflow-cli strataflow-cli/main.cpp)
set_target_properties(strataflow-cli PROPERTIES OUTPUT_NAME strataflow)
target_link_libraries(strataflow-cli PRIVATE StrataFlow::strataflow strataflow_warnings)

# strata-pack: the GGUF -> .strata packer (docs/TASK5_DESIGN.md section 5,
# docs/ENGINE_CORE_DESIGN.md section 5). The packing logic lives in the library
# (src/tws/strata_pack.*), so this tool is a thin CLI wrapper that only needs
# our public src include dir and the static library.
add_executable(strata-pack strata-pack/main.cpp)
target_link_libraries(strata-pack PRIVATE StrataFlow::strataflow strataflow_warnings)
target_include_directories(strata-pack PRIVATE ${CMAKE_SOURCE_DIR}/src)

# make-kquant-moe-gguf: a testdata generator that emits a tiny K-quant (Q4_K /
# Q6_K) llama-arch MoE GGUF. The Python gguf library cannot quantize to
# K-quants, so this tool quantizes with the vendored ggml (ggml_quantize_chunk)
# and writes with gguf. It links the vendored ggml target directly (the one
# place a tool needs ggml); the headers are consumed SYSTEM like the core lib
# does, so our -Werror does not flag the vendored headers.
# strataflow_cxx carries cxx_std_20: every first-party target requests C++20
# through it (the src targets link it directly, strata-pack/strataflow-cli
# inherit it via StrataFlow::strataflow). The project forbids a global
# CMAKE_CXX_STANDARD, so this tool must link it too rather than rely on the
# compiler default.
add_executable(make-mixtral-faithful-gguf testdata/make_mixtral_faithful_gguf.cpp)
target_link_libraries(make-mixtral-faithful-gguf PRIVATE ggml strataflow_cxx strataflow_warnings)
target_include_directories(make-mixtral-faithful-gguf SYSTEM PRIVATE
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>)

add_executable(make-kquant-moe-gguf testdata/make_kquant_moe_gguf.cpp)
target_link_libraries(make-kquant-moe-gguf PRIVATE ggml strataflow_cxx strataflow_warnings)
target_include_directories(make-kquant-moe-gguf SYSTEM PRIVATE
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>)

# make-legacy-moe-gguf: emits a llama-arch MoE GGUF using the LEGACY,
# un-stacked, PER-EXPERT routed-expert tensor naming (blk.N.ffn_{gate,down,up}
# .E.weight, 2-D each) that the real TheBloke Mixtral GGUFs use and that the
# existing make-mixtral-faithful-gguf never emitted. It ALSO emits a matched
# STACKED-equivalent GGUF whose per-expert nb02 slices are byte-identical to the
# legacy standalone tensors (the oracle target llama can load). Same ggml
# K-quant path and SYSTEM include seam as make-kquant-moe-gguf.
add_executable(make-legacy-moe-gguf testdata/make_legacy_moe_gguf.cpp)
target_link_libraries(make-legacy-moe-gguf PRIVATE ggml strataflow_cxx strataflow_warnings)
target_include_directories(make-legacy-moe-gguf SYSTEM PRIVATE
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>)

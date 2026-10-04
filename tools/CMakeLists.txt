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
add_executable(make-kquant-moe-gguf testdata/make_kquant_moe_gguf.cpp)
target_link_libraries(make-kquant-moe-gguf PRIVATE ggml strataflow_cxx strataflow_warnings)
target_include_directories(make-kquant-moe-gguf SYSTEM PRIVATE
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>)

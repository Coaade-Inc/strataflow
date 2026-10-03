# StrataFlow command-line tools.
# Phase 1: `strataflow` CLI. Phase 3 Task 5: `strata-pack` (GGUF -> .strata).
# The OpenAI-compatible server, strata-bench and strata-calibrate are added in
# later phases.

add_executable(strataflow-cli strataflow-cli/main.cpp)
set_target_properties(strataflow-cli PROPERTIES OUTPUT_NAME strataflow)
target_link_libraries(strataflow-cli PRIVATE StrataFlow::strataflow strataflow_warnings)

# strata-pack: the GGUF -> .strata packer (docs/TASK5_DESIGN.md section 5). It
# reads the source GGUF's metadata/tensor-info through the vendored gguf_* API,
# so it needs the ggml include dirs. Those are consumed as SYSTEM headers (same
# as src/) to keep our -Werror flags off the vendored headers. It links our
# static library for parse_expert_tensor_name + the .strata format definitions.
add_executable(strata-pack strata-pack/main.cpp)
target_link_libraries(strata-pack PRIVATE StrataFlow::strataflow strataflow_warnings)
target_link_libraries(strata-pack PRIVATE llama)
target_include_directories(strata-pack PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(strata-pack SYSTEM PRIVATE
    $<TARGET_PROPERTY:llama,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>)

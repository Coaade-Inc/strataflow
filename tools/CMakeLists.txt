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

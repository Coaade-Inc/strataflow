# StrataFlow command-line tools.
# Phase 1: `strataflow` CLI. The OpenAI-compatible server, strata-pack,
# strata-bench and strata-calibrate are added in later phases.

add_executable(strataflow-cli strataflow-cli/main.cpp)
set_target_properties(strataflow-cli PROPERTIES OUTPUT_NAME strataflow)
target_link_libraries(strataflow-cli PRIVATE StrataFlow::strataflow strataflow_warnings)

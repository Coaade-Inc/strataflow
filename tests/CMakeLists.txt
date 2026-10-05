# StrataFlow tests. Zero external deps: a tiny header-only check macro + ctest.

function(sf_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE StrataFlow::strataflow strataflow_warnings)
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

sf_add_test(test_slot_pool)
sf_add_test(test_planner)
sf_add_test(test_llama_placement)
sf_add_test(test_predictor)
sf_add_test(test_api)
sf_add_test(test_model_load)
sf_add_test(test_stream_buft)
sf_add_test(test_profiler_ssd)
sf_add_test(test_moe_stream)
sf_add_test(test_streaming_decode)
sf_add_test(test_async_io)
sf_add_test(test_engine_oracle)
sf_add_test(test_strata_decode)

# test_mixtral_faithful_pack additionally synthesizes a misnamed-MoE GGUF
# in-process (the Defect-1 loud-failure negative assertion), so it needs the
# ggml/gguf writer headers. ggml is linked PRIVATE into StrataFlow and does not
# propagate its include dirs, so add them SYSTEM to just this target (vendored
# headers, exempt from our -Werror like the core lib and the generators do).
sf_add_test(test_mixtral_faithful_pack)
target_link_libraries(test_mixtral_faithful_pack PRIVATE ggml)
target_include_directories(test_mixtral_faithful_pack SYSTEM PRIVATE
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>)

# test_legacy_per_expert_pack packs the FEAT-002 legacy per-expert fixture to a
# .strata and decodes it through the engine against the libllama oracle on the
# matched stacked fixture (the FEAT-003 core-fix gate). It only drives the
# library (pack + engine + oracle) and reads pre-generated fixtures from disk,
# so it needs no ggml writer headers - a plain sf_add_test suffices.
sf_add_test(test_legacy_per_expert_pack)

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

# test_kv_paging is the FEAT-002 lossless KV-paging gate (technique 6.5): it
# forces a SMALL resident window (STRATAFLOW_ENGINE_KV_WINDOW) and decodes a
# prompt + N tokens longer than the window so cold KV is paged out and re-
# assembled, proving the paged decode is byte-identical to the full-resident
# engine (per-step logit max|delta|==0.0) AND to the libllama oracle, while
# resident_kv_bytes() < full_kv_bytes() and the cold-fetch path actually ran.
# Covers the F32 MoE and K-quant fixtures. Drives only the library (engine +
# oracle), so a plain sf_add_test suffices.
sf_add_test(test_kv_paging)

# test_trunk_stream is the FEAT-003 "stream the trunk" gate (technique 6.4): it
# packs a fixture to a v2 trunk-indexed .strata and proves the trunk-streaming
# decode is byte-identical to the fully-resident decode (same file, toggled via
# STRATAFLOW_ENGINE_NO_TRUNK_STREAM) and to the libllama oracle, on both an F32
# and a K-quant fixture; it also asserts resident_weight_bytes drops with
# streaming and that a v1 .strata still opens with no trunk index. Drives only
# the library (pack + engine + reader + oracle), so a plain sf_add_test suffices.
sf_add_test(test_trunk_stream)

# test_trunk_stream_adaptive is the FEAT-003 Part B real-load-path gate for
# ADAPTIVE per-layer trunk streaming (technique 6.4): it packs a v2 .strata and
# (A) proves the engine streaming the per-layer attention projections with a
# bounded hot pool decodes byte-identically to the fully-resident trunk (same
# file, STRATAFLOW_ENGINE_NO_TRUNK_STREAM) and to the oracle over a short AND a
# longer sequence, with resident_weight_bytes strictly lower; and (B) proves the
# planner engages the per-layer trunk stream BY DEFAULT from the conservative
# budget on the real sf::load_model path (plan.trunk_pool_layers in
# (0, n_layers)) and that the plan-driven decode stays bit-identical. Only drives
# the library (pack + load + plan + engine + oracle), so a plain sf_add_test
# suffices.
sf_add_test(test_trunk_stream_adaptive)

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

# test_strata_shape_prepass is the regression gate for the .strata MoE shape
# pre-pass fix: it packs a legacy GGUF to .strata and asserts the loaded plan is
# MoE+streaming (NOT the dense branch). Only drives the library (pack + load +
# plan), so a plain sf_add_test suffices (no extra ggml linkage).
sf_add_test(test_strata_shape_prepass)

# test_kv_window_real_path is the FEAT-002 Part A real-load-path gate: it packs
# the Mixtral-geometry legacy fixture to .strata, loads it through sf::load_model
# under a simulated budget, and proves the lossless auto KV window ENGAGES BY
# DEFAULT (no STRATAFLOW_ENGINE_KV_WINDOW) - geometry reaches the planner, the
# plan chooses kv_window in (0, n_ctx), and an engine loaded with that plan
# window (via the new Engine::load parameter) reserves less than full KV while
# decoding bit-identically to the full-resident run and the libllama oracle.
# Only drives the library (pack + load + plan + engine + oracle), so a plain
# sf_add_test suffices.
sf_add_test(test_kv_window_real_path)

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
sf_add_test(test_predictor)
sf_add_test(test_api)
sf_add_test(test_model_load)

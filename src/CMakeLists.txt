# ---------------------------------------------------------------------------
# StrataFlow core library.
#
# Each subdirectory is a module from docs/PLAN.md section 5. For Phase 1 they
# are compiled into one static+shared library `strataflow`. The seams where
# ggml/llama plug in (model/, kernels/) are isolated so the Phase 1b vendoring
# touches as few modules as possible.
# ---------------------------------------------------------------------------

set(STRATAFLOW_SOURCES
    common/log.cpp
    common/status.cpp
    hw/profiler.cpp
    plan/planner.cpp
    tws/async_io.cpp
    tws/weight_store.cpp
    predict/predictor.cpp
    sched/scheduler.cpp
    kv/kv_store.cpp
    model/model.cpp
    model/ggml_model.cpp
    api/strataflow_api.cpp
)

add_library(strataflow ${STRATAFLOW_SOURCES})
add_library(StrataFlow::strataflow ALIAS strataflow)

target_include_directories(strataflow
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(strataflow PRIVATE STRATAFLOW_BUILD)
target_compile_definitions(strataflow PRIVATE STRATAFLOW_PLATFORM_${STRATAFLOW_PLATFORM})
target_link_libraries(strataflow PRIVATE strataflow_warnings)

find_package(Threads REQUIRED)
target_link_libraries(strataflow PUBLIC Threads::Threads)

# Vendored llama.cpp backend (Phase 1b). Linked PRIVATE: the `llama` target and
# its transitive `ggml` libraries stay an implementation detail of GgmlModel and
# do not leak into our public C ABI. The llama.cpp headers are consumed only by
# src/model/ggml_model.cpp.
#
# llama.cpp's public headers (llama.h / ggml*.h) do not compile cleanly under
# our strict -Werror -Wshadow -Wconversion flags, so consume their include
# directories as SYSTEM headers: that keeps warnings-as-errors on OUR code
# while silencing diagnostics from the vendored headers. We re-expose the
# llama/ggml INTERFACE_INCLUDE_DIRECTORIES as SYSTEM because target_link_libraries
# alone would propagate them as normal (warned) includes.
target_link_libraries(strataflow PRIVATE llama)
target_include_directories(strataflow SYSTEM PRIVATE
    $<TARGET_PROPERTY:llama,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>)

# Backend wiring (Phase 1b): when a backend option is ON, define a macro and,
# later, link the vendored ggml backend target. Kept inert for now.
foreach(bk CUDA VULKAN METAL HIP)
    if(STRATAFLOW_${bk})
        target_compile_definitions(strataflow PRIVATE STRATAFLOW_HAVE_${bk}=1)
        message(STATUS "StrataFlow: ${bk} backend requested (ggml wiring pending)")
    endif()
endforeach()

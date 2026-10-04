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
    plan/llama_placement.cpp
    tws/async_io.cpp
    tws/stream_buft.cpp
    tws/strata_file.cpp
    tws/strata_pack.cpp
    tws/weight_store.cpp
    predict/predictor.cpp
    sched/scheduler.cpp
    kv/kv_store.cpp
    model/model.cpp
    model/ggml_model.cpp
    model/engine/engine.cpp
    api/strataflow_api.cpp
)

# Build STATIC explicitly. A shared build on Windows produces no import .lib
# unless the C API symbols are dllexport'd (SF_API only exports when
# STRATAFLOW_SHARED is defined), which broke the MSVC link. Static is all we
# need for now; a proper shared build with an export map is a later concern.
add_library(strataflow STATIC ${STRATAFLOW_SOURCES})
add_library(StrataFlow::strataflow ALIAS strataflow)

target_include_directories(strataflow
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(strataflow PRIVATE STRATAFLOW_BUILD)
target_compile_definitions(strataflow PRIVATE STRATAFLOW_PLATFORM_${STRATAFLOW_PLATFORM})
target_link_libraries(strataflow PUBLIC strataflow_cxx)
target_link_libraries(strataflow PRIVATE strataflow_warnings)

# ---------------------------------------------------------------------------
# Optional liburing (Linux) for the io_uring + O_DIRECT async backend in
# tws/async_io.cpp. This is strictly optional: when liburing is absent the
# backend falls back to O_DIRECT + pread (and then plain pread), so the build
# must succeed with or without it. We never add a hard dependency.
#
# Detection: pkg-config first (gives correct link flags on most distros), then
# a manual find_path/find_library. When found we define STRATAFLOW_HAVE_IOURING
# and link uring; the io_uring code in async_io.cpp is entirely under that macro.
# ---------------------------------------------------------------------------
if(STRATAFLOW_PLATFORM STREQUAL "linux")
    set(STRATAFLOW_IOURING_FOUND FALSE)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LIBURING QUIET liburing)
    endif()
    if(LIBURING_FOUND)
        target_compile_definitions(strataflow PRIVATE STRATAFLOW_HAVE_IOURING=1)
        target_include_directories(strataflow SYSTEM PRIVATE ${LIBURING_INCLUDE_DIRS})
        target_link_libraries(strataflow PRIVATE ${LIBURING_LIBRARIES})
        set(STRATAFLOW_IOURING_FOUND TRUE)
    else()
        find_path(LIBURING_INCLUDE_DIR liburing.h)
        find_library(LIBURING_LIBRARY uring)
        if(LIBURING_INCLUDE_DIR AND LIBURING_LIBRARY)
            target_compile_definitions(strataflow PRIVATE STRATAFLOW_HAVE_IOURING=1)
            target_include_directories(strataflow SYSTEM PRIVATE ${LIBURING_INCLUDE_DIR})
            target_link_libraries(strataflow PRIVATE ${LIBURING_LIBRARY})
            set(STRATAFLOW_IOURING_FOUND TRUE)
        endif()
    endif()
    if(STRATAFLOW_IOURING_FOUND)
        message(STATUS "StrataFlow: liburing found; io_uring+O_DIRECT backend enabled")
    else()
        message(STATUS "StrataFlow: liburing not found; using O_DIRECT+pread backend")
    endif()
endif()

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

# tws/stream_buft.cpp implements ggml_backend_buffer_type_i / _buffer_i, whose
# definitions live in the PRIVATE ggml header ggml-backend-impl.h under
# ggml/src. The public ggml include dirs above do not cover that path, so add
# it as a SYSTEM include (vendored header, not subject to our -Werror).
target_include_directories(strataflow SYSTEM PRIVATE
    ${STRATAFLOW_LLAMA_DIR}/ggml/src)

# Backend wiring (Phase 1b): when a backend option is ON, define a macro and,
# later, link the vendored ggml backend target. Kept inert for now.
foreach(bk CUDA VULKAN METAL HIP)
    if(STRATAFLOW_${bk})
        target_compile_definitions(strataflow PRIVATE STRATAFLOW_HAVE_${bk}=1)
        message(STATUS "StrataFlow: ${bk} backend requested (ggml wiring pending)")
    endif()
endforeach()

cmake_minimum_required(VERSION 3.20)

project(StrataFlow
    VERSION 0.0.1
    DESCRIPTION "Tiered VRAM/RAM/SSD streaming inference engine for large MoE models"
    LANGUAGES CXX C)

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------
# NOTE: do NOT set CMAKE_CXX_STANDARD globally. StrataFlow is C++20, but the
# vendored llama.cpp (added via add_subdirectory) is written for C++17 and does
# not compile under C++20 (u8"" string literals become char8_t, which breaks
# its chat-template code on MSVC/Clang). A global standard would leak into the
# submodule. Instead we apply C++20 per-target on our own targets only, via the
# `strataflow_cxx` interface defined below, and let llama.cpp use its own
# per-target standard.
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(STRATAFLOW_BUILD_TESTS   "Build StrataFlow tests"            ON)
option(STRATAFLOW_BUILD_TOOLS   "Build StrataFlow CLI/server tools" ON)
option(STRATAFLOW_WARNINGS      "Enable extra warnings"             ON)
option(STRATAFLOW_WERROR        "Treat warnings as errors"          OFF)

# ---------------------------------------------------------------------------
# GPU compute backend selection.
# ---------------------------------------------------------------------------
# STRATAFLOW_GPU is the single, backend-agnostic knob. It maps to exactly one
# vendored ggml backend (GGML_CUDA / GGML_METAL / GGML_VULKAN) and defaults to
# "off", which is BYTE-IDENTICAL to the historical CPU-only build (all GGML_<...>
# backends forced OFF, zero GPU code compiled or linked). See the force-off
# block below the llama.cpp knobs for the hard-constraint comment.
#
# Precedence (resolved into STRATAFLOW_GPU_RESOLVED below):
#   1. If STRATAFLOW_GPU is set to a non-"off" value, it wins.
#   2. Otherwise, the lower-level per-backend booleans STRATAFLOW_CUDA /
#      STRATAFLOW_VULKAN / STRATAFLOW_METAL (kept for backward compatibility and
#      used by the existing release-cuda/vulkan/metal presets) select the
#      backend. At most one may be ON.
#   3. Otherwise the backend is "off" (CPU-only, the default).
set(STRATAFLOW_GPU "off" CACHE STRING
    "GPU compute backend: off|cuda|metal|vulkan (default off = CPU-only)")
set_property(CACHE STRATAFLOW_GPU PROPERTY STRINGS off cuda metal vulkan)

# Backend options (lower-level per-backend toggles; honored when STRATAFLOW_GPU
# is "off"). HIP/ROCm is accepted as a CUDA-family alias here but is not yet a
# first-class STRATAFLOW_GPU value.
option(STRATAFLOW_CUDA   "Enable CUDA backend (via ggml)"   OFF)
option(STRATAFLOW_VULKAN "Enable Vulkan backend (via ggml)" OFF)
option(STRATAFLOW_METAL  "Enable Metal backend (via ggml)"  OFF)
option(STRATAFLOW_HIP    "Enable HIP/ROCm backend (via ggml)" OFF)

# ---------------------------------------------------------------------------
# Warning flags (portable across GCC/Clang/MSVC)
# ---------------------------------------------------------------------------
# C++20 for StrataFlow's own targets only (kept off the vendored llama.cpp).
add_library(strataflow_cxx INTERFACE)
target_compile_features(strataflow_cxx INTERFACE cxx_std_20)

add_library(strataflow_warnings INTERFACE)
if(STRATAFLOW_WARNINGS)
    if(MSVC)
        target_compile_options(strataflow_warnings INTERFACE /W4)
        if(STRATAFLOW_WERROR)
            target_compile_options(strataflow_warnings INTERFACE /WX)
        endif()
    else()
        target_compile_options(strataflow_warnings INTERFACE
            -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-unused-parameter)
        if(STRATAFLOW_WERROR)
            target_compile_options(strataflow_warnings INTERFACE -Werror)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Platform detection -> compile definitions used by the async-IO layer
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(STRATAFLOW_PLATFORM "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(STRATAFLOW_PLATFORM "windows")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(STRATAFLOW_PLATFORM "macos")
else()
    set(STRATAFLOW_PLATFORM "generic")
endif()
message(STATUS "StrataFlow platform: ${STRATAFLOW_PLATFORM}")

# MSVC: use the portable CRT names (getenv, fopen, ...) without deprecation
# warnings, and don't define min/max macros that collide with std::min/max.
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# ---------------------------------------------------------------------------
# Vendored llama.cpp / ggml (Phase 1b)
#
# llama.cpp is pinned as a git submodule under third_party/llama.cpp (tag
# b11379). We take its `llama` and `ggml` library targets and build on top.
# A few rules keep our build lean and portable:
#   * CPU backend only by default. GPU backends are opt-in in later phases and
#     are wired through the STRATAFLOW_<CUDA|VULKAN|METAL|HIP> options below.
#   * Turn OFF llama.cpp's own tools/tests/examples/server and the common/curl
#     utility library so the submodule contributes just the two libraries.
#   * Do NOT apply our strataflow_warnings interface (-Werror -Wconversion ...)
#     to the vendored targets: llama.cpp does not compile cleanly under them.
#     It brings its own warning flags via add_subdirectory. Only OUR code links
#     strataflow_warnings.
#   * GGML_NATIVE is forced OFF for reproducible, portable binaries in CI.
# ---------------------------------------------------------------------------
set(STRATAFLOW_LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp)
if(NOT EXISTS ${STRATAFLOW_LLAMA_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
        "third_party/llama.cpp is missing. Initialize the submodule:\n"
        "  git submodule update --init --recursive")
endif()

# llama.cpp build knobs (set before add_subdirectory so they take effect).
# Build llama/ggml static so the CLI and tests are self-contained — avoids
# Windows DLL-search-path issues for the vendored libraries in CI.
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)  # drops the curl/common deps
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# Compute backend wiring (HARD CONSTRAINT).
#
#   * STRATAFLOW_GPU=off is the DEFAULT and is BYTE-IDENTICAL to the historical
#     CPU-only build: GGML_CUDA / GGML_VULKAN / GGML_METAL / GGML_HIP are all
#     forced OFF below (Metal defaults ON on Apple, so forcing it OFF matters),
#     so ZERO GPU code is compiled or linked on any platform.
#   * Selecting a backend only toggles the vendored ggml GGML_<...> option from
#     THIS top-level CMake, BEFORE add_subdirectory. third_party/llama.cpp is
#     NEVER edited; the GGML_<...> options are the only sanctioned entry point.
#   * A GPU build requires the matching toolkit (CUDA Toolkit, Vulkan SDK) and
#     real hardware to validate; the sandbox/CI is CPU-only and cannot run-test
#     a GPU decode. ggml's own find_package will fail later if the toolkit is
#     absent; the up-front guards below give a clear message first.
# ---------------------------------------------------------------------------

# Resolve the single effective backend from STRATAFLOW_GPU + the per-backend
# booleans (see precedence comment at the options block above).
string(TOLOWER "${STRATAFLOW_GPU}" STRATAFLOW_GPU_RESOLVED)
if(NOT STRATAFLOW_GPU_RESOLVED MATCHES "^(off|cuda|metal|vulkan)$")
    message(FATAL_ERROR
        "STRATAFLOW_GPU='${STRATAFLOW_GPU}' is not valid. "
        "Allowed values: off, cuda, metal, vulkan.")
endif()

if(STRATAFLOW_GPU_RESOLVED STREQUAL "off")
    # Fall back to the lower-level per-backend booleans (back-compat / presets).
    set(STRATAFLOW_GPU_COUNT 0)
    if(STRATAFLOW_CUDA)
        set(STRATAFLOW_GPU_RESOLVED "cuda")
        math(EXPR STRATAFLOW_GPU_COUNT "${STRATAFLOW_GPU_COUNT}+1")
    endif()
    if(STRATAFLOW_VULKAN)
        set(STRATAFLOW_GPU_RESOLVED "vulkan")
        math(EXPR STRATAFLOW_GPU_COUNT "${STRATAFLOW_GPU_COUNT}+1")
    endif()
    if(STRATAFLOW_METAL)
        set(STRATAFLOW_GPU_RESOLVED "metal")
        math(EXPR STRATAFLOW_GPU_COUNT "${STRATAFLOW_GPU_COUNT}+1")
    endif()
    if(STRATAFLOW_HIP)
        # HIP is a CUDA-family alias for the ggml wiring below.
        set(STRATAFLOW_GPU_RESOLVED "cuda")
        math(EXPR STRATAFLOW_GPU_COUNT "${STRATAFLOW_GPU_COUNT}+1")
    endif()
    if(STRATAFLOW_GPU_COUNT GREATER 1)
        message(FATAL_ERROR
            "More than one of STRATAFLOW_CUDA/VULKAN/METAL/HIP is ON. "
            "Select exactly one GPU backend (prefer -DSTRATAFLOW_GPU=cuda|metal|vulkan).")
    endif()
endif()

# Platform / toolkit guards. The OFF default never emits a fatal error.
if(STRATAFLOW_GPU_RESOLVED STREQUAL "metal")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        message(FATAL_ERROR
            "STRATAFLOW_GPU=metal requires macOS (Apple platform). "
            "Host system is '${CMAKE_SYSTEM_NAME}'. "
            "Use STRATAFLOW_GPU=cuda or vulkan on Linux/Windows, or off for CPU-only.")
    endif()
elseif(STRATAFLOW_GPU_RESOLVED STREQUAL "cuda")
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        message(FATAL_ERROR
            "STRATAFLOW_GPU=cuda is not available on macOS. "
            "Use STRATAFLOW_GPU=metal on Apple platforms, or off for CPU-only.")
    endif()
    message(STATUS
        "StrataFlow: CUDA backend selected; requires the CUDA Toolkit (nvcc) and an NVIDIA GPU to run.")
elseif(STRATAFLOW_GPU_RESOLVED STREQUAL "vulkan")
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        message(FATAL_ERROR
            "STRATAFLOW_GPU=vulkan is not supported on macOS here. "
            "Use STRATAFLOW_GPU=metal on Apple platforms, or off for CPU-only.")
    endif()
    message(STATUS
        "StrataFlow: Vulkan backend selected; requires the Vulkan SDK (glslc) and a Vulkan-capable GPU to run.")
endif()

# Toggle the vendored ggml backend options. Default (off): all forced OFF, which
# is byte-identical to the historical CPU-only build. On a selected backend the
# matching GGML_<...> is turned ON and the others stay OFF.
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_HIP    OFF CACHE BOOL "" FORCE)
if(STRATAFLOW_GPU_RESOLVED STREQUAL "cuda")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
elseif(STRATAFLOW_GPU_RESOLVED STREQUAL "metal")
    set(GGML_METAL ON CACHE BOOL "" FORCE)
elseif(STRATAFLOW_GPU_RESOLVED STREQUAL "vulkan")
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

if(STRATAFLOW_GPU_RESOLVED STREQUAL "off")
    message(STATUS "StrataFlow compute backend: off (CPU-only; GGML GPU backends forced OFF)")
else()
    string(TOUPPER "${STRATAFLOW_GPU_RESOLVED}" STRATAFLOW_GPU_UPPER)
    message(STATUS "StrataFlow compute backend: ${STRATAFLOW_GPU_RESOLVED} (GGML_${STRATAFLOW_GPU_UPPER} wiring enabled)")
endif()

add_subdirectory(${STRATAFLOW_LLAMA_DIR} EXCLUDE_FROM_ALL)

add_subdirectory(src)

if(STRATAFLOW_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(STRATAFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "StrataFlow ${PROJECT_VERSION} configured (${CMAKE_BUILD_TYPE})")

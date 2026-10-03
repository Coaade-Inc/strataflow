cmake_minimum_required(VERSION 3.20)

project(StrataFlow
    VERSION 0.0.1
    DESCRIPTION "Tiered VRAM/RAM/SSD streaming inference engine for large MoE models"
    LANGUAGES CXX C)

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

# Backend options (placeholders for Phase 1; wired to ggml in Phase 1b).
option(STRATAFLOW_CUDA   "Enable CUDA backend (via ggml)"   OFF)
option(STRATAFLOW_VULKAN "Enable Vulkan backend (via ggml)" OFF)
option(STRATAFLOW_METAL  "Enable Metal backend (via ggml)"  OFF)
option(STRATAFLOW_HIP    "Enable HIP/ROCm backend (via ggml)" OFF)

# ---------------------------------------------------------------------------
# Warning flags (portable across GCC/Clang/MSVC)
# ---------------------------------------------------------------------------
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

add_subdirectory(src)

if(STRATAFLOW_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(STRATAFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "StrataFlow ${PROJECT_VERSION} configured (${CMAKE_BUILD_TYPE})")

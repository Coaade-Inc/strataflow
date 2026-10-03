# Contributing to StrataFlow

A Coaade Inc. project. Source-available under the Coaade Source-Available
License, Version 1.0 (see [`LICENSE`](LICENSE)) — free for personal,
non-commercial use. By contributing, you agree your contributions are licensed
to Coaade Inc. under the same terms. Commercial licensing: contact@coaade.com.

## Build

Requires CMake ≥ 3.20, a C++20 compiler (GCC 11+, Clang 14+, or MSVC 2022), and
Ninja.

```sh
cmake --preset debug        # Debug build, warnings-as-errors
cmake --build build/debug
ctest --preset debug        # run the tests
```

Release build: use `--preset release`. Backend builds (once ggml is vendored):
`--preset release-cuda`, `release-vulkan`, `release-metal`.

## Layout

See [`docs/PLAN.md §5`](docs/PLAN.md). Each `src/<module>/` directory is one
component from the plan. New engine code goes in a module; the public surface is
`include/strataflow/strataflow.h` (stable C ABI).

## Rules

- **Warnings are errors** in the debug preset and in CI. Keep the build clean on
  GCC, Clang and MSVC — code must compile on Linux, macOS and Windows.
- **Determinism**: greedy generation must be reproducible. The `test_api`
  determinism check guards this; don't break it.
- **Add a test** for new engine logic. The harness is header-only
  (`tests/test_util.h`) — no external deps.
- **Format** with the repo `.clang-format` before committing.
- Keep platform-specific code behind the `STRATAFLOW_PLATFORM_*` macros and
  provide a portable fallback.

## Branches and PRs

- Branch from `main`; open a PR into `main`.
- CI must pass on all four matrix jobs (linux-gcc, linux-clang, macOS, Windows).
- CI/workflow changes always go through a PR, never a direct push to `main`.

// Tiny test harness: no external dependencies.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace sftest {
inline int &failures() { static int f = 0; return f; }
}

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond,          \
                         __FILE__, __LINE__);                                  \
            ++sftest::failures();                                              \
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        auto _va = (a);                                                        \
        auto _vb = (b);                                                        \
        if (!(_va == _vb)) {                                                   \
            std::fprintf(stderr, "CHECK_EQ failed: %s == %s (%s:%d)\n", #a, #b,\
                         __FILE__, __LINE__);                                  \
            ++sftest::failures();                                              \
        }                                                                      \
    } while (0)

#define RUN(fn)                                                                \
    do {                                                                       \
        std::printf("  %-28s", #fn);                                           \
        int before = sftest::failures();                                       \
        fn();                                                                  \
        std::printf("%s\n", before == sftest::failures() ? "ok" : "FAIL");     \
    } while (0)

#define TEST_MAIN()                                                            \
    int main() {                                                               \
        run_all();                                                             \
        if (sftest::failures() == 0) {                                         \
            std::printf("all passed\n");                                       \
            return 0;                                                          \
        }                                                                      \
        std::printf("%d failure(s)\n", sftest::failures());                    \
        return 1;                                                              \
    }

// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#include "common/log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

namespace sf {
namespace {

std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};
std::once_flag   g_init;
std::mutex       g_mutex;

LogLevel level_from_env() {
    const char *env = std::getenv("STRATAFLOW_LOG");
    if (env == nullptr) {
        return LogLevel::Info;
    }
    if (std::strcmp(env, "debug") == 0) return LogLevel::Debug;
    if (std::strcmp(env, "info") == 0)  return LogLevel::Info;
    if (std::strcmp(env, "warn") == 0)  return LogLevel::Warn;
    if (std::strcmp(env, "error") == 0) return LogLevel::Error;
    return LogLevel::Info;
}

const char *tag(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "[debug]";
        case LogLevel::Info:  return "[info ]";
        case LogLevel::Warn:  return "[warn ]";
        case LogLevel::Error: return "[error]";
    }
    return "[?????]";
}

} // namespace

void set_log_level(LogLevel level) {
    g_level.store(static_cast<int>(level));
}

void log(LogLevel level, const std::string &msg) {
    std::call_once(g_init, [] { g_level.store(static_cast<int>(level_from_env())); });
    if (static_cast<int>(level) < g_level.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    std::ostream &out = (level >= LogLevel::Warn) ? std::cerr : std::clog;
    out << "strataflow " << tag(level) << ' ' << msg << '\n';
}

} // namespace sf

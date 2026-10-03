// Minimal logging for the engine internals.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <string>

namespace sf {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

// Set the minimum level that is printed (default Info). Reads the
// STRATAFLOW_LOG env var ("debug"/"info"/"warn"/"error") on first use.
void set_log_level(LogLevel level);

void log(LogLevel level, const std::string &msg);

inline void log_debug(const std::string &m) { log(LogLevel::Debug, m); }
inline void log_info(const std::string &m)  { log(LogLevel::Info,  m); }
inline void log_warn(const std::string &m)  { log(LogLevel::Warn,  m); }
inline void log_error(const std::string &m) { log(LogLevel::Error, m); }

} // namespace sf

// Async block I/O for streaming expert weights off disk.
// See docs/PLAN.md section 3.3. Phase 1 ships a portable synchronous fallback;
// io_uring (Linux), IOCP (Windows) and F_NOCACHE+pread (macOS) land in Phase 3.
// Copyright 2026 Coaade Inc. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>

namespace sf {

// A handle to a file opened for weight streaming. Reads are 4 KiB-aligned so
// the same code path works once O_DIRECT is enabled.
class BlockFile {
public:
    BlockFile() = default;
    ~BlockFile();
    BlockFile(const BlockFile &) = delete;
    BlockFile &operator=(const BlockFile &) = delete;

    // Opens `path` for reading. Returns false on failure.
    bool open(const std::string &path);
    void close();

    bool is_open() const { return fd_ >= 0 || handle_ != nullptr; }
    uint64_t size() const { return size_; }

    // Synchronous read of `len` bytes at `offset` into `dst`.
    // Returns bytes read, or -1 on error.
    int64_t read_at(void *dst, size_t len, uint64_t offset) const;

    // Asynchronous read; Phase 1 implements this on a thread pool, Phase 3
    // swaps in the native async backend behind the same signature.
    std::future<int64_t> read_async(void *dst, size_t len, uint64_t offset) const;

    // Reports which async backend is active ("sync", "io_uring", ...).
    static const char *backend_name();

private:
    int   fd_ = -1;             // POSIX
    void *handle_ = nullptr;    // reserved for Win32 HANDLE
    uint64_t size_ = 0;
};

// Recommended read alignment for direct I/O.
constexpr size_t kIoAlignment = 4096;

inline uint64_t align_down(uint64_t x, uint64_t a) { return x & ~(a - 1); }
inline uint64_t align_up(uint64_t x, uint64_t a)   { return (x + a - 1) & ~(a - 1); }

} // namespace sf

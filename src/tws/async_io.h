// Async block I/O for streaming expert weights off disk.
// See docs/PLAN.md section 3.3 and docs/PHASE3_PLAN.md Task 4. The native
// backends bypass the OS page cache so streamed experts don't evict the
// resident trunk: O_DIRECT (+io_uring when built with liburing) on Linux,
// FILE_FLAG_NO_BUFFERING (+IOCP) on Windows, and F_NOCACHE on macOS. Each
// native path degrades cleanly to a portable synchronous fallback.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#if defined(STRATAFLOW_PLATFORM_windows)
#include <mutex>  // guards the stateful win_crt_ seek+read fallback in read_at
#endif

namespace sf {

// A handle to a file opened for weight streaming. Reads are 4 KiB-aligned so
// the same code path works once O_DIRECT is enabled.
class BlockFile {
public:
    // The I/O path actually selected at open() time. The native backend is an
    // optimization that degrades to kSync when the kernel/filesystem rejects
    // the direct-I/O flags (e.g. O_DIRECT with EINVAL inside a container).
    enum class Backend {
        kSync,       // portable buffered pread / CRT fread fallback
        kDirect,     // O_DIRECT + pread (Linux) / F_NOCACHE + pread (macOS)
        kIoUring,    // io_uring + O_DIRECT (Linux, when built with liburing)
        kNoBuffering // FILE_FLAG_NO_BUFFERING (+ IOCP) on Windows
    };

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
    // Returns bytes read, or -1 on error. For direct-I/O backends unaligned
    // requests are served through an aligned bounce buffer transparently.
    int64_t read_at(void *dst, size_t len, uint64_t offset) const;

    // Asynchronous read; wraps read_at on a thread so callers (the residency
    // pass, Phase 4 prefetch) get a future that completes when the op does.
    std::future<int64_t> read_async(void *dst, size_t len, uint64_t offset) const;

    // The backend selected for this open file (valid while is_open()).
    Backend backend() const { return backend_; }

    // Human-readable name of the backend active on this handle. Falls back to
    // the compiled platform default when no file is open.
    const char *active_backend_name() const;

    // Reports the platform's native async backend ("io_uring+O_DIRECT",
    // "O_DIRECT+pread", "IOCP+no_buffering", "F_NOCACHE+pread", or the
    // "sync (fallback)" when no native path is compiled in). Static so logs
    // can report capability without an open handle.
    static const char *backend_name();

private:
    // Direct-I/O read honoring the required alignment via an aligned bounce
    // buffer. Returns bytes copied into `dst`, or -1 on error.
    int64_t read_direct_aligned(void *dst, size_t len, uint64_t offset) const;

    int   fd_ = -1;             // POSIX file descriptor
    void *handle_ = nullptr;    // Win32 HANDLE (or CRT FILE* fallback)
    uint64_t size_ = 0;
    size_t   alignment_ = 1;    // required I/O alignment for the active backend
    Backend  backend_ = Backend::kSync;
#if defined(STRATAFLOW_PLATFORM_windows)
    bool     win_crt_ = false;  // Windows: handle_ is a FILE* (sync fallback)
    // Guards ONLY the win_crt_ sync-fallback branch of read_at, whose
    // _fseeki64 + fread advance a SHARED, stateful file position on handle_.
    // Once the async-prefetch worker and the compute thread both call read_blob
    // (-> read_at) concurrently, two interleaved seek+read pairs on the same
    // handle would race and return bytes from the wrong offset. A per-call lock
    // around just the seek+read makes each positional read atomic. The POSIX
    // pread path is positional/stateless and intentionally stays lock-free, so
    // this member only exists on Windows. mutable because read_at is const.
    mutable std::mutex win_crt_mu_;
#endif
};

// Recommended read alignment for direct I/O.
constexpr size_t kIoAlignment = 4096;

inline uint64_t align_down(uint64_t x, uint64_t a) { return x & ~(a - 1); }
inline uint64_t align_up(uint64_t x, uint64_t a)   { return (x + a - 1) & ~(a - 1); }

} // namespace sf

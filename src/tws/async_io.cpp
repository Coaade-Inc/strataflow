// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "tws/async_io.h"

#include "common/log.h"

#include <cerrno>
#include <cstring>

#if defined(STRATAFLOW_PLATFORM_windows)
#  include <io.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace sf {

BlockFile::~BlockFile() { close(); }

bool BlockFile::open(const std::string &path) {
    close();
#if defined(STRATAFLOW_PLATFORM_windows)
    // Phase 1: portable CRT path; a native IOCP+FILE_FLAG_NO_BUFFERING path
    // replaces this in Phase 3.
    FILE *f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") != 0 || f == nullptr) {
        log_error("BlockFile: cannot open " + path);
        return false;
    }
    _fseeki64(f, 0, SEEK_END);
    size_ = static_cast<uint64_t>(_ftelli64(f));
    _fseeki64(f, 0, SEEK_SET);
    handle_ = f;
    return true;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        log_error("BlockFile: cannot open " + path + ": " + std::strerror(errno));
        return false;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return false;
    }
    fd_ = fd;
    size_ = static_cast<uint64_t>(st.st_size);
    return true;
#endif
}

void BlockFile::close() {
#if defined(STRATAFLOW_PLATFORM_windows)
    if (handle_ != nullptr) {
        std::fclose(static_cast<FILE *>(handle_));
        handle_ = nullptr;
    }
#else
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    size_ = 0;
}

int64_t BlockFile::read_at(void *dst, size_t len, uint64_t offset) const {
#if defined(STRATAFLOW_PLATFORM_windows)
    if (handle_ == nullptr) return -1;
    FILE *f = static_cast<FILE *>(handle_);
    if (_fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0) return -1;
    return static_cast<int64_t>(std::fread(dst, 1, len, f));
#else
    if (fd_ < 0) return -1;
    ssize_t n = ::pread(fd_, dst, len, static_cast<off_t>(offset));
    return static_cast<int64_t>(n);
#endif
}

std::future<int64_t> BlockFile::read_async(void *dst, size_t len,
                                           uint64_t offset) const {
    // Phase 1: deferred async via std::async. The signature is the one Phase 3
    // keeps, so callers (the scheduler's prefetch path) don't change.
    return std::async(std::launch::deferred,
                      [this, dst, len, offset] { return read_at(dst, len, offset); });
}

const char *BlockFile::backend_name() {
#if defined(STRATAFLOW_PLATFORM_linux)
    return "sync-pread (io_uring pending)";
#elif defined(STRATAFLOW_PLATFORM_windows)
    return "sync-crt (IOCP pending)";
#elif defined(STRATAFLOW_PLATFORM_macos)
    return "sync-pread (F_NOCACHE pending)";
#else
    return "sync";
#endif
}

} // namespace sf

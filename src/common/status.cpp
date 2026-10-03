// Status-code helpers and the version string.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#include "strataflow/strataflow.h"

extern "C" {

const char *sf_version(void) {
    return "0.0.1";
}

const char *sf_status_str(sf_status status) {
    switch (status) {
        case SF_OK:                   return "ok";
        case SF_ERR_INVALID_ARGUMENT: return "invalid argument";
        case SF_ERR_FILE_NOT_FOUND:   return "file not found";
        case SF_ERR_UNSUPPORTED:      return "unsupported (not implemented yet)";
        case SF_ERR_OUT_OF_MEMORY:    return "out of memory";
        case SF_ERR_IO:               return "I/O error";
        case SF_ERR_MODEL_LOAD:       return "model load error";
        case SF_ERR_RUNTIME:          return "runtime error";
    }
    return "unknown status";
}

} // extern "C"

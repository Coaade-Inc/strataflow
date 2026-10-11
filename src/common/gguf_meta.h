// Robust GGUF integer metadata reads.
//
// gguf_get_val_u32 ABORTS (ggml assert) when the key is stored under a
// different integer type than UINT32. Real-world GGUFs encode the same
// hyperparameter key (e.g. llama.context_length, llama.attention.head_count,
// llama.attention.head_count_kv) under VARYING integer widths/signedness
// depending on the converter that wrote the file (u32 vs i32 vs other int
// widths). gguf_read_u32() dispatches on the stored type so the planner and
// the engine read IDENTICAL geometry no matter how the source encoded it,
// instead of silently seeing 0 (the real-Mixtral "n_ctx_train(0)" bug).
//
// Only this one TU-shared helper may be included by any module that already
// pulls in <gguf.h>; it adds no new dependency.
// Copyright 2026 Coaade Inc., a Delaware C corporation. SPDX-License-Identifier: LicenseRef-Coaade-Source-Available-1.0
#pragma once

#include <gguf.h>

#include <cstdint>

namespace sf {

// Read the integer metadata value at key id `id` as a uint32_t, tolerating
// every GGUF integer encoding (u8/i8/u16/i16/u32/i32/u64/i64). A float/string/
// bool/array key, or a negative/clamped integer, yields `fallback` so a
// surprise type degrades to the caller's safe default rather than aborting.
inline uint32_t gguf_read_u32(const gguf_context *gc, int64_t id,
                              uint32_t fallback = 0) {
    if (gc == nullptr || id < 0) return fallback;
    switch (gguf_get_kv_type(gc, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(gc, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gc, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gc, id);
        case GGUF_TYPE_INT8: {
            const int8_t v = gguf_get_val_i8(gc, id);
            return v >= 0 ? static_cast<uint32_t>(v) : fallback;
        }
        case GGUF_TYPE_INT16: {
            const int16_t v = gguf_get_val_i16(gc, id);
            return v >= 0 ? static_cast<uint32_t>(v) : fallback;
        }
        case GGUF_TYPE_INT32: {
            const int32_t v = gguf_get_val_i32(gc, id);
            return v >= 0 ? static_cast<uint32_t>(v) : fallback;
        }
        case GGUF_TYPE_UINT64: {
            const uint64_t v = gguf_get_val_u64(gc, id);
            return v <= UINT32_MAX ? static_cast<uint32_t>(v) : fallback;
        }
        case GGUF_TYPE_INT64: {
            const int64_t v = gguf_get_val_i64(gc, id);
            return (v >= 0 && v <= static_cast<int64_t>(UINT32_MAX))
                       ? static_cast<uint32_t>(v)
                       : fallback;
        }
        default:
            return fallback;  // float/bool/string/array - not an integer key
    }
}

// Convenience: find `key` and read it as a uint32_t via gguf_read_u32().
inline uint32_t gguf_read_u32_key(const gguf_context *gc, const char *key,
                                  uint32_t fallback = 0) {
    if (gc == nullptr || key == nullptr) return fallback;
    return gguf_read_u32(gc, gguf_find_key(gc, key), fallback);
}

} // namespace sf

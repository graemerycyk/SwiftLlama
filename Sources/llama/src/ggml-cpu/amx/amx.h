// Stub header for AMX (Intel Advanced Matrix Extensions)
// AMX is only available on Intel CPUs, not Apple Silicon
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// Stub function - returns nullptr on non-Intel platforms
static inline ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(void) {
    return nullptr;
}

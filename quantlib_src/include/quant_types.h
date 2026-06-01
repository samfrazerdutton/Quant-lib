#pragma once
#include <stdint.h>
#include <cuda_fp16.h>

// ── FP8 formats (two variants used in practice) ───────────────────────────────
// E4M3: 4 exponent bits, 3 mantissa bits. Range ~[-448, 448]. Best for weights.
// E5M2: 5 exponent bits, 2 mantissa bits. Range ~[-57344, 57344]. Best for grads.
typedef uint8_t fp8_e4m3_t;
typedef uint8_t fp8_e5m2_t;

// ── FP4 (packed: two 4-bit values per byte) ───────────────────────────────────
// Layout: high nibble = element[0], low nibble = element[1]
// E2M1: 2 exponent bits, 1 mantissa bit. Range [-6, 6].
typedef uint8_t fp4_packed_t;  // holds 2x FP4 values

// ── Quantization scales (per-tensor, per-channel, per-group) ─────────────────
typedef enum {
    QUANT_PER_TENSOR  = 0,
    QUANT_PER_CHANNEL = 1,
    QUANT_PER_GROUP   = 2   // AWQ / GPTQ use group_size=128 typically
} QuantGranularity;

typedef struct {
    float*           scales;       // GPU pointer to scale factors
    float*           zeros;        // GPU pointer to zero points (GPTQ needs these)
    int              group_size;   // 0 = per-tensor/channel, 128 typical for AWQ
    QuantGranularity granularity;
    int              n_scales;     // total number of scale values
} QuantParams;

// ── Quantized weight matrix ───────────────────────────────────────────────────
typedef struct {
    void*       data;        // GPU pointer (fp8_e4m3_t* or fp4_packed_t*)
    QuantParams params;
    int         rows;
    int         cols;
    int         bits;        // 8 or 4
    char        format[16];  // "E4M3", "E5M2", "E2M1"
} QuantMatrix;

// ── Bit manipulation helpers (host + device) ─────────────────────────────────
// FP32 bit layout: [31]=sign [30:23]=exp(bias 127) [22:0]=mantissa
// FP8 E4M3 layout: [7]=sign  [6:3]=exp(bias 7)    [2:0]=mantissa
// FP8 E5M2 layout: [7]=sign  [6:2]=exp(bias 15)   [1:0]=mantissa
// FP4 E2M1 layout: [3]=sign  [2:1]=exp(bias 1)    [0]=mantissa

__host__ __device__ inline fp8_e4m3_t float_to_fp8_e4m3(float v) {
    // Clamp to E4M3 range [-448, 448]
    v = v > 448.0f ? 448.0f : (v < -448.0f ? -448.0f : v);
    uint32_t bits;
    __builtin_memcpy(&bits, &v, 4);
    uint8_t sign = (bits >> 31) & 1;
    int     exp  = ((bits >> 23) & 0xFF) - 127 + 7;   // rebias to 7
    uint8_t mant = (bits >> 20) & 0x7;                 // top 3 mantissa bits
    if (exp <= 0) { return sign << 7; }                // underflow -> zero
    if (exp > 15) { exp = 15; mant = 7; }              // overflow -> max
    return (sign << 7) | ((uint8_t)exp << 3) | mant;
}

__host__ __device__ inline float fp8_e4m3_to_float(fp8_e4m3_t v) {
    uint8_t sign = (v >> 7) & 1;
    uint8_t exp  = (v >> 3) & 0xF;
    uint8_t mant = v & 0x7;
    if (exp == 0 && mant == 0) return 0.0f;
    float result = (1.0f + mant / 8.0f) * powf(2.0f, (float)exp - 7.0f);
    return sign ? -result : result;
}

__host__ __device__ inline fp8_e5m2_t float_to_fp8_e5m2(float v) {
    v = v > 57344.0f ? 57344.0f : (v < -57344.0f ? -57344.0f : v);
    uint32_t bits;
    __builtin_memcpy(&bits, &v, 4);
    uint8_t sign = (bits >> 31) & 1;
    int     exp  = ((bits >> 23) & 0xFF) - 127 + 15;
    uint8_t mant = (bits >> 21) & 0x3;
    if (exp <= 0) { return sign << 7; }
    if (exp > 31) { exp = 31; mant = 3; }
    return (sign << 7) | ((uint8_t)exp << 2) | mant;
}

__host__ __device__ inline float fp8_e5m2_to_float(fp8_e5m2_t v) {
    uint8_t sign = (v >> 7) & 1;
    uint8_t exp  = (v >> 2) & 0x1F;
    uint8_t mant = v & 0x3;
    if (exp == 0 && mant == 0) return 0.0f;
    float result = (1.0f + mant / 4.0f) * powf(2.0f, (float)exp - 15.0f);
    return sign ? -result : result;
}

// FP4 E2M1: values representable = {0, 0.5, 1, 1.5, 2, 3, 4, 6} +/- 
__host__ __device__ inline uint8_t float_to_fp4_e2m1(float v) {
    static const float lut[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
    uint8_t sign = v < 0 ? 1 : 0;
    float   av   = sign ? -v : v;
    // Find nearest in LUT
    uint8_t best = 0;
    float   bestd = 1e30f;
    for (int i = 0; i < 8; i++) {
        float d = av - lut[i]; if (d < 0) d = -d;
        if (d < bestd) { bestd = d; best = i; }
    }
    return (sign << 3) | (best & 0x7);
}

__host__ __device__ inline float fp4_e2m1_to_float(uint8_t v) {
    static const float lut[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
    uint8_t sign = (v >> 3) & 1;
    float   mag  = lut[v & 0x7];
    return sign ? -mag : mag;
}

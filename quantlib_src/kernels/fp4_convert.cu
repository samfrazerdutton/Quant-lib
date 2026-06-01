#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "../include/quant_types.h"

// ── Per-group scale computation for FP4 ──────────────────────────────────────
// FP4 E2M1 max representable value = 6.0
#define FP4_MAX 6.0f

__global__ void compute_scales_fp4(
    const __half* __restrict__ src,
    float* __restrict__ scales,
    int n_elements, int group_size)
{
    int group  = blockIdx.x;
    int offset = group * group_size;
    int tid    = threadIdx.x;

    float local_max = 0.0f;
    for (int i = tid; i < group_size && offset + i < n_elements; i += blockDim.x)
        local_max = fmaxf(local_max, fabsf(__half2float(src[offset + i])));

    for (int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, mask));

    if (tid == 0)
        scales[group] = (local_max < 1e-8f) ? 1.0f : (FP4_MAX / local_max);
}

// ── FP16 → FP4 packed quantization kernel ────────────────────────────────────
// Processes 2 elements per thread, packs them into 1 byte.
// high nibble = element[2*idx], low nibble = element[2*idx+1]
__global__ void quantize_fp4_kernel(
    const __half*    __restrict__ src,
    fp4_packed_t*    __restrict__ dst,
    const float*     __restrict__ scales,
    int n_elements, int group_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx * 2;
    if (pair_idx >= n_elements) return;

    int   group  = pair_idx / group_size;
    float scale  = scales[group];

    float v0 = __half2float(src[pair_idx])     * scale;
    float v1 = (pair_idx + 1 < n_elements)
               ? __half2float(src[pair_idx+1]) * scale
               : 0.0f;

    uint8_t q0 = float_to_fp4_e2m1(v0);
    uint8_t q1 = float_to_fp4_e2m1(v1);

    // Pack: high nibble = q0, low nibble = q1
    dst[idx] = (q0 << 4) | (q1 & 0x0F);
}

// ── FP4 packed → FP16 dequantization kernel ───────────────────────────────────
__global__ void dequantize_fp4_kernel(
    const fp4_packed_t* __restrict__ src,
    __half*             __restrict__ dst,
    const float*        __restrict__ scales,
    int n_elements, int group_size)
{
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx * 2;
    if (pair_idx >= n_elements) return;

    int   group  = pair_idx / group_size;
    float scale  = scales[group];

    uint8_t packed = src[idx];
    uint8_t q0     = (packed >> 4) & 0x0F;
    uint8_t q1     =  packed       & 0x0F;

    dst[pair_idx]   = __float2half(fp4_e2m1_to_float(q0) / scale);
    if (pair_idx + 1 < n_elements)
        dst[pair_idx+1] = __float2half(fp4_e2m1_to_float(q1) / scale);
}

// ── Host launch functions ─────────────────────────────────────────────────────
QuantMatrix launch_fp16_to_fp4(const __half* src, int rows, int cols,
                                QuantGranularity gran, int group_size,
                                cudaStream_t stream)
{
    int n = rows * cols;
    if (gran == QUANT_PER_TENSOR)  group_size = n;
    if (gran == QUANT_PER_CHANNEL) group_size = cols;

    // group_size must be even for packing
    if (group_size % 2 != 0) group_size++;

    QuantMatrix qm;
    qm.rows = rows; qm.cols = cols; qm.bits = 4;
    snprintf(qm.format, 16, "E2M1");

    int packed_size = (n + 1) / 2;
    cudaMalloc(&qm.data, packed_size * sizeof(fp4_packed_t));

    qm.params.granularity = gran;
    qm.params.group_size  = group_size;
    qm.params.zeros       = nullptr;
    if (gran == QUANT_PER_TENSOR)       qm.params.n_scales = 1;
    else if (gran == QUANT_PER_CHANNEL) qm.params.n_scales = rows;
    else qm.params.n_scales = (n + group_size - 1) / group_size;
    cudaMalloc(&qm.params.scales, qm.params.n_scales * sizeof(float));

    compute_scales_fp4<<<qm.params.n_scales, 128, 0, stream>>>(
        src, qm.params.scales, n, group_size);

    int threads = 256;
    int blocks  = (packed_size + threads - 1) / threads;
    quantize_fp4_kernel<<<blocks, threads, 0, stream>>>(
        src, (fp4_packed_t*)qm.data, qm.params.scales, n, group_size);

    return qm;
}

void launch_fp4_to_fp16(const QuantMatrix* qm, __half* dst, cudaStream_t stream)
{
    int n          = qm->rows * qm->cols;
    int packed_size = (n + 1) / 2;
    int gs         = qm->params.group_size;
    int threads    = 256;
    int blocks     = (packed_size + threads - 1) / threads;

    dequantize_fp4_kernel<<<blocks, threads, 0, stream>>>(
        (fp4_packed_t*)qm->data, dst, qm->params.scales, n, gs);
}

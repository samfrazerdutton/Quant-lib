#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/quant_types.h"
#include <stdio.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d -- %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } } while(0)

// ── Per-group absolute max (scale computation) ────────────────────────────────
// Each block handles one group. Warp-reduce to find max |x|.
__global__ void compute_scales_fp8(
    const __half* __restrict__ src,
    float* __restrict__ scales,
    int n_elements, int group_size, float fp8_max)
{
    int group  = blockIdx.x;
    int offset = group * group_size;
    int tid    = threadIdx.x;

    float local_max = 0.0f;
    for (int i = tid; i < group_size && offset + i < n_elements; i += blockDim.x)
        local_max = fmaxf(local_max, fabsf(__half2float(src[offset + i])));

    // Warp reduce
    for (int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, mask));

    if (tid == 0)
        scales[group] = (local_max < 1e-8f) ? 1.0f : (fp8_max / local_max);
}

// ── FP16 → FP8 E4M3 quantization kernel ──────────────────────────────────────
__global__ void quantize_fp8_e4m3_kernel(
    const __half* __restrict__ src,
    fp8_e4m3_t*  __restrict__ dst,
    const float* __restrict__ scales,
    int n_elements, int group_size)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    int   group = idx / group_size;
    float scale = scales[group];
    float val   = __half2float(src[idx]) * scale;
    dst[idx]    = float_to_fp8_e4m3(val);
}

// ── FP16 → FP8 E5M2 quantization kernel ──────────────────────────────────────
__global__ void quantize_fp8_e5m2_kernel(
    const __half* __restrict__ src,
    fp8_e5m2_t*  __restrict__ dst,
    const float* __restrict__ scales,
    int n_elements, int group_size)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    int   group = idx / group_size;
    float scale = scales[group];
    float val   = __half2float(src[idx]) * scale;
    dst[idx]    = float_to_fp8_e5m2(val);
}

// ── FP8 E4M3 → FP16 dequantization kernel ────────────────────────────────────
__global__ void dequantize_fp8_e4m3_kernel(
    const fp8_e4m3_t* __restrict__ src,
    __half*           __restrict__ dst,
    const float*      __restrict__ scales,
    int n_elements, int group_size)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    int   group = idx / group_size;
    float scale = scales[group];
    float val   = fp8_e4m3_to_float(src[idx]) / scale;
    dst[idx]    = __float2half(val);
}

__global__ void dequantize_fp8_e5m2_kernel(
    const fp8_e5m2_t* __restrict__ src,
    __half*           __restrict__ dst,
    const float*      __restrict__ scales,
    int n_elements, int group_size)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    int   group = idx / group_size;
    float scale = scales[group];
    float val   = fp8_e5m2_to_float(src[idx]) / scale;
    dst[idx]    = __float2half(val);
}

// ── Host launch functions ─────────────────────────────────────────────────────
static void alloc_scales(QuantParams* p, int n_elements, int group_size,
                         QuantGranularity gran, int rows)
{
    p->granularity = gran;
    p->group_size  = group_size;
    if (gran == QUANT_PER_TENSOR)  p->n_scales = 1;
    else if (gran == QUANT_PER_CHANNEL) p->n_scales = rows;
    else p->n_scales = (n_elements + group_size - 1) / group_size;
    cudaMalloc(&p->scales, p->n_scales * sizeof(float));
    p->zeros = nullptr;
}

QuantMatrix launch_fp16_to_fp8_e4m3(const __half* src, int rows, int cols,
                                     QuantGranularity gran, int group_size,
                                     cudaStream_t stream)
{
    int n = rows * cols;
    if (gran == QUANT_PER_TENSOR)  group_size = n;
    if (gran == QUANT_PER_CHANNEL) group_size = cols;

    QuantMatrix qm;
    qm.rows = rows; qm.cols = cols; qm.bits = 8;
    snprintf(qm.format, 16, "E4M3");
    cudaMalloc(&qm.data, n * sizeof(fp8_e4m3_t));
    alloc_scales(&qm.params, n, group_size, gran, rows);

    int n_groups = qm.params.n_scales;
    compute_scales_fp8<<<n_groups, 128, 0, stream>>>(
        src, qm.params.scales, n, group_size, 448.0f);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    quantize_fp8_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        src, (fp8_e4m3_t*)qm.data, qm.params.scales, n, group_size);

    return qm;
}

QuantMatrix launch_fp16_to_fp8_e5m2(const __half* src, int rows, int cols,
                                     QuantGranularity gran, int group_size,
                                     cudaStream_t stream)
{
    int n = rows * cols;
    if (gran == QUANT_PER_TENSOR)  group_size = n;
    if (gran == QUANT_PER_CHANNEL) group_size = cols;

    QuantMatrix qm;
    qm.rows = rows; qm.cols = cols; qm.bits = 8;
    snprintf(qm.format, 16, "E5M2");
    cudaMalloc(&qm.data, n * sizeof(fp8_e5m2_t));
    alloc_scales(&qm.params, n, group_size, gran, rows);

    int n_groups = qm.params.n_scales;
    compute_scales_fp8<<<n_groups, 128, 0, stream>>>(
        src, qm.params.scales, n, group_size, 57344.0f);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    quantize_fp8_e5m2_kernel<<<blocks, threads, 0, stream>>>(
        src, (fp8_e5m2_t*)qm.data, qm.params.scales, n, group_size);

    return qm;
}

void launch_fp8_to_fp16(const QuantMatrix* qm, __half* dst, cudaStream_t stream)
{
    int n       = qm->rows * qm->cols;
    int gs      = qm->params.group_size;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    if (qm->format[1] == '4')
        dequantize_fp8_e4m3_kernel<<<blocks, threads, 0, stream>>>(
            (fp8_e4m3_t*)qm->data, dst, qm->params.scales, n, gs);
    else
        dequantize_fp8_e5m2_kernel<<<blocks, threads, 0, stream>>>(
            (fp8_e5m2_t*)qm->data, dst, qm->params.scales, n, gs);
}

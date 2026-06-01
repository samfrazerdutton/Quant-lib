#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/quantlib.h"

// ── Forward declarations from kernels ─────────────────────────────────────────
extern QuantMatrix launch_fp16_to_fp8_e4m3(const __half*, int, int,
                                            QuantGranularity, int, cudaStream_t);
extern QuantMatrix launch_fp16_to_fp8_e5m2(const __half*, int, int,
                                            QuantGranularity, int, cudaStream_t);
extern QuantMatrix launch_fp16_to_fp4     (const __half*, int, int,
                                            QuantGranularity, int, cudaStream_t);
extern void        launch_fp8_to_fp16     (const QuantMatrix*, __half*, cudaStream_t);
extern void        launch_fp4_to_fp16     (const QuantMatrix*, __half*, cudaStream_t);
extern void        launch_fp8_gemm        (const __half*, const QuantMatrix*,
                                            __half*, int, int, int, cudaStream_t);
extern void        launch_fp4_gemm        (const __half*, const QuantMatrix*,
                                            __half*, int, int, int, cudaStream_t);
extern QuantMatrix launch_awq_quantize    (const __half*, int, int,
                                            const __half*, int, int, cudaStream_t);
extern QuantMatrix launch_gptq_quantize   (const __half*, int, int,
                                            const __half*, int, int, cudaStream_t);

// ── Public API ────────────────────────────────────────────────────────────────
QuantMatrix quantlib_fp16_to_fp8_e4m3(const __half* src, int rows, int cols,
                                       QuantGranularity gran, int group_size,
                                       cudaStream_t stream)
{
    return launch_fp16_to_fp8_e4m3(src, rows, cols, gran, group_size, stream);
}

QuantMatrix quantlib_fp16_to_fp8_e5m2(const __half* src, int rows, int cols,
                                       QuantGranularity gran, int group_size,
                                       cudaStream_t stream)
{
    return launch_fp16_to_fp8_e5m2(src, rows, cols, gran, group_size, stream);
}

QuantMatrix quantlib_fp16_to_fp4(const __half* src, int rows, int cols,
                                  QuantGranularity gran, int group_size,
                                  cudaStream_t stream)
{
    return launch_fp16_to_fp4(src, rows, cols, gran, group_size, stream);
}

void quantlib_fp8_to_fp16(const QuantMatrix* qm, __half* dst, cudaStream_t stream)
{
    launch_fp8_to_fp16(qm, dst, stream);
}

void quantlib_fp4_to_fp16(const QuantMatrix* qm, __half* dst, cudaStream_t stream)
{
    launch_fp4_to_fp16(qm, dst, stream);
}

void quantlib_fp8_gemm(const __half* A, const QuantMatrix* W,
                        __half* C, int M, int N, int K,
                        cudaStream_t stream)
{
    launch_fp8_gemm(A, W, C, M, N, K, stream);
}

void quantlib_fp4_gemm(const __half* A, const QuantMatrix* W,
                        __half* C, int M, int N, int K,
                        cudaStream_t stream)
{
    launch_fp4_gemm(A, W, C, M, N, K, stream);
}

QuantMatrix quantlib_awq_quantize(const __half* weights, int rows, int cols,
                                   const __half* calibration_acts,
                                   int n_calib_tokens,
                                   int group_size, int bits,
                                   cudaStream_t stream)
{
    // bits param reserved for future FP4 AWQ; currently always FP8 E4M3
    (void)bits;
    return launch_awq_quantize(weights, rows, cols,
                                calibration_acts, n_calib_tokens,
                                group_size, stream);
}

QuantMatrix quantlib_gptq_quantize(const __half* weights, int rows, int cols,
                                    const __half* calibration_acts,
                                    int n_calib_tokens,
                                    int group_size, int bits,
                                    cudaStream_t stream)
{
    (void)bits;
    return launch_gptq_quantize(weights, rows, cols,
                                 calibration_acts, n_calib_tokens,
                                 group_size, stream);
}

void quantlib_free(QuantMatrix* qm)
{
    if (!qm) return;
    if (qm->data)           cudaFree(qm->data);
    if (qm->params.scales)  cudaFree(qm->params.scales);
    if (qm->params.zeros)   cudaFree(qm->params.zeros);
    qm->data          = nullptr;
    qm->params.scales = nullptr;
    qm->params.zeros  = nullptr;
}

float quantlib_quant_error(const __half* original, const __half* dequantized,
                            int n_elements)
{
    // Copy both to host and compute max absolute error
    __half* h_orig = (__half*)malloc(n_elements * sizeof(__half));
    __half* h_deq  = (__half*)malloc(n_elements * sizeof(__half));
    cudaMemcpy(h_orig, original,    n_elements * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_deq,  dequantized, n_elements * sizeof(__half), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < n_elements; i++) {
        float err = fabsf(__half2float(h_orig[i]) - __half2float(h_deq[i]));
        if (err > max_err) max_err = err;
    }
    free(h_orig);
    free(h_deq);
    return max_err;
}

void quantlib_print_stats(const QuantMatrix* qm)
{
    size_t raw_bytes  = (size_t)qm->rows * qm->cols * 2;  // fp16 baseline
    size_t quant_bytes;
    if (qm->bits == 8)
        quant_bytes = (size_t)qm->rows * qm->cols;
    else
        quant_bytes = ((size_t)qm->rows * qm->cols + 1) / 2;

    printf("QuantMatrix [%s]  %d x %d  bits=%d\n",
           qm->format, qm->rows, qm->cols, qm->bits);
    printf("  FP16 size    : %.2f MB\n", raw_bytes   / 1e6f);
    printf("  Quant size   : %.2f MB\n", quant_bytes / 1e6f);
    printf("  Compression  : %.2fx\n",   (float)raw_bytes / quant_bytes);
    printf("  Scales       : %d  (granularity=%d  group=%d)\n",
           qm->params.n_scales, qm->params.granularity, qm->params.group_size);
}

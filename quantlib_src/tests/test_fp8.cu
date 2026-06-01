#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/quantlib.h"

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d -- %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } } while(0)

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s\n\n", prop.name);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int rows = 1024, cols = 1024;
    int n    = rows * cols;

    // Generate random FP16 weights on host
    __half* h_weights = (__half*)malloc(n * sizeof(__half));
    for (int i = 0; i < n; i++)
        h_weights[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

    // Upload to GPU
    __half* d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, n * sizeof(__half),
                          cudaMemcpyHostToDevice));

    // ── Test 1: FP8 E4M3 per-group quantization ───────────────────────────────
    printf("=== FP8 E4M3 (per-group, group=128) ===\n");
    QuantMatrix qm_e4m3 = quantlib_fp16_to_fp8_e4m3(
        d_weights, rows, cols, QUANT_PER_GROUP, 128, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    quantlib_print_stats(&qm_e4m3);

    // Dequantize and measure error
    __half* d_deq;
    CUDA_CHECK(cudaMalloc(&d_deq, n * sizeof(__half)));
    quantlib_fp8_to_fp16(&qm_e4m3, d_deq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float err_e4m3 = quantlib_quant_error(d_weights, d_deq, n);
    printf("  Max abs error  : %.6e\n", err_e4m3);
    printf("  Correctness    : %s\n\n", err_e4m3 < 0.25f ? "[PASS]" : "[FAIL]");

    quantlib_free(&qm_e4m3);

    // ── Test 2: FP8 E5M2 per-channel quantization ─────────────────────────────
    printf("=== FP8 E5M2 (per-channel) ===\n");
    QuantMatrix qm_e5m2 = quantlib_fp16_to_fp8_e5m2(
        d_weights, rows, cols, QUANT_PER_CHANNEL, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    quantlib_print_stats(&qm_e5m2);

    quantlib_fp8_to_fp16(&qm_e5m2, d_deq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float err_e5m2 = quantlib_quant_error(d_weights, d_deq, n);
    printf("  Max abs error  : %.6e\n", err_e5m2);
    printf("  Correctness    : %s\n\n", err_e5m2 < 0.25f ? "[PASS]" : "[FAIL]");

    quantlib_free(&qm_e5m2);

    // ── Test 3: FP8 E4M3 per-tensor ───────────────────────────────────────────
    printf("=== FP8 E4M3 (per-tensor) ===\n");
    QuantMatrix qm_pt = quantlib_fp16_to_fp8_e4m3(
        d_weights, rows, cols, QUANT_PER_TENSOR, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    quantlib_print_stats(&qm_pt);

    quantlib_fp8_to_fp16(&qm_pt, d_deq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float err_pt = quantlib_quant_error(d_weights, d_deq, n);
    printf("  Max abs error  : %.6e\n", err_pt);
    printf("  Correctness    : %s\n\n", err_pt < 0.25f ? "[PASS]" : "[FAIL]");

    quantlib_free(&qm_pt);

    // ── Test 4: Benchmark quantization throughput ─────────────────────────────
    printf("=== Benchmark: FP16 → FP8 E4M3 (1024x1024, group=128) ===\n");
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    int reps = 100;
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < reps; i++) {
        QuantMatrix qm_tmp = quantlib_fp16_to_fp8_e4m3(
            d_weights, rows, cols, QUANT_PER_GROUP, 128, stream);
        quantlib_free(&qm_tmp);
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float avg_ms = ms / reps;
    float gb_s   = (n * sizeof(__half)) / (avg_ms * 1e-3f) / 1e9f;
    printf("  Avg time       : %.3f ms\n", avg_ms);
    printf("  Throughput     : %.1f GB/s\n\n", gb_s);

    cudaFree(d_weights);
    cudaFree(d_deq);
    free(h_weights);
    cudaStreamDestroy(stream);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    printf("=== All FP8 tests complete ===\n");
    return 0;
}

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

    // Random FP16 weights
    __half* h_weights = (__half*)malloc(n * sizeof(__half));
    for (int i = 0; i < n; i++)
        h_weights[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

    __half* d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, n * sizeof(__half),
                          cudaMemcpyHostToDevice));

    // ── Test 1: FP4 per-group quantization ────────────────────────────────────
    printf("=== FP4 E2M1 (per-group, group=128) ===\n");
    QuantMatrix qm = quantlib_fp16_to_fp4(
        d_weights, rows, cols, QUANT_PER_GROUP, 128, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    quantlib_print_stats(&qm);

    // Dequantize and check error
    __half* d_deq;
    CUDA_CHECK(cudaMalloc(&d_deq, n * sizeof(__half)));
    quantlib_fp4_to_fp16(&qm, d_deq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float err = quantlib_quant_error(d_weights, d_deq, n);
    printf("  Max abs error  : %.6e\n", err);
    // FP4 has only 8 representable values — error up to ~0.25 is expected
    printf("  Correctness    : %s\n\n", err < 0.5f ? "[PASS]" : "[FAIL]");

    quantlib_free(&qm);

    // ── Test 2: FP4 per-channel ───────────────────────────────────────────────
    printf("=== FP4 E2M1 (per-channel) ===\n");
    QuantMatrix qm_pc = quantlib_fp16_to_fp4(
        d_weights, rows, cols, QUANT_PER_CHANNEL, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    quantlib_print_stats(&qm_pc);

    quantlib_fp4_to_fp16(&qm_pc, d_deq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float err_pc = quantlib_quant_error(d_weights, d_deq, n);
    printf("  Max abs error  : %.6e\n", err_pc);
    printf("  Correctness    : %s\n\n", err_pc < 0.5f ? "[PASS]" : "[FAIL]");

    quantlib_free(&qm_pc);

    // ── Test 3: Verify packing roundtrip on known values ─────────────────────
    printf("=== FP4 packing roundtrip (known values) ===\n");
    int   test_n   = 8;
    float test_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, -1.0f, -2.0f, 3.0f, -3.0f};

    __half* h_test = (__half*)malloc(test_n * sizeof(__half));
    for (int i = 0; i < test_n; i++)
        h_test[i] = __float2half(test_vals[i]);

    __half* d_test;
    CUDA_CHECK(cudaMalloc(&d_test, test_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_test, h_test, test_n * sizeof(__half),
                          cudaMemcpyHostToDevice));

    QuantMatrix qm_rt = quantlib_fp16_to_fp4(
        d_test, 1, test_n, QUANT_PER_TENSOR, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    __half* d_rt;
    CUDA_CHECK(cudaMalloc(&d_rt, test_n * sizeof(__half)));
    quantlib_fp4_to_fp16(&qm_rt, d_rt, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    __half* h_rt = (__half*)malloc(test_n * sizeof(__half));
    CUDA_CHECK(cudaMemcpy(h_rt, d_rt, test_n * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < test_n; i++) {
        float orig = test_vals[i];
        float deq  = __half2float(h_rt[i]);
        float err2 = fabsf(orig - deq);
        printf("  [%d] orig=% .3f  deq=% .3f  err=%.4f  %s\n",
               i, orig, deq, err2, err2 < 0.3f ? "OK" : "FAIL");
        if (err2 >= 0.3f) pass = 0;
    }
    printf("  Roundtrip : %s\n\n", pass ? "[PASS]" : "[FAIL]");

    // ── Test 4: Benchmark ─────────────────────────────────────────────────────
    printf("=== Benchmark: FP16 → FP4 (1024x1024, group=128) ===\n");
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    int reps = 100;
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < reps; i++) {
        QuantMatrix qm_tmp = quantlib_fp16_to_fp4(
            d_weights, rows, cols, QUANT_PER_GROUP, 128, stream);
        quantlib_free(&qm_tmp);
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms  = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float avg_ms = ms / reps;
    float gb_s   = (n * sizeof(__half)) / (avg_ms * 1e-3f) / 1e9f;
    printf("  Avg time       : %.3f ms\n", avg_ms);
    printf("  Throughput     : %.1f GB/s\n\n", gb_s);

    quantlib_free(&qm_rt);
    cudaFree(d_weights); cudaFree(d_deq);
    cudaFree(d_test);    cudaFree(d_rt);
    free(h_weights);     free(h_test); free(h_rt);
    cudaStreamDestroy(stream);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    printf("=== All FP4 tests complete ===\n");
    return 0;
}

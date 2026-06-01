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

    // Simulate a linear layer: out=256, in=512
    int out_features  = 256;
    int in_features   = 512;
    int n_calib       = 64;    // 64 calibration tokens
    int group_size    = 128;
    int n_weights     = out_features * in_features;
    int n_acts        = n_calib * in_features;

    // ── Generate synthetic weights with salient channels ──────────────────────
    // Channels 0..31 have 10x larger activations — these are the "salient" ones
    // AWQ should protect them from quantization error.
    __half* h_weights = (__half*)malloc(n_weights * sizeof(__half));
    __half* h_acts    = (__half*)malloc(n_acts    * sizeof(__half));

    srand(42);
    for (int i = 0; i < n_weights; i++)
        h_weights[i] = __float2half(((float)rand()/RAND_MAX)*2.0f - 1.0f);

    for (int t = 0; t < n_calib; t++)
        for (int c = 0; c < in_features; c++) {
            float scale = (c < 32) ? 10.0f : 1.0f;  // salient channels
            h_acts[t * in_features + c] =
                __float2half(scale * (((float)rand()/RAND_MAX)*2.0f - 1.0f));
        }

    __half *d_weights, *d_acts;
    CUDA_CHECK(cudaMalloc(&d_weights, n_weights * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_acts,    n_acts    * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights,
                          n_weights * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_acts, h_acts,
                          n_acts * sizeof(__half), cudaMemcpyHostToDevice));

    // ── Baseline: naive per-group FP8 (no AWQ) ───────────────────────────────
    printf("=== Baseline: naive FP8 E4M3 (no AWQ) ===\n");
    QuantMatrix qm_naive = quantlib_fp16_to_fp8_e4m3(
        d_weights, out_features, in_features,
        QUANT_PER_GROUP, group_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    __half* d_deq_naive;
    CUDA_CHECK(cudaMalloc(&d_deq_naive, n_weights * sizeof(__half)));
    quantlib_fp8_to_fp16(&qm_naive, d_deq_naive, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float err_naive = quantlib_quant_error(d_weights, d_deq_naive, n_weights);
    printf("  Max abs error  : %.6e\n", err_naive);
    quantlib_print_stats(&qm_naive);
    quantlib_free(&qm_naive);

    // ── AWQ quantization ──────────────────────────────────────────────────────
    printf("=== AWQ FP8 E4M3 (activation-aware) ===\n");
    QuantMatrix qm_awq = quantlib_awq_quantize(
        d_weights, out_features, in_features,
        d_acts, n_calib, group_size, 8, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    __half* d_deq_awq;
    CUDA_CHECK(cudaMalloc(&d_deq_awq, n_weights * sizeof(__half)));
    quantlib_fp8_to_fp16(&qm_awq, d_deq_awq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float err_awq = quantlib_quant_error(d_weights, d_deq_awq, n_weights);
    printf("  Max abs error  : %.6e\n", err_awq);
    quantlib_print_stats(&qm_awq);

    // ── Compare salient vs non-salient channel errors ─────────────────────────
    printf("\n=== Salient channel analysis ===\n");
    __half* h_orig     = (__half*)malloc(n_weights * sizeof(__half));
    __half* h_deq_naive = (__half*)malloc(n_weights * sizeof(__half));
    __half* h_deq_awq  = (__half*)malloc(n_weights * sizeof(__half));

    CUDA_CHECK(cudaMemcpy(h_orig,      d_weights,   n_weights*sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_deq_naive, d_deq_naive, n_weights*sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_deq_awq,   d_deq_awq,   n_weights*sizeof(__half), cudaMemcpyDeviceToHost));

    // Error on salient input channels (cols 0..31)
    float salient_err_naive = 0.0f, salient_err_awq = 0.0f;
    float normal_err_naive  = 0.0f, normal_err_awq  = 0.0f;

    for (int row = 0; row < out_features; row++) {
        for (int col = 0; col < in_features; col++) {
            float orig  = __half2float(h_orig[row * in_features + col]);
            float en    = fabsf(orig - __half2float(h_deq_naive[row*in_features+col]));
            float ea    = fabsf(orig - __half2float(h_deq_awq  [row*in_features+col]));
            if (col < 32) {
                salient_err_naive = fmaxf(salient_err_naive, en);
                salient_err_awq   = fmaxf(salient_err_awq,   ea);
            } else {
                normal_err_naive  = fmaxf(normal_err_naive,  en);
                normal_err_awq    = fmaxf(normal_err_awq,    ea);
            }
        }
    }

    printf("  Salient channels (high activation):\n");
    printf("    Naive max err : %.6e\n", salient_err_naive);
    printf("    AWQ   max err : %.6e\n", salient_err_awq);
    printf("    Improvement   : %.2fx\n",
           salient_err_naive / (salient_err_awq + 1e-10f));

    printf("  Normal channels:\n");
    printf("    Naive max err : %.6e\n", normal_err_naive);
    printf("    AWQ   max err : %.6e\n", normal_err_awq);

    int awq_better = (salient_err_awq <= salient_err_naive * 1.1f);
    printf("\n  AWQ salient protection : %s\n", awq_better ? "[PASS]" : "[FAIL]");

    quantlib_free(&qm_awq);
    cudaFree(d_weights); cudaFree(d_acts);
    cudaFree(d_deq_naive); cudaFree(d_deq_awq);
    free(h_weights); free(h_acts);
    free(h_orig); free(h_deq_naive); free(h_deq_awq);
    cudaStreamDestroy(stream);

    printf("\n=== AWQ test complete ===\n");
    return 0;
}

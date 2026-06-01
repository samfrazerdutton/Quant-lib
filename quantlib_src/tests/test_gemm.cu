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

// CPU reference GEMM: C = A x W^T  (fp32)
void cpu_gemm(const __half* A, const __half* W,
              float* C, int M, int N, int K)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += __half2float(A[m*K+k]) * __half2float(W[n*K+k]);
            C[m*N+n] = s;
        }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s\n\n", prop.name);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Small matrix: A [32x64] x W [64x64]^T = C [32x64]
    int M=32, N=64, K=64;
    int nA = M*K, nW = N*K, nC = M*N;

    __half* hA = (__half*)malloc(nA*sizeof(__half));
    __half* hW = (__half*)malloc(nW*sizeof(__half));
    float*  hC_ref = (float*)malloc(nC*sizeof(float));
    __half* hC_gpu = (__half*)malloc(nC*sizeof(__half));

    srand(7);
    for (int i = 0; i < nA; i++)
        hA[i] = __float2half(((float)rand()/RAND_MAX)*0.2f - 0.1f);
    for (int i = 0; i < nW; i++)
        hW[i] = __float2half(((float)rand()/RAND_MAX)*0.2f - 0.1f);

    __half *dA, *dW, *dC;
    CUDA_CHECK(cudaMalloc(&dA, nA*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dW, nW*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, nC*sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dA, hA, nA*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW, nW*sizeof(__half), cudaMemcpyHostToDevice));

    // CPU reference
    cpu_gemm(hA, hW, hC_ref, M, N, K);

    // ── Test 1: FP8 GEMM ──────────────────────────────────────────────────────
    printf("=== FP8 GEMM (PTX Tensor Core, m16n8k8) ===\n");
    QuantMatrix qm_fp8 = quantlib_fp16_to_fp8_e4m3(
        dW, N, K, QUANT_PER_GROUP, 64, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaMemset(dC, 0, nC*sizeof(__half));
    quantlib_fp8_gemm(dA, &qm_fp8, dC, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(hC_gpu, dC, nC*sizeof(__half), cudaMemcpyDeviceToHost));

    float max_err_fp8 = 0.f, mean_err_fp8 = 0.f;
    for (int i = 0; i < nC; i++) {
        float err = fabsf(__half2float(hC_gpu[i]) - hC_ref[i]);
        if (err > max_err_fp8) max_err_fp8 = err;
        mean_err_fp8 += err;
    }
    mean_err_fp8 /= nC;
    printf("  Max  abs error vs CPU : %.6e\n", max_err_fp8);
    printf("  Mean abs error vs CPU : %.6e\n", mean_err_fp8);
    printf("  Correctness           : %s\n\n",
           max_err_fp8 < 1.0f ? "[PASS]" : "[FAIL]");
    quantlib_free(&qm_fp8);

    // ── Test 2: FP4 GEMM ──────────────────────────────────────────────────────
    printf("=== FP4 GEMM (PTX Tensor Core, m16n8k8) ===\n");
    QuantMatrix qm_fp4 = quantlib_fp16_to_fp4(
        dW, N, K, QUANT_PER_GROUP, 64, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaMemset(dC, 0, nC*sizeof(__half));
    quantlib_fp4_gemm(dA, &qm_fp4, dC, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(hC_gpu, dC, nC*sizeof(__half), cudaMemcpyDeviceToHost));

    float max_err_fp4 = 0.f, mean_err_fp4 = 0.f;
    for (int i = 0; i < nC; i++) {
        float err = fabsf(__half2float(hC_gpu[i]) - hC_ref[i]);
        if (err > max_err_fp4) max_err_fp4 = err;
        mean_err_fp4 += err;
    }
    mean_err_fp4 /= nC;
    printf("  Max  abs error vs CPU : %.6e\n", max_err_fp4);
    printf("  Mean abs error vs CPU : %.6e\n", mean_err_fp4);
    printf("  Correctness           : %s\n\n",
           max_err_fp4 < 2.0f ? "[PASS]" : "[FAIL]");
    quantlib_free(&qm_fp4);

    // ── Benchmark ─────────────────────────────────────────────────────────────
    printf("=== GEMM Benchmark (M=512 N=512 K=512) ===\n");
    int BM=512, BN=512, BK=512;
    __half *dA2, *dW2, *dC2;
    CUDA_CHECK(cudaMalloc(&dA2, BM*BK*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dW2, BN*BK*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC2, BM*BN*sizeof(__half)));

    QuantMatrix qm_b8 = quantlib_fp16_to_fp8_e4m3(
        dW2, BN, BK, QUANT_PER_GROUP, 128, stream);
    QuantMatrix qm_b4 = quantlib_fp16_to_fp4(
        dW2, BN, BK, QUANT_PER_GROUP, 128, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    int reps = 200;
    float ms;

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < reps; i++)
        quantlib_fp8_gemm(dA2, &qm_b8, dC2, BM, BN, BK, stream);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("  FP8 GEMM avg : %.3f ms  (%.1f GFLOPS)\n",
           ms/reps, 2.0f*BM*BN*BK/(ms/reps*1e-3f)/1e9f);

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < reps; i++)
        quantlib_fp4_gemm(dA2, &qm_b4, dC2, BM, BN, BK, stream);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("  FP4 GEMM avg : %.3f ms  (%.1f GFLOPS)\n",
           ms/reps, 2.0f*BM*BN*BK/(ms/reps*1e-3f)/1e9f);

    quantlib_free(&qm_b8);
    quantlib_free(&qm_b4);
    cudaFree(dA); cudaFree(dW); cudaFree(dC);
    cudaFree(dA2); cudaFree(dW2); cudaFree(dC2);
    free(hA); free(hW); free(hC_ref); free(hC_gpu);
    cudaStreamDestroy(stream);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    printf("\n=== GEMM tests complete ===\n");
    return 0;
}

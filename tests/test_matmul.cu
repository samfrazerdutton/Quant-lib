#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

extern void launch_matmul(const float*, const float*, float*,
                          int, int, int, cudaStream_t);

void cpu_matmul(const float* A, const float* B, float* C,
                int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += A[i*K + k] * B[k*N + j];
            C[i*N + j] = s;
        }
}

void bench(int M, int N, int K,
           float* dA, float* dB, float* dC, cudaStream_t stream) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch_matmul(dA, dB, dC, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < 200; i++)
        launch_matmul(dA, dB, dC, M, N, K, stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 200.0f;

    double tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12;
    printf("  %4dx%4dx%4d | %7.4f ms | %6.3f TFLOPS\n", M, N, K, ms, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    // ── Correctness check at 512x512 ─────────────────────────────────────
    int M = 512, N = 512, K = 512;
    size_t sA = M*K*sizeof(float);
    size_t sB = K*N*sizeof(float);
    size_t sC = M*N*sizeof(float);

    float *hA      = (float*)malloc(sA);
    float *hB      = (float*)malloc(sB);
    float *hC_gpu  = (float*)malloc(sC);
    float *hC_cpu  = (float*)malloc(sC);

    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = (float)rand()/RAND_MAX - 0.5f;
    for (int i = 0; i < K*N; i++) hB[i] = (float)rand()/RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sA));
    CUDA_CHECK(cudaMalloc(&dB, sB));
    CUDA_CHECK(cudaMalloc(&dC, sC));
    CUDA_CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    launch_matmul(dA, dB, dC, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hC_gpu, dC, sC, cudaMemcpyDeviceToHost));
    cpu_matmul(hA, hB, hC_cpu, M, N, K);

    float maxErr = 0;
    for (int i = 0; i < M*N; i++)
        maxErr = fmaxf(maxErr, fabsf(hC_gpu[i] - hC_cpu[i]));
    printf("Correctness (512x512): max_err=%.2e  %s\n\n",
           maxErr, maxErr < 1e-2 ? "[PASS]" : "[FAIL]");

    // ── Benchmark sweep ───────────────────────────────────────────────────
    printf("Benchmark:\n");
    printf("  %4s x%4s x%4s | %9s | %s\n", "M","N","K","time","TFLOPS");
    printf("  -----------------------------------------------\n");

    int BIG = 4096;
    float *dA2, *dB2, *dC2;
    CUDA_CHECK(cudaMalloc(&dA2, (size_t)BIG*BIG*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB2, (size_t)BIG*BIG*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC2, (size_t)BIG*BIG*sizeof(float)));

    int sizes[] = {512, 1024, 2048, 4096};
    for (int s = 0; s < 4; s++)
        bench(sizes[s], sizes[s], sizes[s], dA2, dB2, dC2, stream);

    cudaFree(dA);  cudaFree(dB);  cudaFree(dC);
    cudaFree(dA2); cudaFree(dB2); cudaFree(dC2);
    free(hA); free(hB); free(hC_gpu); free(hC_cpu);
    return 0;
}

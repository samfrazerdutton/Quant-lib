#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

extern void launch_softmax(const float*, float*, int, int, cudaStream_t);

void cpu_softmax(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float mx = -FLT_MAX;
        for (int c = 0; c < cols; c++) mx = fmaxf(mx, in[r*cols+c]);
        float s = 0;
        for (int c = 0; c < cols; c++) {
            out[r*cols+c] = expf(in[r*cols+c] - mx);
            s += out[r*cols+c];
        }
        for (int c = 0; c < cols; c++) out[r*cols+c] /= s;
    }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s\n\n", prop.name);

    // Dimensions matching real attention: 2048 rows (tokens), 2048 cols (seq_len)
    int rows = 2048, cols = 2048;
    size_t sz = rows * cols * sizeof(float);

    float *hIn  = (float*)malloc(sz);
    float *hOut_gpu = (float*)malloc(sz);
    float *hOut_cpu = (float*)malloc(sz);

    srand(7);
    for (int i = 0; i < rows*cols; i++)
        hIn[i] = (float)rand()/RAND_MAX * 4.0f - 2.0f;

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn,  sz));
    CUDA_CHECK(cudaMalloc(&dOut, sz));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, sz, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    launch_softmax(dIn, dOut, rows, cols, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < 1000; i++)
        launch_softmax(dIn, dOut, rows, cols, stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 1000.0f;

    // Bandwidth = read input + write output = 2 * sz bytes
    double gb_s = (2.0 * sz) / (ms * 1e-3) / 1e9;
    printf("Softmax %dx%d\n", rows, cols);
    printf("  Time      : %.4f ms\n", ms);
    printf("  Bandwidth : %.1f GB/s\n", gb_s);
    printf("  Peak DRAM  : ~%.0f GB/s  (RTX 2060 Max-Q spec)\n\n", 336.0);

    // Correctness
    CUDA_CHECK(cudaMemcpy(hOut_gpu, dOut, sz, cudaMemcpyDeviceToHost));
    cpu_softmax(hIn, hOut_cpu, rows, cols);

    float maxErr = 0, sumErr = 0;
    for (int i = 0; i < rows*cols; i++) {
        float e = fabsf(hOut_gpu[i] - hOut_cpu[i]);
        if (e > maxErr) maxErr = e;
        sumErr += e;
    }
    printf("  Max abs error  : %e\n", maxErr);
    printf("  Mean abs error : %e\n", sumErr / (rows*cols));

    // Each row must sum to 1.0
    float maxSumErr = 0;
    for (int r = 0; r < rows; r++) {
        float s = 0;
        for (int c = 0; c < cols; c++) s += hOut_gpu[r*cols+c];
        maxSumErr = fmaxf(maxSumErr, fabsf(s - 1.0f));
    }
    printf("  Max row-sum err: %e\n", maxSumErr);
    printf("  Correctness    : %s\n",
           (maxErr < 1e-5 && maxSumErr < 1e-5) ? "[PASS]" : "[FAIL]");

    cudaFree(dIn); cudaFree(dOut);
    free(hIn); free(hOut_gpu); free(hOut_cpu);
    return 0;
}

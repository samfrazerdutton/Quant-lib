#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/kv_cache.h"
#include "../include/weights.h"
#include "quantlib.h"

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d -- %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } } while(0)

__global__ void fp32_to_fp16_kernel(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void fp16_to_fp32_kernel(const __half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

static void tensor_to_qtensor(const Tensor* src, QTensor* dst,
                               int group_size, cudaStream_t stream)
{
    int n = src->rows * src->cols;
    __half* d_fp16;
    CUDA_CHECK(cudaMalloc(&d_fp16, n * sizeof(__half)));

    int threads = 256, blocks = (n + 255) / 256;
    fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        src->data, d_fp16, n);

    QuantMatrix qm = quantlib_fp16_to_fp8_e4m3(
        d_fp16, src->rows, src->cols,
        QUANT_PER_GROUP, group_size, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    dst->data        = qm.data;
    dst->scales      = qm.params.scales;
    dst->zeros       = qm.params.zeros;
    dst->rows        = qm.rows;
    dst->cols        = qm.cols;
    dst->bits        = qm.bits;
    dst->group_size  = qm.params.group_size;
    dst->n_scales    = qm.params.n_scales;
    dst->valid       = 1;
    memcpy(dst->format, qm.format, 16);

    cudaFree(d_fp16);
}

void quantize_model(ModelWeights* w, int group_size, cudaStream_t stream)
{
    printf("Quantizing %d layers to FP8 E4M3 (group=%d)...\n",
           w->n_layers, group_size);

    size_t freed = 0;
    for (int l = 0; l < w->n_layers; l++) {
        LayerWeights* lw = &w->layers[l];

        tensor_to_qtensor(&lw->wq, &lw->qwq, group_size, stream);
        tensor_to_qtensor(&lw->wk, &lw->qwk, group_size, stream);
        tensor_to_qtensor(&lw->wv, &lw->qwv, group_size, stream);
        tensor_to_qtensor(&lw->wo, &lw->qwo, group_size, stream);
        tensor_to_qtensor(&lw->w1, &lw->qw1, group_size, stream);
        tensor_to_qtensor(&lw->w2, &lw->qw2, group_size, stream);
        tensor_to_qtensor(&lw->w3, &lw->qw3, group_size, stream);

        freed += (size_t)(lw->wq.rows*lw->wq.cols +
                          lw->wk.rows*lw->wk.cols +
                          lw->wv.rows*lw->wv.cols +
                          lw->wo.rows*lw->wo.cols +
                          lw->w1.rows*lw->w1.cols +
                          lw->w2.rows*lw->w2.cols +
                          lw->w3.rows*lw->w3.cols) * sizeof(float);

        cudaFree(lw->wq.data); lw->wq.data = nullptr;
        cudaFree(lw->wk.data); lw->wk.data = nullptr;
        cudaFree(lw->wv.data); lw->wv.data = nullptr;
        cudaFree(lw->wo.data); lw->wo.data = nullptr;
        cudaFree(lw->w1.data); lw->w1.data = nullptr;
        cudaFree(lw->w2.data); lw->w2.data = nullptr;
        cudaFree(lw->w3.data); lw->w3.data = nullptr;

        if ((l+1) % 4 == 0)
            printf("  layer %2d/%d done\n", l+1, w->n_layers);
    }

    printf("FP32 freed : %.0f MB\n", freed / 1e6f);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU after quant: %zu MB used / %zu MB total\n\n",
           (total_mem - free_mem) / (1<<20), total_mem / (1<<20));
}

void qgemm(const __half* A, const QTensor* W, __half* C,
           int M, int N, int K, cudaStream_t stream)
{
    QuantMatrix qm;
    qm.data               = W->data;
    qm.params.scales      = W->scales;
    qm.params.zeros       = W->zeros;
    qm.params.group_size  = W->group_size;
    qm.params.n_scales    = W->n_scales;
    qm.params.granularity = QUANT_PER_GROUP;
    qm.rows               = W->rows;
    qm.cols               = W->cols;
    qm.bits               = W->bits;
    memcpy(qm.format, W->format, 16);
    quantlib_fp8_gemm(A, &qm, C, M, N, K, stream);
}

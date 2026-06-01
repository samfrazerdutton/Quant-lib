#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/kv_cache.h"
#include "../include/weights.h"

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d -- %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } } while(0)

extern ModelWeights* weights_load_bin(const char*, int, int, int, int, int);
extern void          quantize_model(ModelWeights*, int, cudaStream_t);

struct ForwardState;
extern ForwardState* forward_state_create(int, int, int, int);
extern void          forward_state_free(ForwardState*);
extern float*        model_forward(cublasHandle_t, ModelWeights*,
                                   KVCache*, ForwardState*,
                                   int, cudaStream_t);

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1]
                             : "/home/samfrazerdutton/tinyllama/weights.bin";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s  (%zu MB VRAM)\n\n", prop.name,
           (size_t)prop.totalGlobalMem / (1<<20));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    // TinyLlama-1.1B config
    int n_layers   = 22;
    int hidden_dim = 2048;
    int n_heads    = 32;
    int head_dim   = 64;
    int vocab_size = 32000;

    // ── Baseline: fp32 memory usage ───────────────────────────────────────────
    printf("=== Step 1: Load weights (FP32) ===\n");
    size_t free0, total;
    cudaMemGetInfo(&free0, &total);

    ModelWeights* w = weights_load_bin(weights_path,
                                        n_layers, hidden_dim,
                                        n_heads, head_dim, vocab_size);
    if (!w) { fprintf(stderr, "Failed to load weights\n"); return 1; }

    size_t free1;
    cudaMemGetInfo(&free1, &total);
    printf("FP32 weights VRAM : %zu MB\n\n",
           (free0 - free1) / (1<<20));

    // ── Benchmark fp32 generation ─────────────────────────────────────────────
    printf("=== Step 2: FP32 baseline generation (16 tokens) ===\n");
    KVCache*      kv_fp32 = kvcache_create(n_layers, 512,
                                            w->kv_dim > 0 ? 4 : n_heads,
                                            head_dim);
    ForwardState* fs_fp32 = forward_state_create(hidden_dim, w->kv_dim > 0
                                                  ? w->kv_dim : hidden_dim,
                                                  vocab_size, n_layers);

    int prompt[] = {1, 1724, 338};  // "What is"
    int n_prompt = 3;

    // Warm up
    model_forward(cublas, w, kv_fp32, fs_fp32, prompt[0], stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    kv_fp32->current_len = 0;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < 16; i++)
        model_forward(cublas, w, kv_fp32, fs_fp32,
                      prompt[i % n_prompt], stream);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms_fp32;
    CUDA_CHECK(cudaEventElapsedTime(&ms_fp32, t0, t1));
    printf("  FP32 : 16 tokens in %.1f ms  (%.1f tok/s)\n\n",
           ms_fp32, 16000.0f / ms_fp32);

    kvcache_free(kv_fp32);
    forward_state_free(fs_fp32);

    // ── Quantize to FP8 ───────────────────────────────────────────────────────
    printf("=== Step 3: Quantize to FP8 E4M3 (group=128) ===\n");
    size_t free2;
    cudaMemGetInfo(&free2, &total);

    quantize_model(w, 128, stream);

    size_t free3;
    cudaMemGetInfo(&free3, &total);
    printf("FP8 weights VRAM  : %zu MB\n",   (free0 - free3) / (1<<20));
    { size_t sv = free3 < free2 ? (free2-free3)/(1<<20) : (free3-free2)/(1<<20); printf("VRAM delta        : %zu MB\n\n", sv); }

    // ── Benchmark fp8 generation ──────────────────────────────────────────────
    printf("=== Step 4: FP8 quantized generation (16 tokens) ===\n");
    KVCache*      kv_fp8 = kvcache_create(n_layers, 512,
                                           w->kv_dim > 0 ? 4 : n_heads,
                                           head_dim);
    ForwardState* fs_fp8 = forward_state_create(hidden_dim, w->kv_dim > 0
                                                 ? w->kv_dim : hidden_dim,
                                                 vocab_size, n_layers);

    // Warm up
    model_forward(cublas, w, kv_fp8, fs_fp8, prompt[0], stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    kv_fp8->current_len = 0;

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < 16; i++)
        model_forward(cublas, w, kv_fp8, fs_fp8,
                      prompt[i % n_prompt], stream);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms_fp8;
    CUDA_CHECK(cudaEventElapsedTime(&ms_fp8, t0, t1));
    printf("  FP8  : 16 tokens in %.1f ms  (%.1f tok/s)\n",
           ms_fp8, 16000.0f / ms_fp8);
    printf("  Speedup vs FP32 : %.2fx\n\n", ms_fp32 / ms_fp8);

    // ── Summary ───────────────────────────────────────────────────────────────
    printf("=== Summary ===\n");
    printf("  FP32 tok/s : %.1f\n",   16000.0f / ms_fp32);
    printf("  FP8  tok/s : %.1f\n",   16000.0f / ms_fp8);
    printf("  Speedup    : %.2fx\n",  ms_fp32 / ms_fp8);
    { size_t sv = free3 < free2 ? (free2-free3)/(1<<20) : (free3-free2)/(1<<20); printf("  VRAM saved : %zu MB\n", sv); }

    kvcache_free(kv_fp8);
    forward_state_free(fs_fp8);
    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return 0;
}

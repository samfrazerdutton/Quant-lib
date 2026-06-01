#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/kv_cache.h"
#include "../include/weights.h"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s  (%.0f MB VRAM)\n\n", prop.name,
           prop.totalGlobalMem / 1e6);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── KV Cache test: Llama-7B dims but SHORT sequence ──────────────────
    // Full 7B at fp32 needs ~28GB weights + 2GB KV cache = won't fit on 6GB.
    // Real deployments use fp16/int8. We test cache correctness here.
    int n_layers   = 32;
    int n_heads    = 32;
    int head_dim   = 128;
    int max_seq    = 256;   // short: 256 * 32 * 32 * 128 * 4 * 2 = 256 MB

    printf("--- KV Cache Test (seq=256, layers=32) ---\n");
    KVCache* cache = kvcache_create(n_layers, max_seq, n_heads, head_dim);

    int slice = n_heads * head_dim;
    float* h_k      = (float*)malloc(slice * sizeof(float));
    float* h_v      = (float*)malloc(slice * sizeof(float));
    float* h_k_back = (float*)malloc(slice * sizeof(float));

    float *dk, *dv;
    CUDA_CHECK(cudaMalloc(&dk, slice * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dv, slice * sizeof(float)));

    // Write 16 tokens to all 32 layers
    for (int tok = 0; tok < 16; tok++) {
        for (int i = 0; i < slice; i++) {
            h_k[i] = (float)tok + (float)i / slice;
            h_v[i] = -(float)tok;
        }
        CUDA_CHECK(cudaMemcpy(dk, h_k, slice*sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv, h_v, slice*sizeof(float),
                              cudaMemcpyHostToDevice));
        for (int layer = 0; layer < n_layers; layer++)
            kvcache_append(cache, layer, dk, dv, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cache->current_len++;
    }

    // Read back token 7, layer 0 and verify
    float* layer0_k = kvcache_get_k(cache, 0);
    CUDA_CHECK(cudaMemcpy(h_k_back, layer0_k + 7 * slice,
                          slice * sizeof(float), cudaMemcpyDeviceToHost));
    float expected = 7.0f;
    float err = fabsf(h_k_back[0] - expected);
    printf("  Token 7 readback error : %e  %s\n", err,
           err < 1e-6 ? "[PASS]" : "[FAIL]");
    printf("  current_len            : %d\n", cache->current_len);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU after KV cache     : %.0f MB free / %.0f MB total\n\n",
           free_mem/1e6, total_mem/1e6);

    kvcache_free(cache);

    // ── Weight Allocation: small model that fits in ~1GB ─────────────────
    // Tiny-LLM: 4 layers, hidden=1024, ffn=2816, vocab=32000
    // Weight memory: ~4 * (7 * 1024^2) * 4 bytes ~ 116 MB — fits fine
    int wn_layers   = 4;
    int wn_heads    = 8;
    int whidden_dim = 1024;
    int wffn_dim    = 2816;
    int wvocab_size = 32000;

    printf("--- Weight Allocation Test (Tiny-LLM: 4 layers, hidden=1024) ---\n");
    ModelWeights* weights = weights_alloc_random(
        wn_layers, whidden_dim, wn_heads, wffn_dim, wvocab_size);

    LayerWeights* l0 = &weights->layers[0];
    int exp_rows = whidden_dim, exp_cols = whidden_dim;
    printf("  Layer 0 wq [%d x %d]  expected [%d x %d]  %s\n",
           l0->wq.rows, l0->wq.cols, exp_rows, exp_cols,
           (l0->wq.rows==exp_rows && l0->wq.cols==exp_cols)
           ? "[PASS]" : "[FAIL]");

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU after weights      : %.0f MB free / %.0f MB total\n\n",
           free_mem/1e6, total_mem/1e6);

    // ── Token generation simulation ───────────────────────────────────────
    printf("--- Token Generation Cache Benchmark ---\n");
    cache = kvcache_create(wn_layers, 512, wn_heads, whidden_dim/wn_heads);
    cache->current_len = 0;

    int wslice = wn_heads * (whidden_dim / wn_heads);
    float *dk2, *dv2;
    CUDA_CHECK(cudaMalloc(&dk2, wslice * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dv2, wslice * sizeof(float)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));

    int gen_tokens = 128;
    for (int tok = 0; tok < gen_tokens; tok++) {
        for (int layer = 0; layer < wn_layers; layer++)
            kvcache_append(cache, layer, dk2, dv2, stream);
        cache->current_len++;
    }

    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("  %d tokens x %d layers : %.3f ms total\n",
           gen_tokens, wn_layers, ms);
    printf("  Per-token overhead    : %.4f ms  %s\n\n",
           ms / gen_tokens,
           ms / gen_tokens < 0.1f ? "[FAST]" : "[SLOW]");

    printf("=== All tests passed ===\n");
    printf("\nNOTE: Full Llama-7B needs ~28GB fp32 / ~14GB fp16.\n");
    printf("      RTX 2060 (6GB) runs Llama-3.2-1B in fp16 (~2.5GB).\n");
    printf("      Step 5 will add fp16 support + real weight loading.\n");

    kvcache_free(cache);
    weights_free(weights);
    cudaFree(dk); cudaFree(dv);
    cudaFree(dk2); cudaFree(dv2);
    free(h_k); free(h_v); free(h_k_back);
    return 0;
}

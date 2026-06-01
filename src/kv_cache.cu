#include "kv_cache.h"
#include <stdlib.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

KVCache* kvcache_create(int n_layers, int max_seq_len,
                        int n_heads, int head_dim) {
    KVCache* c = (KVCache*)malloc(sizeof(KVCache));
    c->n_layers    = n_layers;
    c->max_seq_len = max_seq_len;
    c->n_heads     = n_heads;
    c->head_dim    = head_dim;
    c->current_len = 0;

    size_t total = (size_t)n_layers * max_seq_len * n_heads * head_dim;
    size_t bytes  = total * sizeof(float);

    CUDA_CHECK(cudaMalloc(&c->k_cache, bytes));
    CUDA_CHECK(cudaMalloc(&c->v_cache, bytes));
    CUDA_CHECK(cudaMemset(c->k_cache, 0, bytes));
    CUDA_CHECK(cudaMemset(c->v_cache, 0, bytes));

    printf("KV Cache allocated: %.2f MB (%.0f MB x2 for K+V)\n",
           bytes / 1e6, bytes / 1e6);
    return c;
}

void kvcache_free(KVCache* cache) {
    cudaFree(cache->k_cache);
    cudaFree(cache->v_cache);
    free(cache);
}

// Write ONE layer's K,V at position current_len
void kvcache_append(KVCache* cache, int layer,
                    const float* new_k,
                    const float* new_v,
                    cudaStream_t stream) {
    int    pos        = cache->current_len;
    int    slice_dim  = cache->n_heads * cache->head_dim;
    size_t bytes      = slice_dim * sizeof(float);
    size_t layer_stride = (size_t)cache->max_seq_len * slice_dim;
    size_t offset     = (size_t)layer * layer_stride + (size_t)pos * slice_dim;

    CUDA_CHECK(cudaMemcpyAsync(cache->k_cache + offset, new_k,
                               bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(cache->v_cache + offset, new_v,
                               bytes, cudaMemcpyDeviceToDevice, stream));
}

float* kvcache_get_k(KVCache* cache, int layer) {
    size_t layer_stride = (size_t)cache->max_seq_len
                        * cache->n_heads * cache->head_dim;
    return cache->k_cache + layer * layer_stride;
}

float* kvcache_get_v(KVCache* cache, int layer) {
    size_t layer_stride = (size_t)cache->max_seq_len
                        * cache->n_heads * cache->head_dim;
    return cache->v_cache + layer * layer_stride;
}

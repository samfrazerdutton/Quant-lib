#pragma once
#include <cuda_runtime.h>
#include <stddef.h>

// ── KV Cache ──────────────────────────────────────────────────────────────────
// Stores Key and Value tensors for all past tokens so we never recompute them.
// This is the core memory structure behind fast autoregressive generation.
//
// Layout: [n_layers, max_seq_len, n_heads, head_dim]
// On each new token we append one slice at position `current_len`.

typedef struct {
    float* k_cache;      // [n_layers * max_seq_len * n_heads * head_dim]
    float* v_cache;
    int    n_layers;
    int    max_seq_len;
    int    n_heads;
    int    head_dim;
    int    current_len;  // how many tokens have been cached so far
} KVCache;

// Allocate cache entirely on GPU
KVCache* kvcache_create(int n_layers, int max_seq_len,
                        int n_heads, int head_dim);

void kvcache_free(KVCache* cache);

// Write new K,V slices for one layer at current_len position
void kvcache_append(KVCache* cache, int layer,
                    const float* new_k,   // [1, n_heads, head_dim] on GPU
                    const float* new_v,
                    cudaStream_t stream);

// Get pointer to full K history for one layer: [current_len, n_heads, head_dim]
float* kvcache_get_k(KVCache* cache, int layer);
float* kvcache_get_v(KVCache* cache, int layer);

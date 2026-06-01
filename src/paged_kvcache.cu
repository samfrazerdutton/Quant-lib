#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/paged_kvcache.h"

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d -- %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } } while(0)

extern "C" PagedKVCache* paged_kv_create(int n_blocks, int n_layers, int n_kv_heads, int head_dim, int max_seqs, int max_blocks_per_seq) {
    PagedKVCache* cache = (PagedKVCache*)malloc(sizeof(PagedKVCache));
    cache->n_blocks = n_blocks;
    cache->n_layers = n_layers;
    cache->n_kv_heads = n_kv_heads;
    cache->head_dim = head_dim;
    cache->max_seqs = max_seqs;
    cache->max_blocks_per_seq = max_blocks_per_seq;

    // Allocate the contiguous memory pool on the device
    size_t block_sz = (size_t)n_layers * PAGE_SIZE * n_kv_heads * head_dim * sizeof(__half);
    
    __half* d_k_pool;
    __half* d_v_pool;
    CUDA_CHECK(cudaMalloc(&d_k_pool, block_sz * n_blocks));
    CUDA_CHECK(cudaMalloc(&d_v_pool, block_sz * n_blocks));

    // Create host arrays of pointers to map out the blocks
    __half** h_k_blocks = (__half**)malloc(n_blocks * sizeof(__half*));
    __half** h_v_blocks = (__half**)malloc(n_blocks * sizeof(__half*));
    
    for(int i = 0; i < n_blocks; i++) {
        h_k_blocks[i] = d_k_pool + i * (block_sz / sizeof(__half));
        h_v_blocks[i] = d_v_pool + i * (block_sz / sizeof(__half));
    }

    // Allocate device arrays of pointers for kernel lookups
    CUDA_CHECK(cudaMalloc(&cache->k_blocks, n_blocks * sizeof(__half*)));
    CUDA_CHECK(cudaMalloc(&cache->v_blocks, n_blocks * sizeof(__half*)));
    CUDA_CHECK(cudaMemcpy(cache->k_blocks, h_k_blocks, n_blocks * sizeof(__half*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache->v_blocks, h_v_blocks, n_blocks * sizeof(__half*), cudaMemcpyHostToDevice));

    // Setup the LIFO free blocks stack
    cache->free_blocks = (int*)malloc(n_blocks * sizeof(int));
    for(int i = 0; i < n_blocks; i++) {
        cache->free_blocks[i] = n_blocks - 1 - i; 
    }
    cache->free_count = n_blocks;

    // Initialize sequence block tables
    cache->block_tables = (int**)malloc(max_seqs * sizeof(int*));
    for(int i = 0; i < max_seqs; i++) {
        cache->block_tables[i] = (int*)malloc(max_blocks_per_seq * sizeof(int));
        for(int j=0; j < max_blocks_per_seq; j++) cache->block_tables[i][j] = -1;
    }

    free(h_k_blocks);
    free(h_v_blocks);
    return cache;
}

extern "C" void paged_kv_free(PagedKVCache* cache) {
    if (!cache) return;
    
    // Retrieve base pool pointer from device to free contiguous blocks
    __half* d_k_pool;
    __half* d_v_pool;
    CUDA_CHECK(cudaMemcpy(&d_k_pool, cache->k_blocks, sizeof(__half*), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&d_v_pool, cache->v_blocks, sizeof(__half*), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_k_pool));
    CUDA_CHECK(cudaFree(d_v_pool));
    CUDA_CHECK(cudaFree(cache->k_blocks));
    CUDA_CHECK(cudaFree(cache->v_blocks));

    free(cache->free_blocks);
    for(int i = 0; i < cache->max_seqs; i++) {
        free(cache->block_tables[i]);
    }
    free(cache->block_tables);
    free(cache);
}

extern "C" int paged_kv_allocate_block(PagedKVCache* cache) {
    if (cache->free_count == 0) return -1; // Out of memory
    cache->free_count--;
    return cache->free_blocks[cache->free_count];
}

extern "C" void paged_kv_free_block(PagedKVCache* cache, int block_idx) {
    cache->free_blocks[cache->free_count] = block_idx;
    cache->free_count++;
}

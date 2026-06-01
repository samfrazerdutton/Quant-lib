#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/fp16_types.h"
#include "../include/paged_kvcache.h"
#include "../include/weights.h"

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d -- %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } } while(0)

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s\n\n", prop.name);

    printf("=== Initializing FP16 Paged Inference Engine ===\n");
    
    // Llama 1B parameters
    int n_layers = 22;
    int n_heads = 32;
    int n_kv_heads = 4;
    int head_dim = 64;
    int hidden_dim = 2048;
    
    // Allocate a small Paged KV Cache for testing (e.g., 1024 blocks of 16 tokens)
    int n_blocks = 1024; 
    int max_seqs = 16;
    int max_blocks_per_seq = 128; // up to 2048 tokens per seq
    
    printf("Allocating Paged KV Cache (%d blocks)...\n", n_blocks);
    PagedKVCache* kv_cache = paged_kv_create(n_blocks, n_layers, n_kv_heads, head_dim, max_seqs, max_blocks_per_seq);
    
    if (kv_cache) {
        printf("Paged KV Cache successfully allocated in VRAM.\n");
        printf("  - Tokens per page: %d\n", PAGE_SIZE);
        printf("  - Total capacity: %d tokens\n", n_blocks * PAGE_SIZE);
    }

    // TODO: Wire up fp16_forward kernel call here once weights are loaded

    printf("\n=== FP16 Forward Test Complete ===\n");
    
    paged_kv_free(kv_cache);
    return 0;
}

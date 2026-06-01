#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ── Paged KV Cache Structures ─────────────────────────────────────────────────
#define PAGE_SIZE 16  // tokens per page/block

typedef struct {
    __half** k_blocks;     // [n_blocks] GPU pointers to K pages
    __half** v_blocks;     // [n_blocks] GPU pointers to V pages
    int      n_blocks;
    int      n_layers;
    int      n_kv_heads;
    int      head_dim;
    
    // Block tracking
    int* free_blocks;  // Stack of available block indices
    int      free_count;
    
    // Sequence mapping: seq_id -> array of block indices
    int** block_tables; 
    int      max_seqs;
    int      max_blocks_per_seq;
} PagedKVCache;

#ifdef __cplusplus
extern "C" {
#endif

PagedKVCache* paged_kv_create(int n_blocks, int n_layers, int n_kv_heads, int head_dim, int max_seqs, int max_blocks_per_seq);
void paged_kv_free(PagedKVCache* cache);
int paged_kv_allocate_block(PagedKVCache* cache);
void paged_kv_free_block(PagedKVCache* cache, int block_idx);

#ifdef __cplusplus
}
#endif

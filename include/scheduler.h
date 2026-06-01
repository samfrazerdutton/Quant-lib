#pragma once
#include "paged_kvcache.h"
#include "weights.h"
#include <stdint.h>

typedef enum { PENDING, PREFILL, DECODE, FINISHED } ReqState;

typedef struct {
    int req_id;
    int* tokens;          // All tokens (prompt + generated)
    int seq_len;          // Current total length
    int prompt_len;       // Length of initial prompt
    int max_tokens;       // Max tokens to generate
    ReqState state;
    
    // Paged KV routing
    int* logical_blocks;  // Array of physical block indices allocated to this sequence
    int num_blocks;
    int max_blocks;
    
    // Benchmarking metrics
    double start_time;
    double prefill_time;
    double total_decode_time;
    int tokens_generated;
} Sequence;

typedef struct {
    Sequence** active_seqs;
    int num_active;
    int max_active;
    
    Sequence** pending_seqs;
    int num_pending;
    
    PagedKVCache* kv_cache;
    ModelWeights* weights; // Hooked directly to loaded GGUF structures
    
    // Global output buffer for intermediate logits
    float* d_logits;
} Scheduler;

#ifdef __cplusplus
extern "C" {
#endif

Scheduler* scheduler_create(PagedKVCache* kv_cache, ModelWeights* weights, int max_active);
void scheduler_add_request(Scheduler* sched, int req_id, const int* prompt, int prompt_len, int max_gen);
void scheduler_step(Scheduler* sched);
void scheduler_free(Scheduler* sched);

#ifdef __cplusplus
}
#endif

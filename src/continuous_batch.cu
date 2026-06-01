#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../include/scheduler.h"

// Declaration of your external FP16 forward pass pipeline
extern "C" void fp16_forward(ModelWeights* weights, int* tokens, int seq_len, int prompt_len, PagedKVCache* kv, int* block_table, int num_blocks, float* d_logits);

// Inline simple argmax kernel for greedy token selection
__global__ void argmax_kernel(const float* logits, int vocab_size, int* out_token) {
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid != 0) return; // Single-thread execution for simple mock sampling

    float max_val = -1e20f;
    int max_idx = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    *out_token = max_idx;
}

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

extern "C" Scheduler* scheduler_create(PagedKVCache* kv_cache, ModelWeights* weights, int max_active) {
    Scheduler* s = (Scheduler*)malloc(sizeof(Scheduler));
    s->kv_cache = kv_cache;
    s->weights = weights;
    s->max_active = max_active;
    s->active_seqs = (Sequence**)malloc(max_active * sizeof(Sequence*));
    s->num_active = 0;
    s->pending_seqs = (Sequence**)malloc(1024 * sizeof(Sequence*));
    s->num_pending = 0;
    
    // Allocate GPU buffer for final vocabulary logit distribution
    cudaMalloc(&s->d_logits, weights->vocab_size * sizeof(float));
    return s;
}

extern "C" void scheduler_add_request(Scheduler* sched, int req_id, const int* prompt, int prompt_len, int max_gen) {
    Sequence* seq = (Sequence*)malloc(sizeof(Sequence));
    seq->req_id = req_id;
    seq->max_tokens = prompt_len + max_gen;
    seq->tokens = (int*)malloc(seq->max_tokens * sizeof(int));
    memcpy(seq->tokens, prompt, prompt_len * sizeof(int));
    seq->seq_len = prompt_len;
    seq->prompt_len = prompt_len;
    seq->state = PENDING;
    
    seq->max_blocks = (seq->max_tokens + PAGE_SIZE - 1) / PAGE_SIZE;
    seq->logical_blocks = (int*)malloc(seq->max_blocks * sizeof(int));
    seq->num_blocks = 0;
    
    seq->start_time = get_time_ms();
    seq->prefill_time = 0.0;
    seq->total_decode_time = 0.0;
    seq->tokens_generated = 0;
    
    sched->pending_seqs[sched->num_pending++] = seq;
}

extern "C" void scheduler_step(Scheduler* sched) {
    // 1. Queue management: Promote requests when space allows
    while (sched->num_pending > 0 && sched->num_active < sched->max_active) {
        Sequence* seq = sched->pending_seqs[0];
        int blocks_needed = (seq->prompt_len + PAGE_SIZE - 1) / PAGE_SIZE;
        
        if (sched->kv_cache->free_count >= blocks_needed) {
            for(int i = 0; i < blocks_needed; i++) {
                seq->logical_blocks[seq->num_blocks++] = paged_kv_allocate_block(sched->kv_cache);
            }
            seq->state = PREFILL;
            sched->active_seqs[sched->num_active++] = seq;
            sched->num_pending--;
            memmove(sched->pending_seqs, sched->pending_seqs + 1, sched->num_pending * sizeof(Sequence*));
        } else {
            break; 
        }
    }
    
    // 2. Compute Execution Loop (Continuous Interleaving)
    int* d_sampled_token;
    cudaMallocManaged(&d_sampled_token, sizeof(int));

    for(int i = 0; i < sched->num_active; i++) {
        Sequence* seq = sched->active_seqs[i];
        
        if (seq->state == PREFILL) {
            double t0 = get_time_ms();
            
            // Execute actual prefill tensor operations over prompt sequence
            fp16_forward(sched->weights, seq->tokens, seq->seq_len, seq->prompt_len, 
                         sched->kv_cache, seq->logical_blocks, seq->num_blocks, sched->d_logits);
            
            // Sample token 1
            argmax_kernel<<<1, 1>>>(sched->d_logits, sched->weights->vocab_size, d_sampled_token);
            cudaDeviceSynchronize();
            
            seq->prefill_time = get_time_ms() - t0;
            seq->state = DECODE;
            
            printf("[ENGINE] Req %d Prefill Complete: %d tokens processed in %.2f ms\n", 
                   seq->req_id, seq->prompt_len, seq->prefill_time);
                   
        } else if (seq->state == DECODE) {
            // Check dynamic boundary conditions before computing
            if (seq->seq_len % PAGE_SIZE == 0) {
                if (sched->kv_cache->free_count > 0) {
                    seq->logical_blocks[seq->num_blocks++] = paged_kv_allocate_block(sched->kv_cache);
                } else {
                    printf("[WARN] VRAM Out-of-Memory. Evicting Req %d Early.\n", seq->req_id);
                    seq->state = FINISHED;
                    continue;
                }
            }
            
            double t0 = get_time_ms();
            
            // Decode pass: Sequence length is evaluated step-by-step
            fp16_forward(sched->weights, seq->tokens, seq->seq_len, seq->prompt_len, 
                         sched->kv_cache, seq->logical_blocks, seq->num_blocks, sched->d_logits);
            
            argmax_kernel<<<1, 1>>>(sched->d_logits, sched->weights->vocab_size, d_sampled_token);
            cudaDeviceSynchronize();
            
            seq->total_decode_time += (get_time_ms() - t0);
            seq->tokens_generated++;
            
            // Append the actual evaluated token back to the sequence matrix
            seq->tokens[seq->seq_len++] = *d_sampled_token;
            
            if (seq->seq_len >= seq->max_tokens || *d_sampled_token == 2) { // 2 = Typical EOS Token ID
                seq->state = FINISHED;
            }
        }
    }
    cudaFree(d_sampled_token);
    
    // 3. Ejection and Detailed Performance Auditing
    for(int i = 0; i < sched->num_active; i++) {
        Sequence* seq = sched->active_seqs[i];
        if (seq->state == FINISHED) {
            double total_latency = get_time_ms() - seq->start_time;
            double tokens_per_sec = (seq->tokens_generated / (seq->total_decode_time / 1000.0));
            
            printf("\n============================= METRICS: REQ %d =============================\n", seq->req_id);
            printf("  -> Prefill Execution Latency  : %.2f ms\n", seq->prefill_time);
            printf("  -> Total Generation Time      : %.2f ms\n", seq->total_decode_time);
            printf("  -> Total Request Wall Latency : %.2f ms\n", total_latency);
            printf("  -> Tokens Generated           : %d\n", seq->tokens_generated);
            printf("  -> Sustained Generation Speed : **%.2f tokens/sec**\n", tokens_per_sec);
            printf("========================================================================\n\n");
            
            for(int b = 0; b < seq->num_blocks; b++) {
                paged_kv_free_block(sched->kv_cache, seq->logical_blocks[b]);
            }
            free(seq->tokens);
            free(seq->logical_blocks);
            free(seq);
            
            sched->num_active--;
            sched->active_seqs[i] = sched->active_seqs[sched->num_active];
            i--; 
        }
    }
}

extern "C" void scheduler_free(Scheduler* sched) {
    cudaFree(sched->d_logits);
    free(sched->active_seqs);
    free(sched->pending_seqs);
    free(sched);
}

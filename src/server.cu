#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include "../include/paged_kvcache.h"
#include "../include/scheduler.h"
#include "../include/weights.h"
#include "../include/gguf.h"

// Global Engine State
Scheduler* global_scheduler = NULL;
int request_counter = 0;

// Background thread to simulate incoming network requests
void* network_listener(void* arg) {
    while(1) {
        sleep(3); // New request arrives every 3 seconds
        
        int dummy_prompt[15] = {1, 29871, 13, 13}; 
        int prompt_len = 4;
        int max_gen = 25;
        
        request_counter++;
        printf("\n[NETWORK] Received Request ID: %d\n", request_counter);
        
        if (global_scheduler) {
            scheduler_add_request(global_scheduler, request_counter, dummy_prompt, prompt_len, max_gen);
        }
    }
    return NULL;
}

int main(int argc, char** argv) {
    printf("====================================================\n");
    printf("  Custom LLM Inference Engine - Server Init\n");
    printf("====================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[INIT] GPU: %s\n", prop.name);

    // 1. Initialize Paged KV Cache
    int n_blocks = 2048; 
    int n_layers = 22;
    int n_kv_heads = 4;
    int head_dim = 64;
    
    printf("[INIT] Allocating Paged KV Cache...\n");
    PagedKVCache* kv_cache = paged_kv_create(n_blocks, n_layers, n_kv_heads, head_dim, 64, 2048);

    // 2. Load GGUF Weights FIRST so we can pass them to the scheduler
    const char* model_path = argc > 1 ? argv[1] : "../models/tinyllama.gguf";
    printf("[INIT] Loading Weights from: %s\n", model_path);
    ModelWeights* weights = load_gguf_weights(model_path, n_layers, 2048, 32, head_dim, 5632, 32000);

    // 3. Initialize Continuous Batching Scheduler with weights
    printf("[INIT] Booting Scheduler...\n");
    global_scheduler = scheduler_create(kv_cache, weights, 8); 
    
    // 4. Spin up mock network listener thread
    pthread_t net_thread;
    pthread_create(&net_thread, NULL, network_listener, NULL);

    printf("\n[SERVER] Engine online. Entering main execution loop...\n");
    
    // 5. Main Inference Loop (The Engine Tick)
    while(1) {
        if (global_scheduler && (global_scheduler->num_active > 0 || global_scheduler->num_pending > 0)) {
            scheduler_step(global_scheduler);
            usleep(100000); 
        } else {
            usleep(500000);
        }
    }

    return 0;
}

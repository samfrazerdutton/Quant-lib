#include <stdio.h>
#include <stdlib.h>
#include "../include/paged_kvcache.h"
#include "../include/scheduler.h"

int main() {
    printf("=== Testing Continuous Batching Memory Scheduler ===\n");
    
    // Micro-cache: Only 10 blocks of 16 tokens = 160 tokens maximum VRAM capacity
    PagedKVCache* kv = paged_kv_create(10, 22, 4, 64, 16, 128);
    Scheduler* sched = scheduler_create(kv, 4);
    
    int dummy_prompt[60] = {0}; // Fake prompt data
    
    printf("\nInjecting Concurrent Requests...\n");
    scheduler_add_request(sched, 1, dummy_prompt, 30, 20); // Needs 2 blocks to start
    scheduler_add_request(sched, 2, dummy_prompt, 50, 15); // Needs 4 blocks to start
    scheduler_add_request(sched, 3, dummy_prompt, 60, 10); // Needs 4 blocks to start
    scheduler_add_request(sched, 4, dummy_prompt, 20, 10); // Queue should block here initially
    
    int step = 1;
    while(sched->num_active > 0 || sched->num_pending > 0) {
        printf("\n--- Engine Tick %d ---\n", step++);
        scheduler_step(sched);
        if(step > 25) break; // infinite loop protection
    }
    
    printf("\n=== Batching Simulation Complete ===\n");
    return 0;
}

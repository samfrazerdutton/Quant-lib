#include <cuda_runtime.h>
#include <cublas_v2.h>
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

// Forward declarations from forward.cu
struct ForwardState;
extern ForwardState* forward_state_create(int, int, int, int);
extern void          forward_state_free(ForwardState*);
extern float*        model_forward(cublasHandle_t, ModelWeights*,
                                   KVCache*, ForwardState*,
                                   int, cudaStream_t);

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s\n\n", prop.name);

    // Tiny model config — fits easily in 6GB
    int n_layers   = 4;
    int n_heads    = 8;
    int hidden_dim = 512;
    int ffn_dim    = 1408;
    int vocab_size = 32000;
    int max_seq    = 128;

    printf("=== Tiny Transformer Forward Pass ===\n");
    printf("layers=%d  heads=%d  hidden=%d  ffn=%d  vocab=%d\n\n",
           n_layers, n_heads, hidden_dim, ffn_dim, vocab_size);

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate model
    ModelWeights* weights = weights_alloc_random(
        n_layers, hidden_dim, n_heads, ffn_dim, vocab_size);

    KVCache* cache = kvcache_create(
        n_layers, max_seq, n_heads, hidden_dim / n_heads);

    ForwardState* state = forward_state_create(
        1, hidden_dim, ffn_dim, vocab_size);

    // ── Single token forward pass ─────────────────────────────────────────
    printf("--- Single token forward pass ---\n");
    int test_token = 42;
    float* logits = model_forward(cublas, weights, cache,
                                  state, test_token, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Copy logits back and find argmax (greedy next token)
    float* h_logits = (float*)malloc(vocab_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, logits,
                          vocab_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_logit = -1e30f;
    int   next_token = 0;
    float sum_exp = 0.0f;
    float mx = h_logits[0];
    for (int i = 1; i < vocab_size; i++) if (h_logits[i] > mx) mx = h_logits[i];
    for (int i = 0; i < vocab_size; i++) sum_exp += expf(h_logits[i] - mx);

    for (int i = 0; i < vocab_size; i++) {
        if (h_logits[i] > max_logit) {
            max_logit  = h_logits[i];
            next_token = i;
        }
    }

    printf("  Input token  : %d\n", test_token);
    printf("  Next token   : %d  (greedy argmax)\n", next_token);
    printf("  Max logit    : %.4f\n", max_logit);
    printf("  Logit range  : [%.4f, %.4f]\n", h_logits[0], h_logits[vocab_size-1]);
    printf("  KV cache len : %d\n\n", cache->current_len);

    // ── Autoregressive generation: 16 tokens ─────────────────────────────
    printf("--- Autoregressive generation (16 tokens) ---\n");

    // Reset cache
    kvcache_free(cache);
    cache = kvcache_create(n_layers, max_seq, n_heads, hidden_dim / n_heads);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    int token = 1;   // BOS token
    printf("  Tokens: %d", token);

    CUDA_CHECK(cudaEventRecord(t0, stream));

    for (int step = 0; step < 16; step++) {
        float* lg = model_forward(cublas, weights, cache,
                                  state, token, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpy(h_logits, lg,
                              vocab_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // Greedy decode
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (h_logits[i] > h_logits[best]) best = i;
        token = best;
        printf(" -> %d", token);
        fflush(stdout);
    }

    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("\n\n  16 tokens generated in %.2f ms\n", ms);
    printf("  Per-token latency     : %.2f ms\n", ms / 16.0f);
    printf("  Tokens per second     : %.1f tok/s\n\n", 16000.0f / ms);

    // ── Memory report ─────────────────────────────────────────────────────
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU memory used : %.0f MB / %.0f MB\n\n",
           (total_mem - free_mem) / 1e6, total_mem / 1e6);

    printf("=== Forward pass complete ===\n");
    printf("You now have a working transformer inference engine.\n");
    printf("Next: load real Llama-3.2-1B weights from GGUF.\n");

    forward_state_free(state);
    kvcache_free(cache);
    weights_free(weights);
    cublasDestroy(cublas);
    free(h_logits);
    return 0;
}

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

extern ModelWeights* weights_load_bin(const char*, int, int, int, int, int);
struct ForwardState;
extern ForwardState* forward_state_create(int, int, int, int);
extern void          forward_state_free(ForwardState*);
extern float*        model_forward(cublasHandle_t, ModelWeights*,
                                   KVCache*, ForwardState*, int, cudaStream_t);

static int argmax(const float* v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

int main(int argc, char** argv) {
    char default_path[512];
    snprintf(default_path, sizeof(default_path), "%s/tinyllama/weights.bin",
             getenv("HOME") ? getenv("HOME") : "/root");
    const char* path = (argc > 1) ? argv[1] : default_path;

    // TinyLlama-1.1B exact config
    int n_layers   = 22;
    int n_heads    = 32;
    int hidden_dim = 2048;
    int ffn_dim    = 5632;
    int vocab_size = 32000;
    int max_seq    = 256;

    printf("=== TinyLlama-1.1B — Custom CUDA Inference Engine ===\n");
    printf("weights: %s\n\n", path);

    ModelWeights* weights = weights_load_bin(
        path, n_layers, hidden_dim, n_heads, ffn_dim, vocab_size);
    if (!weights) return 1;

    // kv_dim from actual loaded weights (256 for TinyLlama GQA)
    int kv_dim   = weights->kv_dim;
    int head_dim = hidden_dim / n_heads;   // 64
    int n_kv_heads = kv_dim / head_dim;   // 4

    printf("Architecture: n_heads=%d  n_kv_heads=%d  head_dim=%d\n\n",
           n_heads, n_kv_heads, head_dim);

    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // KV cache sized for KV heads only (memory efficient)
    KVCache* cache = kvcache_create(n_layers, max_seq,
                                    n_kv_heads, head_dim);
    struct ForwardState* state = forward_state_create(
        1, hidden_dim, ffn_dim, vocab_size);

    float* h_logits = (float*)malloc(vocab_size * sizeof(float));

    size_t fm, tm;
    cudaMemGetInfo(&fm, &tm);
    printf("GPU: %.0f MB used / %.0f MB total\n\n", (tm-fm)/1e6, tm/1e6);

    // BOS token = 1 for TinyLlama (Llama tokenizer)
    int prompt[] = {1, 1724, 338, 7431};  // <s> What is Paris
    int prompt_len = 4;
    int gen_tokens = 32;

    printf("Prefill (%d tokens)...\n", prompt_len);
    for (int i = 0; i < prompt_len; i++) {
        model_forward(cublas, weights, cache, state, prompt[i], stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
    }

    printf("Generating %d tokens (greedy)...\n", gen_tokens);
    printf("Output token IDs: ");
    for (int i = 0; i < prompt_len; i++) printf("%d ", prompt[i]);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));

    int token = prompt[prompt_len - 1];
    for (int step = 0; step < gen_tokens; step++) {
        float* logits = model_forward(cublas, weights, cache,
                                      state, token, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_logits, logits,
                              vocab_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
        token = argmax(h_logits, vocab_size);
        printf("%d ", token);
        fflush(stdout);
    }

    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    printf("\n\n=== Results ===\n");
    printf("  %d tokens in %.1f ms\n", gen_tokens, ms);
    printf("  %.2f ms / token\n", ms / gen_tokens);
    printf("  %.1f tok/s\n\n", gen_tokens * 1000.0f / ms);
    printf("Next step: decode token IDs with sentencepiece tokenizer.\n");

    forward_state_free(state);
    kvcache_free(cache);
    weights_free(weights);
    cublasDestroy(cublas);
    free(h_logits);
    return 0;
}

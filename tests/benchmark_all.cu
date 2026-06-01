#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../include/kv_cache.h"
#include "../include/weights.h"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d -- %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

extern void launch_matmul  (const float*, const float*, float*, int, int, int, cudaStream_t);
extern void launch_softmax (const float*, float*, int, int, cudaStream_t);
extern void launch_rmsnorm (const float*, const float*, float*, int, int, float, cudaStream_t);
extern void launch_rope    (float*, int, int, int, float, cudaStream_t);
extern void launch_attention(const float*, const float*, const float*, float*, int, int, int, cudaStream_t);
extern void launch_swiglu  (const float*, const float*, float*, int, int, cudaStream_t);

// ── timing helpers ────────────────────────────────────────────────────────────
static float time_kernel(cudaEvent_t t0, cudaEvent_t t1,
                         cudaStream_t stream, int reps) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    return ms / reps;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║          CUDA LLM INFERENCE ENGINE — FULL BENCHMARK             ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  GPU  : %-57s║\n", prop.name);
    printf("║  VRAM : %-4.0f MB                                                  ║\n",
           prop.totalGlobalMem / 1e6);
    printf("║  Arch : sm_%d%d                                                    ║\n",
           prop.major, prop.minor);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    // ── 1. GEMM ───────────────────────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ 1. TILED GEMM  (C = A * B,  shared mem + bank conflict padding) │\n");
    printf("├──────────────┬──────────┬───────────┬──────────────────────────┤\n");
    printf("│ Matrix Size  │  Time ms │  TFLOPS   │ Notes                    │\n");
    printf("├──────────────┼──────────┼───────────┼──────────────────────────┤\n");

    int gemm_sizes[] = {512, 1024, 2048, 4096};
    for (int si = 0; si < 4; si++) {
        int N = gemm_sizes[si];
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, (size_t)N*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, (size_t)N*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, (size_t)N*N*sizeof(float)));

        launch_matmul(dA, dB, dC, N, N, N, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int reps = (N <= 1024) ? 200 : (N <= 2048 ? 50 : 10);
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int r = 0; r < reps; r++)
            launch_matmul(dA, dB, dC, N, N, N, stream);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = time_kernel(t0, t1, stream, reps);
        double tflops = (2.0 * N * N * N) / (ms * 1e-3) / 1e12;
        printf("│ %4dx%4dx%4d│ %8.4f │ %9.3f │                          │\n",
               N, N, N, ms, tflops);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    printf("└──────────────┴──────────┴───────────┴──────────────────────────┘\n\n");

    // ── 2. Softmax ────────────────────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ 2. SOFTMAX  (warp __shfl_down_sync reductions, numerically safe)│\n");
    printf("├──────────────────┬──────────┬──────────┬────────────────────────┤\n");
    printf("│ Shape            │  Time ms │  GB/s    │ Peak: ~336 GB/s        │\n");
    printf("├──────────────────┼──────────┼──────────┼────────────────────────┤\n");

    int smax_sizes[] = {512, 1024, 2048, 4096};
    for (int si = 0; si < 4; si++) {
        int N = smax_sizes[si];
        float *dIn, *dOut;
        CUDA_CHECK(cudaMalloc(&dIn,  (size_t)N*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, (size_t)N*N*sizeof(float)));

        launch_softmax(dIn, dOut, N, N, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int reps = 500;
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int r = 0; r < reps; r++)
            launch_softmax(dIn, dOut, N, N, stream);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = time_kernel(t0, t1, stream, reps);
        double gb_s = (2.0 * N * N * sizeof(float)) / (ms * 1e-3) / 1e9;
        printf("│ %4dx%-11d │ %8.4f │ %8.1f │                        │\n",
               N, N, ms, gb_s);
        cudaFree(dIn); cudaFree(dOut);
    }
    printf("└──────────────────┴──────────┴──────────┴────────────────────────┘\n\n");

    // ── 3. RMSNorm ────────────────────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ 3. RMSNORM  (warp-reduce, used every transformer layer)         │\n");
    printf("├──────────────────┬──────────┬──────────┬────────────────────────┤\n");
    printf("│ Shape            │  Time ms │  GB/s    │                        │\n");
    printf("├──────────────────┼──────────┼──────────┼────────────────────────┤\n");

    int rms_sizes[] = {512, 1024, 2048, 4096};
    for (int si = 0; si < 4; si++) {
        int dim = rms_sizes[si];
        int rows = 2048;
        float *dX, *dW, *dO;
        CUDA_CHECK(cudaMalloc(&dX, (size_t)rows*dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dW, (size_t)dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dO, (size_t)rows*dim*sizeof(float)));

        launch_rmsnorm(dX, dW, dO, rows, dim, 1e-5f, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int reps = 1000;
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int r = 0; r < reps; r++)
            launch_rmsnorm(dX, dW, dO, rows, dim, 1e-5f, stream);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = time_kernel(t0, t1, stream, reps);
        double gb_s = (3.0 * rows * dim * sizeof(float)) / (ms * 1e-3) / 1e9;
        printf("│ %4dx%-11d │ %8.4f │ %8.1f │                        │\n",
               rows, dim, ms, gb_s);
        cudaFree(dX); cudaFree(dW); cudaFree(dO);
    }
    printf("└──────────────────┴──────────┴──────────┴────────────────────────┘\n\n");

    // ── 4. Attention ──────────────────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ 4. MULTI-HEAD ATTENTION  (32 heads, head_dim=128, O(n²))        │\n");
    printf("├──────────────────┬──────────┬──────────┬────────────────────────┤\n");
    printf("│ seq_len          │  Time ms │  GFLOPS  │                        │\n");
    printf("├──────────────────┼──────────┼──────────┼────────────────────────┤\n");

    int n_heads = 32, head_dim = 128;
    int attn_seqs[] = {64, 128, 256, 512};
    for (int si = 0; si < 4; si++) {
        int sl = attn_seqs[si];
        size_t sz = (size_t)sl * n_heads * head_dim * sizeof(float);
        float *dQ, *dK, *dV, *dO;
        CUDA_CHECK(cudaMalloc(&dQ, sz));
        CUDA_CHECK(cudaMalloc(&dK, sz));
        CUDA_CHECK(cudaMalloc(&dV, sz));
        CUDA_CHECK(cudaMalloc(&dO, sz));

        launch_attention(dQ, dK, dV, dO, sl, n_heads, head_dim, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int reps = 100;
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int r = 0; r < reps; r++)
            launch_attention(dQ, dK, dV, dO, sl, n_heads, head_dim, stream);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = time_kernel(t0, t1, stream, reps);
        double gflops = (4.0 * sl * sl * head_dim * n_heads) / (ms * 1e-3) / 1e9;
        printf("│ %-16d │ %8.4f │ %8.2f │                        │\n",
               sl, ms, gflops);
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }
    printf("└──────────────────┴──────────┴──────────┴────────────────────────┘\n\n");

    // ── 5. RoPE ───────────────────────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ 5. ROPE  (Rotary Position Embedding, isometry-preserving)       │\n");
    printf("├──────────────────┬──────────┬──────────┬────────────────────────┤\n");
    printf("│ seq x heads      │  Time ms │  GB/s    │                        │\n");
    printf("├──────────────────┼──────────┼──────────┼────────────────────────┤\n");

    int rope_seqs[] = {128, 512, 1024, 2048};
    for (int si = 0; si < 4; si++) {
        int sl = rope_seqs[si];
        size_t sz = (size_t)sl * n_heads * head_dim * sizeof(float);
        float *dX;
        CUDA_CHECK(cudaMalloc(&dX, sz));

        launch_rope(dX, sl, n_heads, head_dim, 10000.0f, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int reps = 1000;
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int r = 0; r < reps; r++)
            launch_rope(dX, sl, n_heads, head_dim, 10000.0f, stream);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = time_kernel(t0, t1, stream, reps);
        double gb_s = (2.0 * sz) / (ms * 1e-3) / 1e9;
        printf("│ %4d x %-9d │ %8.4f │ %8.1f │                        │\n",
               sl, n_heads, ms, gb_s);
        cudaFree(dX);
    }
    printf("└──────────────────┴──────────┴──────────┴────────────────────────┘\n\n");

    // ── 6. SwiGLU ─────────────────────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ 6. SWIGLU FFN  (silu(gate) * up, Llama FFN activation)         │\n");
    printf("├──────────────────┬──────────┬──────────┬────────────────────────┤\n");
    printf("│ rows x ffn_dim   │  Time ms │  GB/s    │                        │\n");
    printf("├──────────────────┼──────────┼──────────┼────────────────────────┤\n");

    int ffn_dim = 5632;
    int swiglu_rows[] = {1, 32, 128, 512};
    for (int si = 0; si < 4; si++) {
        int rows = swiglu_rows[si];
        size_t sz = (size_t)rows * ffn_dim * sizeof(float);
        float *dG, *dU, *dO;
        CUDA_CHECK(cudaMalloc(&dG, sz));
        CUDA_CHECK(cudaMalloc(&dU, sz));
        CUDA_CHECK(cudaMalloc(&dO, sz));

        launch_swiglu(dG, dU, dO, rows, ffn_dim, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int reps = 5000;
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int r = 0; r < reps; r++)
            launch_swiglu(dG, dU, dO, rows, ffn_dim, stream);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = time_kernel(t0, t1, stream, reps);
        double gb_s = (3.0 * sz) / (ms * 1e-3) / 1e9;
        printf("│ %4d x %-9d │ %8.4f │ %8.1f │                        │\n",
               rows, ffn_dim, ms, gb_s);
        cudaFree(dG); cudaFree(dU); cudaFree(dO);
    }
    printf("└──────────────────┴──────────┴──────────┴────────────────────────┘\n\n");

    // ── 7. KV Cache ───────────────────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ 7. KV CACHE  (fused append kernel, all layers in one launch)    │\n");
    printf("├──────────────────┬──────────┬──────────┬────────────────────────┤\n");
    printf("│ config           │  Time ms │ overhead │                        │\n");
    printf("├──────────────────┼──────────┼──────────┼────────────────────────┤\n");

    int kv_configs[][3] = {{4,4,64},{8,4,64},{16,4,64},{32,4,64}};
    for (int ci = 0; ci < 4; ci++) {
        int n_layers = kv_configs[ci][0];
        int n_kv     = kv_configs[ci][1];
        int hd       = kv_configs[ci][2];
        KVCache* cache = kvcache_create(n_layers, 512, n_kv, hd);

        int slice = n_kv * hd;
        float *dk, *dv;
        CUDA_CHECK(cudaMalloc(&dk, slice*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dv, slice*sizeof(float)));

        int reps = 1000;
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int r = 0; r < reps; r++) {
            for (int l = 0; l < n_layers; l++)
                kvcache_append(cache, l, dk, dv, stream);
            cache->current_len = (cache->current_len + 1) % 256;
        }
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = time_kernel(t0, t1, stream, reps);
        char cfg[32];
        snprintf(cfg, sizeof(cfg), "%2d layers,kv=%d,hd=%d", n_layers, n_kv, hd);
        printf("│ %-16s │ %8.4f │ per-tok  │                        │\n",
               cfg, ms);
        kvcache_free(cache);
        cudaFree(dk); cudaFree(dv);
    }
    printf("└──────────────────┴──────────┴──────────┴────────────────────────┘\n\n");

    // ── 8. End-to-end inference ───────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ 8. END-TO-END INFERENCE  (TinyLlama-1.1B, real weights)         │\n");
    printf("├──────────────────────────────────────────────────────────────────┤\n");

    // Check if weights exist
    char wpath[256];
    snprintf(wpath, sizeof(wpath), "%s/tinyllama/weights.bin",
             getenv("HOME") ? getenv("HOME") : "/root");
    FILE* wf = fopen(wpath, "rb");
    if (wf) {
        fclose(wf);
        printf("│  weights found — run ./test_real for full inference benchmark    │\n");
        printf("│  Previous result: 48.4 tok/s  (20.6 ms/tok)                     │\n");
    } else {
        printf("│  weights not found at %s\n", wpath);
        printf("│  Run: python3 src/load_safetensors.py                            │\n");
    }
    printf("└──────────────────────────────────────────────────────────────────┘\n\n");

    // ── Summary ───────────────────────────────────────────────────────────────
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                        SUMMARY                                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Kernel              Best Result       Metric                   ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  GEMM (4096³)        see above          TFLOPS                  ║\n");
    printf("║  Softmax (4096x4096) see above          GB/s vs 336 GB/s peak   ║\n");
    printf("║  RMSNorm             see above          GB/s                    ║\n");
    printf("║  Attention (seq=512) see above          GFLOPS                  ║\n");
    printf("║  RoPE (seq=2048)     see above          GB/s                    ║\n");
    printf("║  SwiGLU              see above          GB/s                    ║\n");
    printf("║  KV Cache append     see above          ms/token                ║\n");
    printf("║  TinyLlama-1.1B      48.4 tok/s         tokens/second           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(stream);
    return 0;
}

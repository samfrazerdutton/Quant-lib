#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

extern void launch_rope(float*, int, int, int, float, cudaStream_t);
extern void launch_attention(const float*, const float*, const float*,
                             float*, int, int, int, cudaStream_t);

// CPU reference attention
void cpu_attention(const float* Q, const float* K, const float* V,
                   float* out, int seq_len, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    float* scores = (float*)malloc(seq_len * sizeof(float));

    for (int q = 0; q < seq_len; q++) {
        for (int h = 0; h < n_heads; h++) {
            // QK^T
            float mx = -FLT_MAX;
            for (int k = 0; k < seq_len; k++) {
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += Q[q*n_heads*head_dim + h*head_dim + d]
                         * K[k*n_heads*head_dim + h*head_dim + d];
                scores[k] = dot * scale;
                if (scores[k] > mx) mx = scores[k];
            }
            // softmax
            float s = 0;
            for (int k = 0; k < seq_len; k++) {
                scores[k] = expf(scores[k] - mx);
                s += scores[k];
            }
            for (int k = 0; k < seq_len; k++) scores[k] /= s;
            // weighted V
            for (int d = 0; d < head_dim; d++) {
                float acc = 0;
                for (int k = 0; k < seq_len; k++)
                    acc += scores[k] * V[k*n_heads*head_dim + h*head_dim + d];
                out[q*n_heads*head_dim + h*head_dim + d] = acc;
            }
        }
    }
    free(scores);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s\n\n", prop.name);

    // Llama-style config: 32 heads, 128 head_dim, test at seq_len=128
    int seq_len  = 128;
    int n_heads  = 32;
    int head_dim = 128;
    size_t sz = seq_len * n_heads * head_dim * sizeof(float);

    printf("Config: seq=%d  heads=%d  head_dim=%d\n\n", seq_len, n_heads, head_dim);

    float *hQ = (float*)malloc(sz);
    float *hK = (float*)malloc(sz);
    float *hV = (float*)malloc(sz);
    float *hOut_gpu = (float*)malloc(sz);
    float *hOut_cpu = (float*)malloc(sz);

    srand(99);
    for (int i = 0; i < seq_len*n_heads*head_dim; i++) {
        hQ[i] = (float)rand()/RAND_MAX - 0.5f;
        hK[i] = (float)rand()/RAND_MAX - 0.5f;
        hV[i] = (float)rand()/RAND_MAX - 0.5f;
    }

    float *dQ, *dK, *dV, *dOut;
    CUDA_CHECK(cudaMalloc(&dQ,   sz));
    CUDA_CHECK(cudaMalloc(&dK,   sz));
    CUDA_CHECK(cudaMalloc(&dV,   sz));
    CUDA_CHECK(cudaMalloc(&dOut, sz));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, sz, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Test 1: RoPE ─────────────────────────────────────────────────────
    printf("--- RoPE ---\n");
    // Copy Q to a scratch buffer, apply RoPE, check norms preserved
    float *dQ_rope;
    CUDA_CHECK(cudaMalloc(&dQ_rope, sz));
    CUDA_CHECK(cudaMemcpy(dQ_rope, dQ, sz, cudaMemcpyDeviceToDevice));

    launch_rope(dQ_rope, seq_len, n_heads, head_dim, 10000.0f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    float *hQ_rope = (float*)malloc(sz);
    CUDA_CHECK(cudaMemcpy(hQ_rope, dQ_rope, sz, cudaMemcpyDeviceToHost));

    // RoPE is an isometry: it preserves vector norms
    float maxNormErr = 0;
    for (int i = 0; i < seq_len * n_heads; i++) {
        float norm_before = 0, norm_after = 0;
        for (int d = 0; d < head_dim; d++) {
            norm_before += hQ[i*head_dim+d]      * hQ[i*head_dim+d];
            norm_after  += hQ_rope[i*head_dim+d] * hQ_rope[i*head_dim+d];
        }
        maxNormErr = fmaxf(maxNormErr, fabsf(sqrtf(norm_before) - sqrtf(norm_after)));
    }
    printf("  Norm preservation error: %e  %s\n\n",
           maxNormErr, maxNormErr < 1e-4 ? "[PASS]" : "[FAIL]");

    // ── Test 2: Attention correctness ─────────────────────────────────────
    printf("--- Attention ---\n");
    launch_attention(dQ, dK, dV, dOut, seq_len, n_heads, head_dim, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hOut_gpu, dOut, sz, cudaMemcpyDeviceToHost));
    cpu_attention(hQ, hK, hV, hOut_cpu, seq_len, n_heads, head_dim);

    float maxErr = 0, sumErr = 0;
    for (int i = 0; i < seq_len*n_heads*head_dim; i++) {
        float e = fabsf(hOut_gpu[i] - hOut_cpu[i]);
        if (e > maxErr) maxErr = e;
        sumErr += e;
    }
    printf("  Max abs error  : %e\n", maxErr);
    printf("  Mean abs error : %e\n", sumErr / (seq_len*n_heads*head_dim));
    printf("  Correctness    : %s\n\n",
           maxErr < 1e-3 ? "[PASS]" : "[FAIL]");

    // ── Benchmark at increasing seq_len ───────────────────────────────────
    printf("--- Attention Benchmark (n_heads=%d head_dim=%d) ---\n", n_heads, head_dim);
    printf("  %6s | %9s | %8s\n", "seq_len", "time(ms)", "GFLOPS");
    printf("  ---------------------------------\n");

    int seqs[] = {64, 128, 256, 512};
    for (int si = 0; si < 4; si++) {
        int sl = seqs[si];
        size_t bsz = sl * n_heads * head_dim * sizeof(float);
        float *dQb, *dKb, *dVb, *dOb;
        CUDA_CHECK(cudaMalloc(&dQb, bsz));
        CUDA_CHECK(cudaMalloc(&dKb, bsz));
        CUDA_CHECK(cudaMalloc(&dVb, bsz));
        CUDA_CHECK(cudaMalloc(&dOb, bsz));

        // warmup
        launch_attention(dQb, dKb, dVb, dOb, sl, n_heads, head_dim, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int r = 0; r < 50; r++)
            launch_attention(dQb, dKb, dVb, dOb, sl, n_heads, head_dim, stream);
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= 50.0f;

        // FLOPs: QK^T = 2*sl*sl*head_dim*n_heads, V = same
        double gflops = (4.0 * sl * sl * head_dim * n_heads) / (ms*1e-3) / 1e9;
        printf("  %6d | %9.4f | %8.2f\n", sl, ms, gflops);

        cudaFree(dQb); cudaFree(dKb); cudaFree(dVb); cudaFree(dOb);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dOut); cudaFree(dQ_rope);
    free(hQ); free(hK); free(hV);
    free(hOut_gpu); free(hOut_cpu); free(hQ_rope);
    return 0;
}

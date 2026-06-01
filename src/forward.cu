#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
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

#define CUBLAS_CHECK(call)                                                \
    do {                                                                  \
        cublasStatus_t s = (call);                                        \
        if (s != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d -- %d\n",             \
                    __FILE__, __LINE__, s);                               \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

extern void launch_rmsnorm(const float*, const float*, float*,
                           int, int, float, cudaStream_t);
extern void launch_rope(float*, int, int, int, float, cudaStream_t);
extern void launch_attention(const float*, const float*, const float*,
                             float*, int, int, int, cudaStream_t);
extern void launch_swiglu(const float*, const float*, float*,
                          int, int, cudaStream_t);

// C = A * B^T  row-major A[M,K] B[N,K] -> C[M,N]
static void gemm(cublasHandle_t handle,
                 const float* A, const float* B, float* C,
                 int M, int N, int K, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        &alpha, B, K, A, K, &beta, C, N));
}

// Forward declarations for quantized path
extern void qgemm(const __half*, const QTensor*, __half*, int, int, int, cudaStream_t);
extern void fp32_to_fp16_kernel(const float*, __half*, int) __attribute__((global));
extern void fp16_to_fp32_kernel(const __half*, float*, int) __attribute__((global));

// Smart gemm: uses fp32 cublas if weights loaded, fp8 quantlib if quantized
static void sgemm(cublasHandle_t handle,
                  const float* A, const Tensor* W, const QTensor* QW,
                  float* C, int M, int N, int K,
                  float* fp16_buf_a, float* fp16_buf_c,
                  cudaStream_t stream) {
    if (W->data != nullptr) {
        gemm(handle, A, W->data, C, M, N, K, stream);
    } else {
        int threads = 256;
        __half* hA = (__half*)fp16_buf_a;
        __half* hC = (__half*)fp16_buf_c;
        dim3 blk((M*K+255)/256);
        fp32_to_fp16_kernel<<<blk, threads, 0, stream>>>(A, hA, M*K);
        qgemm(hA, QW, hC, M, N, K, stream);
        dim3 blk2((M*N+255)/256);
        fp16_to_fp32_kernel<<<blk2, threads, 0, stream>>>(hC, C, M*N);
    }
}

typedef struct {
    float* x;
    float* x_norm;
    float* q;
    float* k;
    float* v;
    float* attn_out;
    float* gate;
    float* up;
    float* ffn_mid;
    float* ffn_out;
    float* logits;
    int    hidden_dim;
    int    ffn_dim;
    int    kv_dim;
    __half* fp16_a;  // scratch: fp32->fp16 input
    __half* fp16_c;  // scratch: fp16->fp32 output
} ForwardState;

ForwardState* forward_state_create(int seq_len, int hidden_dim,
                                   int ffn_dim, int vocab_size) {
    ForwardState* s = (ForwardState*)calloc(1, sizeof(ForwardState));
    s->hidden_dim = hidden_dim;
    s->ffn_dim    = ffn_dim;
    s->kv_dim     = hidden_dim;  // updated after weight load
    (void)seq_len;

    size_t hd = hidden_dim * sizeof(float);
    size_t fd = ffn_dim    * sizeof(float);

    CUDA_CHECK(cudaMalloc(&s->x,        hd));
    CUDA_CHECK(cudaMalloc(&s->x_norm,   hd));
    CUDA_CHECK(cudaMalloc(&s->q,        hd));
    CUDA_CHECK(cudaMalloc(&s->k,        hd));  // max possible
    CUDA_CHECK(cudaMalloc(&s->v,        hd));
    CUDA_CHECK(cudaMalloc(&s->attn_out, hd));
    CUDA_CHECK(cudaMalloc(&s->gate,     fd));
    CUDA_CHECK(cudaMalloc(&s->up,       fd));
    CUDA_CHECK(cudaMalloc(&s->ffn_mid,  fd));
    CUDA_CHECK(cudaMalloc(&s->ffn_out,  hd));
    CUDA_CHECK(cudaMalloc(&s->logits,   vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->fp16_a, hidden_dim * hidden_dim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&s->fp16_c, hidden_dim * hidden_dim * sizeof(__half)));
    return s;
}

void forward_state_free(ForwardState* s) {
    cudaFree(s->x);    cudaFree(s->x_norm);
    cudaFree(s->q);    cudaFree(s->k);   cudaFree(s->v);
    cudaFree(s->attn_out);
    cudaFree(s->gate); cudaFree(s->up);
    cudaFree(s->ffn_mid); cudaFree(s->ffn_out);
    cudaFree(s->logits);
    free(s);
}

__global__ void add_residual_kernel(float* x, const float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += d[i];
}
static void add_residual(float* x, const float* d, int n,
                         cudaStream_t stream) {
    add_residual_kernel<<<(n+255)/256, 256, 0, stream>>>(x, d, n);
}

__global__ void embed_kernel(float* out, const float* tbl,
                             int tok, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hd) out[i] = tbl[(size_t)tok * hd + i];
}

// GQA repeat-KV kernel: expand KV from n_kv_heads to n_heads
// Each KV head is shared by (n_heads / n_kv_heads) query heads
__global__ void repeat_kv_kernel(
    const float* kv_in,   // [seq_len, n_kv_heads, head_dim]
    float* kv_out,        // [seq_len, n_heads, head_dim]
    int seq_len, int n_kv_heads, int n_heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_heads * head_dim;
    if (idx >= total) return;

    int d        = idx % head_dim;
    int h        = (idx / head_dim) % n_heads;
    int s        = idx / (head_dim * n_heads);
    int kv_head  = h / (n_heads / n_kv_heads);

    kv_out[idx] = kv_in[s * n_kv_heads * head_dim + kv_head * head_dim + d];
}

static void forward_layer(
    cublasHandle_t handle,
    ForwardState*  s,
    LayerWeights*  lw,
    KVCache*       cache,
    int            layer_idx,
    int            n_heads,
    int            kv_dim,      // actual rows of wk/wv
    int            pos,
    cudaStream_t   stream)
{
    int hd        = s->hidden_dim;
    int fd        = s->ffn_dim;
    int head_dim  = hd / n_heads;
    int n_kv_heads = kv_dim / head_dim;
    int full_seq  = pos + 1;

    // 1. Pre-attn RMSNorm
    launch_rmsnorm(s->x, lw->rms_attn.data, s->x_norm,
                   1, hd, 1e-5f, stream);

    // 2. QKV projections
    // Q: [1, hd] x [hd, hd]^T = [1, hd]
    sgemm(handle, s->x_norm, &lw->wq, &lw->qwq, s->q, 1, hd, hd, (float*)s->fp16_a, (float*)s->fp16_c, stream);
    // K: [1, hd] x [kv_dim, hd]^T = [1, kv_dim]
    sgemm(handle, s->x_norm, &lw->wk, &lw->qwk, s->k, 1, kv_dim, hd, (float*)s->fp16_a, (float*)s->fp16_c, stream);
    // V: [1, hd] x [kv_dim, hd]^T = [1, kv_dim]
    sgemm(handle, s->x_norm, &lw->wv, &lw->qwv, s->v, 1, kv_dim, hd, (float*)s->fp16_a, (float*)s->fp16_c, stream);

    // 3. RoPE on Q (all heads) and K (kv_heads only)
    launch_rope(s->q, 1, n_heads,    head_dim, 10000.0f, stream);
    launch_rope(s->k, 1, n_kv_heads, head_dim, 10000.0f, stream);

    // 4. Append this token's K,V to cache
    kvcache_append(cache, layer_idx, s->k, s->v, stream);

    // 5. Get full K,V history and expand for GQA if needed
    float* raw_k = kvcache_get_k(cache, layer_idx);
    float* raw_v = kvcache_get_v(cache, layer_idx);

    float* full_k = raw_k;
    float* full_v = raw_v;

    // If GQA: expand [full_seq, n_kv_heads, head_dim] -> [full_seq, n_heads, head_dim]
    if (n_kv_heads < n_heads) {
        int total_k = full_seq * n_heads * head_dim;
        int total_v = full_seq * n_heads * head_dim;
        CUDA_CHECK(cudaMalloc(&full_k, total_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&full_v, total_v * sizeof(float)));
        repeat_kv_kernel<<<(total_k+255)/256, 256, 0, stream>>>(
            raw_k, full_k, full_seq, n_kv_heads, n_heads, head_dim);
        repeat_kv_kernel<<<(total_v+255)/256, 256, 0, stream>>>(
            raw_v, full_v, full_seq, n_kv_heads, n_heads, head_dim);
    }

    // 6. Attention
    launch_attention(s->q, full_k, full_v, s->attn_out,
                     full_seq, n_heads, head_dim, stream);

    if (n_kv_heads < n_heads) {
        cudaFree(full_k);
        cudaFree(full_v);
    }

    // 7. Output projection + residual
    sgemm(handle, s->attn_out, &lw->wo, &lw->qwo, s->ffn_out, 1, hd, hd, (float*)s->fp16_a, (float*)s->fp16_c, stream);
    add_residual(s->x, s->ffn_out, hd, stream);

    // 8. Pre-FFN RMSNorm
    launch_rmsnorm(s->x, lw->rms_ffn.data, s->x_norm,
                   1, hd, 1e-5f, stream);

    // 9. SwiGLU FFN
    sgemm(handle, s->x_norm, &lw->w1, &lw->qw1, s->gate, 1, fd, hd, (float*)s->fp16_a, (float*)s->fp16_c, stream);
    sgemm(handle, s->x_norm, &lw->w3, &lw->qw3, s->up, 1, fd, hd, (float*)s->fp16_a, (float*)s->fp16_c, stream);
    launch_swiglu(s->gate, s->up, s->ffn_mid, 1, fd, stream);
    sgemm(handle, s->ffn_mid, &lw->w2, &lw->qw2, s->ffn_out, 1, hd, fd, (float*)s->fp16_a, (float*)s->fp16_c, stream);

    // 10. Residual
    add_residual(s->x, s->ffn_out, hd, stream);
}

float* model_forward(
    cublasHandle_t  handle,
    ModelWeights*   weights,
    KVCache*        cache,
    ForwardState*   state,
    int             token_id,
    cudaStream_t    stream)
{
    int hd  = weights->hidden_dim;
    int pos = cache->current_len;
    int kv_dim = weights->kv_dim > 0 ? weights->kv_dim : hd;

    embed_kernel<<<(hd+255)/256, 256, 0, stream>>>(
        state->x, weights->tok_embeddings.data, token_id, hd);

    for (int l = 0; l < weights->n_layers; l++) {
        forward_layer(handle, state, &weights->layers[l],
                      cache, l, weights->n_heads, kv_dim, pos, stream);
    }

    cache->current_len = pos + 1;

    launch_rmsnorm(state->x, weights->norm.data, state->x_norm,
                   1, hd, 1e-5f, stream);

    gemm(handle, state->x_norm, weights->output.data,
         state->logits, 1, weights->vocab_size, hd, stream);

    return state->logits;
}

extern "C" {
    ForwardState* create_forward_state(int seq, int hd, int fd, int vocab) {
        return forward_state_create(seq, hd, fd, vocab);
    }
    void free_forward_state(ForwardState* s) { forward_state_free(s); }
}

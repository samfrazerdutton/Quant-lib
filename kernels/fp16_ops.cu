#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include "../include/fp16_types.h"

// ── FP16 RMSNorm ──────────────────────────────────────────────────────────────
// out[i] = x[i] / rms(x) * weight[i]  all in fp16
__global__ void fp16_rmsnorm_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half*       __restrict__ out,
    int dim, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[32];

    float ss = 0.f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(x[row * dim + i]);
        ss += v * v;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, mask);
    if (tid % 32 == 0) smem[tid / 32] = ss;
    __syncthreads();

    if (tid < 32) {
        ss = (tid < (blockDim.x / 32)) ? smem[tid] : 0.f;
        for (int mask = 16; mask > 0; mask >>= 1)
            ss += __shfl_down_sync(0xffffffff, ss, mask);
        if (tid == 0) smem[0] = rsqrtf(ss / dim + eps);
    }
    __syncthreads();

    float scale = smem[0];
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(x[row * dim + i]) * scale
                * __half2float(weight[i]);
        out[row * dim + i] = __float2half(v);
    }
}

// ── FP16 RoPE (in-place) ──────────────────────────────────────────────────────
__global__ void fp16_rope_kernel(
    __half* __restrict__ x,      // [seq_len, n_heads, head_dim]
    int seq_len, int n_heads, int head_dim, float theta)
{
    int pos      = blockIdx.x;
    int head     = blockIdx.y;
    int pair_idx = threadIdx.x;
    if (pair_idx >= head_dim / 2) return;

    int base = pos * n_heads * head_dim + head * head_dim;
    float freq = 1.0f / powf(theta, (float)(2 * pair_idx) / head_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle), sin_a = sinf(angle);

    float x0 = __half2float(x[base + 2 * pair_idx]);
    float x1 = __half2float(x[base + 2 * pair_idx + 1]);
    x[base + 2 * pair_idx]     = __float2half(x0 * cos_a - x1 * sin_a);
    x[base + 2 * pair_idx + 1] = __float2half(x0 * sin_a + x1 * cos_a);
}

// ── FP16 SwiGLU ───────────────────────────────────────────────────────────────
__global__ void fp16_swiglu_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half*       __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float silu_g = g / (1.f + expf(-g));
    out[i] = __float2half(silu_g * u);
}

// ── FP16 embedding lookup ─────────────────────────────────────────────────────
__global__ void fp16_embed_kernel(
    const __half* __restrict__ table,  // [vocab, hidden]
    __half*       __restrict__ out,    // [hidden]
    int token_id, int hidden_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim)
        out[i] = table[token_id * hidden_dim + i];
}

// ── FP16 residual add ─────────────────────────────────────────────────────────
__global__ void fp16_add_kernel(
    __half*       __restrict__ x,
    const __half* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

// ── FP16 → FP32 logits conversion ────────────────────────────────────────────
__global__ void fp16_to_fp32_logits(
    const __half* __restrict__ src,
    float*        __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// ── Host launch functions ─────────────────────────────────────────────────────
void launch_fp16_rmsnorm(const __half* x, const __half* w,
                          __half* out, int rows, int dim,
                          float eps, cudaStream_t stream)
{
    fp16_rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, w, out, dim, eps);
}

void launch_fp16_rope(__half* x, int seq_len, int n_heads,
                       int head_dim, float theta, cudaStream_t stream)
{
    dim3 block(head_dim / 2);
    dim3 grid(seq_len, n_heads);
    fp16_rope_kernel<<<grid, block, 0, stream>>>(
        x, seq_len, n_heads, head_dim, theta);
}

void launch_fp16_swiglu(const __half* gate, const __half* up,
                         __half* out, int rows, int ffn_dim,
                         cudaStream_t stream)
{
    int n = rows * ffn_dim;
    fp16_swiglu_kernel<<<(n+255)/256, 256, 0, stream>>>(gate, up, out, n);
}

void launch_fp16_embed(const __half* table, __half* out,
                        int token_id, int hidden_dim, cudaStream_t stream)
{
    fp16_embed_kernel<<<(hidden_dim+255)/256, 256, 0, stream>>>(
        table, out, token_id, hidden_dim);
}

void launch_fp16_add(__half* x, const __half* y, int n, cudaStream_t stream)
{
    fp16_add_kernel<<<(n+255)/256, 256, 0, stream>>>(x, y, n);
}

void launch_fp16_to_fp32(const __half* src, float* dst,
                          int n, cudaStream_t stream)
{
    fp16_to_fp32_logits<<<(n+255)/256, 256, 0, stream>>>(src, dst, n);
}

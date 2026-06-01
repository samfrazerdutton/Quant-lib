#include <cuda_runtime.h>
#include <math.h>

// ── Rotary Position Embedding (RoPE) ─────────────────────────────────────────
// Used in Llama, Mistral, Falcon. Rotates query/key vectors by position angle.
// For each pair of dims (2i, 2i+1) at position pos:
//   out[2i]   = x[2i]   * cos(pos * theta^(-2i/d)) - x[2i+1] * sin(...)
//   out[2i+1] = x[2i+1] * cos(...)                 + x[2i]   * sin(...)
//
// Shape: input [seq_len, n_heads, head_dim]
// One thread per (position, head, dim_pair)

__global__ void rope_kernel(
    float* __restrict__ x,        // in-place: [seq_len, n_heads, head_dim]
    int seq_len,
    int n_heads,
    int head_dim,
    float theta)                  // base = 10000.0f in original paper
{
    int pos      = blockIdx.x;                      // token position
    int head     = blockIdx.y;                      // attention head
    int pair_idx = threadIdx.x;                     // dim pair index (0..head_dim/2)

    if (pos >= seq_len || head >= n_heads || pair_idx >= head_dim / 2)
        return;

    int offset = pos * n_heads * head_dim
               + head * head_dim
               + pair_idx * 2;

    float x0 = x[offset];
    float x1 = x[offset + 1];

    // Compute rotation angle
    float freq = 1.0f / powf(theta, (2.0f * pair_idx) / head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    x[offset]     = x0 * cos_a - x1 * sin_a;
    x[offset + 1] = x0 * sin_a + x1 * cos_a;
}

void launch_rope(float* x, int seq_len, int n_heads,
                 int head_dim, float theta, cudaStream_t stream)
{
    dim3 grid(seq_len, n_heads);
    dim3 block(head_dim / 2);
    rope_kernel<<<grid, block, 0, stream>>>(x, seq_len, n_heads, head_dim, theta);
}

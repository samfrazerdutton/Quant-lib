#include <cuda_runtime.h>

// ── SiLU activation: silu(x) = x * sigmoid(x) ────────────────────────────────
__device__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ── SwiGLU Feed-Forward (Llama FFN) ──────────────────────────────────────────
// out[i] = silu(gate[i]) * up[i]
// gate = x @ W1^T  [seq_len, ffn_dim]
// up   = x @ W3^T  [seq_len, ffn_dim]
// down = (gate*up) @ W2^T  [seq_len, hidden_dim]
// This fused kernel computes the elementwise gate*up in one pass.

__global__ void swiglu_kernel(
    const float* __restrict__ gate,   // [rows, ffn_dim]
    const float* __restrict__ up,     // [rows, ffn_dim]
    float* __restrict__ out,          // [rows, ffn_dim]
    int rows, int ffn_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * ffn_dim;
    if (idx >= total) return;
    out[idx] = silu(gate[idx]) * up[idx];
}

void launch_swiglu(const float* gate, const float* up,
                   float* out, int rows, int ffn_dim,
                   cudaStream_t stream) {
    int total = rows * ffn_dim;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(
        gate, up, out, rows, ffn_dim);
}

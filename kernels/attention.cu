#include <cuda_runtime.h>
#include <float.h>

// ── Scaled Dot-Product Attention ─────────────────────────────────────────────
// Computes: Attention(Q,K,V) = softmax(Q*K^T / sqrt(head_dim)) * V
//
// This is the O(n^2) baseline. Every FlashAttention paper starts here.
// Shapes: Q,K,V = [seq_len, n_heads, head_dim]
//         out   = [seq_len, n_heads, head_dim]
//
// Strategy: one block per (query_pos, head)
//           threads cooperate to dot-product Q[pos] against all K[0..seq_len]

__global__ void attention_kernel(
    const float* __restrict__ Q,    // [seq_len, n_heads, head_dim]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int seq_len,
    int n_heads,
    int head_dim,
    float scale)                    // 1/sqrt(head_dim)
{
    int q_pos = blockIdx.x;   // which query token
    int head  = blockIdx.y;   // which head

    extern __shared__ float smem[];
    float* scores = smem;                    // [seq_len] attention scores
    float* v_acc  = smem + seq_len;          // [head_dim] output accumulator

    int tid = threadIdx.x;

    // Zero the output accumulator
    for (int d = tid; d < head_dim; d += blockDim.x)
        v_acc[d] = 0.0f;

    // ── Step 1: compute Q[q_pos] · K[k_pos] for all k_pos ────────────────
    // Each thread handles some k positions
    for (int k_pos = tid; k_pos < seq_len; k_pos += blockDim.x) {
        float dot = 0.0f;
        int q_off = q_pos * n_heads * head_dim + head * head_dim;
        int k_off = k_pos * n_heads * head_dim + head * head_dim;

        #pragma unroll 4
        for (int d = 0; d < head_dim; d++)
            dot += Q[q_off + d] * K[k_off + d];

        scores[k_pos] = dot * scale;
    }
    __syncthreads();

    // ── Step 2: softmax over scores (thread 0 drives, then broadcast) ─────
    // Simple single-thread softmax for clarity — production uses warp reduce
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int k = 0; k < seq_len; k++) mx = fmaxf(mx, scores[k]);

        float s = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            scores[k] = __expf(scores[k] - mx);
            s += scores[k];
        }
        float inv = 1.0f / s;
        for (int k = 0; k < seq_len; k++) scores[k] *= inv;
    }
    __syncthreads();

    // ── Step 3: weighted sum of V ─────────────────────────────────────────
    // out[q_pos, head, d] = sum_k scores[k] * V[k, head, d]
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            int v_off = k * n_heads * head_dim + head * head_dim;
            acc += scores[k] * V[v_off + d];
        }
        int out_off = q_pos * n_heads * head_dim + head * head_dim;
        out[out_off + d] = acc;
    }
}

void launch_attention(
    const float* Q, const float* K, const float* V,
    float* out,
    int seq_len, int n_heads, int head_dim,
    cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)head_dim);

    // shared mem: scores[seq_len] + v_acc[head_dim]
    size_t smem = (seq_len + head_dim) * sizeof(float);

    dim3 grid(seq_len, n_heads);
    dim3 block(128);   // 128 threads cooperate per (q_pos, head)

    attention_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, out, seq_len, n_heads, head_dim, scale);
}

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include "../include/fp16_types.h"

// ── Flash Attention v2 ────────────────────────────────────────────────────────
// Paper: Dao et al. 2022/2023 "FlashAttention-2"
//
// Key insight: standard attention writes O(n^2) scores to DRAM.
// FlashAttention tiles Q,K,V into SRAM blocks and computes attention
// without ever materializing the full score matrix.
//
// Algorithm (one block per query head):
//   For each tile of K,V:
//     1. Load Q tile, K tile, V tile into shared memory
//     2. Compute S = Q * K^T / sqrt(d)
//     3. Online softmax: track running max m and sum l
//     4. Accumulate O += softmax(S) * V
//     5. Rescale O when max changes
//
// This implementation: one block per (batch, head, query_tile)
// Block size: 128 threads = 4 warps

#define FLASH_BQ  64    // queries per block
#define FLASH_BKV 64    // keys/values per block
#define FLASH_D   64    // head dimension (TinyLlama uses 64)

__global__ void flash_attention_fp16_kernel(
    const __half* __restrict__ Q,    // [seq_len, n_heads, head_dim]
    const __half* __restrict__ K,    // [seq_len, n_kv_heads, head_dim]
    const __half* __restrict__ V,    // [seq_len, n_kv_heads, head_dim]
    __half*       __restrict__ Out,  // [seq_len, n_heads, head_dim]
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    float scale, int causal)
{
    // Each block handles one query tile for one head
    int head    = blockIdx.y;
    int q_start = blockIdx.x * FLASH_BQ;
    int tid     = threadIdx.x;   // 0..127
    int lane    = tid % 32;
    int warp    = tid / 32;

    // GQA: map query head to kv head
    int kv_head = head * n_kv_heads / n_heads;

    __shared__ __half smQ[FLASH_BQ][FLASH_D + 2];
    __shared__ __half smK[FLASH_BKV][FLASH_D + 2];
    __shared__ __half smV[FLASH_BKV][FLASH_D + 2];
    __shared__ float  smO[FLASH_BQ][FLASH_D + 2];
    __shared__ float  sm_m[FLASH_BQ];   // running max
    __shared__ float  sm_l[FLASH_BQ];   // running sum

    // Initialize O, m, l
    for (int i = tid; i < FLASH_BQ; i += blockDim.x) {
        sm_m[i] = -FLT_MAX;
        sm_l[i] = 0.0f;
    }
    for (int i = tid; i < FLASH_BQ * (FLASH_D + 2); i += blockDim.x)
        ((float*)smO)[i] = 0.0f;
    __syncthreads();

    // Load Q tile
    for (int i = tid; i < FLASH_BQ * head_dim; i += blockDim.x) {
        int qi = i / head_dim, di = i % head_dim;
        int gq = q_start + qi;
        smQ[qi][di] = (gq < seq_len)
            ? Q[gq * n_heads * head_dim + head * head_dim + di]
            : __float2half(0.f);
    }
    __syncthreads();

    // Iterate over KV tiles
    int n_kv_tiles = (seq_len + FLASH_BKV - 1) / FLASH_BKV;
    for (int kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * FLASH_BKV;

        // Load K tile
        for (int i = tid; i < FLASH_BKV * head_dim; i += blockDim.x) {
            int ki = i / head_dim, di = i % head_dim;
            int gk = kv_start + ki;
            smK[ki][di] = (gk < seq_len)
                ? K[gk * n_kv_heads * head_dim + kv_head * head_dim + di]
                : __float2half(0.f);
        }

        // Load V tile
        for (int i = tid; i < FLASH_BKV * head_dim; i += blockDim.x) {
            int vi = i / head_dim, di = i % head_dim;
            int gv = kv_start + vi;
            smV[vi][di] = (gv < seq_len)
                ? V[gv * n_kv_heads * head_dim + kv_head * head_dim + di]
                : __float2half(0.f);
        }
        __syncthreads();

        // Each warp handles FLASH_BQ/4 query rows
        int q_per_warp = FLASH_BQ / 4;
        int q_local    = warp * q_per_warp;

        for (int qi = q_local; qi < q_local + q_per_warp; qi++) {
            if (q_start + qi >= seq_len) continue;

            // Compute S[qi, :] = Q[qi] * K^T * scale
            float scores[FLASH_BKV];
            for (int ki = 0; ki < FLASH_BKV; ki++) {
                if (causal && kv_start + ki > q_start + qi) {
                    scores[ki] = -FLT_MAX;
                    continue;
                }
                if (kv_start + ki >= seq_len) {
                    scores[ki] = -FLT_MAX;
                    continue;
                }
                float s = 0.f;
                for (int di = lane; di < head_dim; di += 32)
                    s += __half2float(smQ[qi][di]) * __half2float(smK[ki][di]);
                // Warp reduce
                for (int mask = 16; mask > 0; mask >>= 1)
                    s += __shfl_down_sync(0xffffffff, s, mask);
                scores[ki] = (lane == 0) ? s * scale : 0.f;
                scores[ki] = __shfl_sync(0xffffffff, scores[ki], 0);
            }

            // Online softmax update
            float m_new = sm_m[qi];
            for (int ki = 0; ki < FLASH_BKV; ki++)
                m_new = fmaxf(m_new, scores[ki]);

            float l_new = 0.f;
            for (int ki = 0; ki < FLASH_BKV; ki++) {
                scores[ki] = expf(scores[ki] - m_new);
                l_new += scores[ki];
            }

            // Rescale old O
            float alpha = expf(sm_m[qi] - m_new);
            if (lane == 0) {
                for (int di = 0; di < head_dim; di++)
                    smO[qi][di] = smO[qi][di] * alpha;
            }

            // Accumulate O += scores * V
            if (lane == 0) {
                for (int ki = 0; ki < FLASH_BKV; ki++) {
                    float sv = scores[ki];
                    for (int di = 0; di < head_dim; di++)
                        smO[qi][di] += sv * __half2float(smV[ki][di]);
                }
                sm_m[qi] = m_new;
                sm_l[qi] = alpha * sm_l[qi] + l_new;
            }
        }
        __syncthreads();
    }

    // Normalize and write output
    for (int i = tid; i < FLASH_BQ * head_dim; i += blockDim.x) {
        int qi = i / head_dim, di = i % head_dim;
        int gq = q_start + qi;
        if (gq < seq_len) {
            float val = smO[qi][di] / (sm_l[qi] + 1e-8f);
            Out[gq * n_heads * head_dim + head * head_dim + di] =
                __float2half(val);
        }
    }
}

void launch_flash_attention(
    const __half* Q, const __half* K, const __half* V,
    __half* Out,
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    int causal, cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)head_dim);
    dim3 block(128);
    dim3 grid((seq_len + FLASH_BQ - 1) / FLASH_BQ, n_heads);
    flash_attention_fp16_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, Out,
        seq_len, n_heads, n_kv_heads, head_dim,
        scale, causal);
}

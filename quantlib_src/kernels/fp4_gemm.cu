#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "../include/quant_types.h"

// ── FP4 GEMM with PTX Tensor Core (sm_75 Turing) ─────────────────────────────
// Same mma.sync.aligned.m16n8k8 as fp8_gemm.
// Difference: W is packed fp4 (2 values per byte), unpacked to fp16 in smem.

#define TILE_M 16
#define TILE_K 8

__device__ void mma_m16n8k8_fp4(
    uint32_t a0, uint32_t a1,
    uint32_t b0,
    float& c0, float& c1, float& c2, float& c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},"
        "{%4,%5},"
        "{%6},"
        "{%7,%8,%9,%10};\n"
        : "=f"(c0),"=f"(c1),"=f"(c2),"=f"(c3)
        : "r"(a0),"r"(a1),
          "r"(b0),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3)
    );
}

__device__ __forceinline__ void unpack_fp4(
    fp4_packed_t byte, float scale,
    __half& out_hi, __half& out_lo)
{
    uint8_t q0 = (byte >> 4) & 0x0F;
    uint8_t q1 =  byte       & 0x0F;
    out_hi = __float2half(fp4_e2m1_to_float(q0) / scale);
    out_lo = __float2half(fp4_e2m1_to_float(q1) / scale);
}

__global__ void fp4_gemm_kernel(
    const __half*       __restrict__ A,
    const fp4_packed_t* __restrict__ W,
    const float*        __restrict__ scales,
    __half*             __restrict__ C,
    int M, int N, int K, int group_size)
{
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    int tile_m = blockIdx.y * TILE_M;
    int tile_n = blockIdx.x * 8 + warp * 8;

    float c0=0,c1=0,c2=0,c3=0;

    __shared__ __half smA[TILE_M][TILE_K + 2];
    __shared__ __half smW[8][TILE_K + 2];

    int tid = threadIdx.x;

    for (int k = 0; k < K; k += TILE_K) {

        // Load A tile [16 x 8]
        if (tid < TILE_M * TILE_K) {
            int r = tid / TILE_K, c = tid % TILE_K;
            int gr = tile_m + r, gc = k + c;
            smA[r][c] = (gr < M && gc < K) ? A[gr*K+gc] : __float2half(0.f);
        }

        // Unpack W tile [8 x 8] from fp4 packed bytes
        // W is stored as [N x K/2] — 2 fp4 per byte
        if (tid < 8 * TILE_K) {
            int r = tid / TILE_K, c = tid % TILE_K;
            int wr = tile_n + r;
            int wc = k + c;
            if (wr < N && wc < K) {
                int pair_idx = wr * (K/2) + wc/2;
                int elem_idx = wr * K + wc;
                float scale  = scales[elem_idx / group_size];
                fp4_packed_t byte = W[pair_idx];
                __half hi, lo;
                unpack_fp4(byte, scale, hi, lo);
                smW[r][c] = (wc % 2 == 0) ? hi : lo;
            } else {
                smW[r][c] = __float2half(0.f);
            }
        }

        __syncthreads();

        // A fragment
        int a_row0 = lane / 4;
        int a_row1 = a_row0 + 8;
        int a_col  = (lane % 4) * 2;

        __half a_h[4];
        a_h[0] = smA[a_row0 % TILE_M][a_col    % TILE_K];
        a_h[1] = smA[a_row0 % TILE_M][(a_col+1)% TILE_K];
        a_h[2] = smA[a_row1 % TILE_M][a_col    % TILE_K];
        a_h[3] = smA[a_row1 % TILE_M][(a_col+1)% TILE_K];

        uint32_t a0r, a1r;
        __builtin_memcpy(&a0r, &a_h[0], 4);
        __builtin_memcpy(&a1r, &a_h[2], 4);

        // B fragment
        int b_row = (lane % 4) * 2;
        int b_col =  lane / 4;

        __half b_h[2];
        b_h[0] = smW[b_col % 8][b_row    % TILE_K];
        b_h[1] = smW[b_col % 8][(b_row+1)% TILE_K];

        uint32_t b0r;
        __builtin_memcpy(&b0r, &b_h[0], 4);

        mma_m16n8k8_fp4(a0r, a1r, b0r, c0, c1, c2, c3);

        __syncthreads();
    }

    // Write output
    int out_r0 = tile_m + lane / 4;
    int out_r1 = out_r0 + 8;
    int out_c0 = tile_n + (lane % 4) * 2;
    int out_c1 = out_c0 + 1;

    if (out_r0 < M && out_c0 < N) C[out_r0*N+out_c0] = __float2half(c0);
    if (out_r0 < M && out_c1 < N) C[out_r0*N+out_c1] = __float2half(c1);
    if (out_r1 < M && out_c0 < N) C[out_r1*N+out_c0] = __float2half(c2);
    if (out_r1 < M && out_c1 < N) C[out_r1*N+out_c1] = __float2half(c3);
}

void launch_fp4_gemm(const __half* A, const QuantMatrix* W,
                     __half* C, int M, int N, int K,
                     cudaStream_t stream)
{
    dim3 block(64);
    dim3 grid((N + 15) / 16, (M + TILE_M - 1) / TILE_M);
    fp4_gemm_kernel<<<grid, block, 0, stream>>>(
        A, (fp4_packed_t*)W->data, W->params.scales,
        C, M, N, K, W->params.group_size);
}

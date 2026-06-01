#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "../include/quant_types.h"

// ── FP8 GEMM with PTX Tensor Core (sm_75 Turing) ─────────────────────────────
// Turing PTX: mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
// A fragment : 2 x uint32 (4 fp16 values per thread)
// B fragment : 1 x uint32 (2 fp16 values per thread)
// C/D        : 4 x float  (fp32 accumulators per thread)

#define TILE_M 16
#define TILE_N 16
#define TILE_K 8

__device__ void mma_m16n8k8(
    uint32_t a0, uint32_t a1,   // A fragment: 2 regs = 4 fp16
    uint32_t b0,                // B fragment: 1 reg  = 2 fp16
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

// ── FP8 GEMM kernel ───────────────────────────────────────────────────────────
// C [MxN] fp16 = A [MxK] fp16  x  W [NxK] fp8-E4M3 (dequant inline)
// One warp per 16x8 output tile.
__global__ void fp8_gemm_kernel(
    const __half*     __restrict__ A,
    const fp8_e4m3_t* __restrict__ W,
    const float*      __restrict__ scales,
    __half*           __restrict__ C,
    int M, int N, int K, int group_size)
{
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    int tile_m = blockIdx.y * TILE_M;
    int tile_n = blockIdx.x * 8 + warp * 8;  // each warp owns 8 cols

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

        // Load W tile [8 x 8], dequant fp8->fp16
        if (tid < 8 * TILE_K) {
            int r = tid / TILE_K, c = tid % TILE_K;
            int wr = tile_n + r, wc = k + c;
            if (wr < N && wc < K) {
                int   idx   = wr * K + wc;
                float scale = scales[idx / group_size];
                smW[r][c] = __float2half(fp8_e4m3_to_float(W[idx]) / scale);
            } else {
                smW[r][c] = __float2half(0.f);
            }
        }

        __syncthreads();

        // Build A fragment for this lane
        // m16n8k8: lane t owns rows {t/4, t/4+8}, cols {(t%4)*2, (t%4)*2+1}
        int a_row0 = lane / 4;
        int a_row1 = a_row0 + 8;
        int a_col  = (lane % 4) * 2;

        __half a_h[4];
        a_h[0] = smA[a_row0 % TILE_M][a_col   % TILE_K];
        a_h[1] = smA[a_row0 % TILE_M][(a_col+1)%TILE_K];
        a_h[2] = smA[a_row1 % TILE_M][a_col   % TILE_K];
        a_h[3] = smA[a_row1 % TILE_M][(a_col+1)%TILE_K];

        uint32_t a0_reg, a1_reg;
        __builtin_memcpy(&a0_reg, &a_h[0], 4);
        __builtin_memcpy(&a1_reg, &a_h[2], 4);

        // Build B fragment: lane t owns rows {(t%4)*2,(t%4)*2+1}, col {t/4}
        int b_row = (lane % 4) * 2;
        int b_col =  lane / 4;

        __half b_h[2];
        b_h[0] = smW[b_col % 8][b_row   % TILE_K];
        b_h[1] = smW[b_col % 8][(b_row+1)%TILE_K];

        uint32_t b0_reg;
        __builtin_memcpy(&b0_reg, &b_h[0], 4);

        mma_m16n8k8(a0_reg, a1_reg, b0_reg, c0, c1, c2, c3);

        __syncthreads();
    }

    // Write output
    // lane t writes to rows {t/4, t/4+8}, cols {tile_n + t%4*... }
    int out_r0 = tile_m + lane / 4;
    int out_r1 = out_r0 + 8;
    int out_c0 = tile_n + (lane % 4) * 2;
    int out_c1 = out_c0 + 1;

    if (out_r0 < M && out_c0 < N) C[out_r0*N+out_c0] = __float2half(c0);
    if (out_r0 < M && out_c1 < N) C[out_r0*N+out_c1] = __float2half(c1);
    if (out_r1 < M && out_c0 < N) C[out_r1*N+out_c0] = __float2half(c2);
    if (out_r1 < M && out_c1 < N) C[out_r1*N+out_c1] = __float2half(c3);
}

void launch_fp8_gemm(const __half* A, const QuantMatrix* W,
                     __half* C, int M, int N, int K,
                     cudaStream_t stream)
{
    // Each block: 2 warps, covers 16 rows x 16 cols
    dim3 block(64);
    dim3 grid((N + 15) / 16, (M + TILE_M - 1) / TILE_M);
    fp8_gemm_kernel<<<grid, block, 0, stream>>>(
        A, (fp8_e4m3_t*)W->data, W->params.scales,
        C, M, N, K, W->params.group_size);
}

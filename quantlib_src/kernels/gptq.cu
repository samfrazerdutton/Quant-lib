#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdio.h>
#include "../include/quant_types.h"

// ── GPTQ: Generative Pre-Trained Transformer Quantization ────────────────────
// Paper: Frantar et al. 2022 "GPTQ: Accurate Post-Training Quantization"
//
// Key insight: use second-order information (Hessian) to minimize quantization
// error. Quantize weights column by column, propagating rounding error forward.
//
// Algorithm:
//   1. Compute Hessian H = 2 * X^T * X  from calibration activations X
//   2. Cholesky decompose H to get upper triangular C
//   3. For each column q (left to right):
//      a. Quantize W[:,q] → W_q (round to nearest in quant grid)
//      b. Compute error: e = W[:,q] - W_q
//      c. Propagate error to remaining columns:
//         W[:,q+1:] -= e * C[q, q+1:] / C[q,q]
//
// GPU strategy:
//   - H accumulation: parallel outer products
//   - Cholesky: cuBLAS dpotrf or our own blocked kernel
//   - Column quantization + error propagation: one block per output row

// ── Step 1: Accumulate Hessian H = X^T * X ───────────────────────────────────
// X: [n_tokens x in_features] fp16
// H: [in_features x in_features] float (symmetric)
__global__ void accumulate_hessian(
    const __half* __restrict__ X,
    float*        __restrict__ H,
    int n_tokens, int in_features)
{
    int col_i = blockIdx.x * blockDim.x + threadIdx.x;
    int col_j = blockIdx.y * blockDim.y + threadIdx.y;
    if (col_i >= in_features || col_j >= in_features || col_j < col_i) return;

    float sum = 0.0f;
    for (int t = 0; t < n_tokens; t++)
        sum += __half2float(X[t * in_features + col_i])
             * __half2float(X[t * in_features + col_j]);

    sum *= 2.0f;
    H[col_i * in_features + col_j] = sum;
    H[col_j * in_features + col_i] = sum;  // symmetric
}

// ── Step 2: Simple Cholesky on GPU (blocked, for small in_features) ───────────
// For large matrices use cuSOLVER dpotrf. This handles up to ~2048 cols.
__global__ void cholesky_kernel(float* __restrict__ H, int n)
{
    // Single-block Cholesky: sequential column updates
    // Only launch with 1 block, n threads
    int tid = threadIdx.x;

    for (int k = 0; k < n; k++) {
        // Diagonal element
        if (tid == 0) {
            float diag = H[k * n + k];
            H[k * n + k] = (diag > 1e-8f) ? sqrtf(diag) : 1e-4f;
        }
        __syncthreads();

        float diag_inv = 1.0f / H[k * n + k];

        // Scale column below diagonal
        for (int i = k + 1 + tid; i < n; i += blockDim.x)
            H[i * n + k] *= diag_inv;
        __syncthreads();

        // Rank-1 update of remaining submatrix
        for (int i = k + 1 + tid; i < n; i += blockDim.x) {
            float lik = H[i * n + k];
            for (int j = i; j < n; j++)
                H[i * n + j] -= lik * H[j * n + k];
        }
        __syncthreads();
    }
}

// ── Step 3: GPTQ column-wise quantization with error propagation ──────────────
// W:      [out_features x in_features] fp16 (modified in place during quant)
// H_inv:  [in_features x in_features] float — inverse Hessian diagonal info
// dst:    [out_features x in_features] fp8 E4M3
// scales: [out_features x n_groups]
// zeros:  [out_features x n_groups]
__global__ void gptq_quantize_kernel(
    __half*        __restrict__ W,
    const float*   __restrict__ H,      // Cholesky factor (upper triangular)
    fp8_e4m3_t*    __restrict__ dst,
    float*         __restrict__ scales,
    float*         __restrict__ zeros,
    int out_features, int in_features,
    int group_size, float fp8_max)
{
    int row = blockIdx.x;   // one block per output row
    if (row >= out_features) return;

    int n_groups = (in_features + group_size - 1) / group_size;

    // Process columns left to right
    for (int q = 0; q < in_features; q++) {

        // Recompute group scale at group boundary
        if (q % group_size == 0) {
            int group = q / group_size;
            int start = group * group_size;
            int end   = min(start + group_size, in_features);

            float local_max = 0.0f;
            for (int i = start; i < end; i++)
                local_max = fmaxf(local_max,
                                  fabsf(__half2float(W[row * in_features + i])));

            float scale = (local_max < 1e-8f) ? 1.0f : (fp8_max / local_max);
            scales[row * n_groups + group] = scale;

            // Zero point: midpoint of quantized range
            zeros[row * n_groups + group] = 0.0f;  // symmetric quant
        }

        int   group = q / group_size;
        float scale = scales[row * n_groups + group];

        // Quantize this weight
        float w_fp  = __half2float(W[row * in_features + q]);
        float w_q   = fp8_e4m3_to_float(float_to_fp8_e4m3(w_fp * scale)) / scale;

        // Store quantized value
        dst[row * in_features + q] = float_to_fp8_e4m3(w_fp * scale);

        // Quantization error
        float err = w_fp - w_q;

        // Hessian diagonal for stability
        float h_qq = H[q * in_features + q];
        if (fabsf(h_qq) < 1e-8f) continue;

        // Propagate error to remaining columns
        // W[:,q+1:] -= err * H[q, q+1:] / H[q,q]
        for (int j = q + 1; j < in_features; j++) {
            float h_qj = H[q * in_features + j];
            float delta = err * h_qj / h_qq;
            float w_new = __half2float(W[row * in_features + j]) - delta;
            W[row * in_features + j] = __float2half(w_new);
        }
    }
}

// ── Host launch ───────────────────────────────────────────────────────────────
QuantMatrix launch_gptq_quantize(
    const __half* weights, int out_features, int in_features,
    const __half* acts,    int n_tokens,
    int group_size, cudaStream_t stream)
{
    int n_groups = (in_features + group_size - 1) / group_size;

    // Copy weights to mutable buffer (GPTQ modifies W in place)
    __half* W_buf;
    cudaMalloc(&W_buf, out_features * in_features * sizeof(__half));
    cudaMemcpyAsync(W_buf, weights,
                    out_features * in_features * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);

    // Accumulate Hessian
    float* H;
    cudaMalloc(&H, in_features * in_features * sizeof(float));
    cudaMemsetAsync(H, 0, in_features * in_features * sizeof(float), stream);

    dim3 hblock(16, 16);
    dim3 hgrid((in_features + 15) / 16, (in_features + 15) / 16);
    accumulate_hessian<<<hgrid, hblock, 0, stream>>>(
        acts, H, n_tokens, in_features);

    // Cholesky (single block, sequential — fine for in_features <= 2048)
    int chol_threads = min(in_features, 512);
    cholesky_kernel<<<1, chol_threads, 0, stream>>>(H, in_features);

    // Allocate output
    QuantMatrix qm;
    qm.rows = out_features; qm.cols = in_features; qm.bits = 8;
    snprintf(qm.format, 16, "E4M3");
    cudaMalloc(&qm.data, out_features * in_features * sizeof(fp8_e4m3_t));

    int total_scales = out_features * n_groups;
    cudaMalloc(&qm.params.scales, total_scales * sizeof(float));
    cudaMalloc(&qm.params.zeros,  total_scales * sizeof(float));
    qm.params.group_size  = group_size;
    qm.params.granularity = QUANT_PER_GROUP;
    qm.params.n_scales    = total_scales;

    // GPTQ quantize: one block per output row
    gptq_quantize_kernel<<<out_features, 1, 0, stream>>>(
        W_buf, H,
        (fp8_e4m3_t*)qm.data,
        qm.params.scales, qm.params.zeros,
        out_features, in_features, group_size, 448.0f);

    cudaFree(W_buf);
    cudaFree(H);

    return qm;
}

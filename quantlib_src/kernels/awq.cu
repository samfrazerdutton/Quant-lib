#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdio.h>
#include "../include/quant_types.h"

// ── Activation-Aware Weight Quantization (AWQ) ────────────────────────────────
// Paper: Lin et al. 2023 "AWQ: Activation-aware Weight Quantization for LLM"
//
// Key insight: not all weights are equally important.
// Weights multiplied by HIGH-activation channels cause more error when quantized.
// Solution: scale those weight channels UP before quantizing (then scale acts DOWN).
// This shifts quantization error away from salient channels.
//
// Algorithm per group:
//   1. Compute per-channel activation magnitude: s[c] = mean(|act[:,c]|)
//   2. Find optimal per-channel scale: alpha[c] = s[c]^0.5  (geometric mean)
//   3. Scale weights: W_scaled[c] = W[c] * alpha[c]
//   4. Quantize W_scaled with standard per-group quantization
//   5. Store 1/alpha alongside scales so runtime can compensate activations

// ── Step 1: Compute per-channel activation magnitude ─────────────────────────
// acts: [n_tokens x in_features] fp16
// out:  [in_features] float — mean absolute value per input channel
__global__ void compute_act_scales(
    const __half* __restrict__ acts,
    float*        __restrict__ act_scales,
    int n_tokens, int in_features)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= in_features) return;

    float sum = 0.0f;
    for (int t = 0; t < n_tokens; t++)
        sum += fabsf(__half2float(acts[t * in_features + col]));

    act_scales[col] = sum / (float)n_tokens;
}

// ── Step 2: Compute optimal AWQ scale per group ───────────────────────────────
// For each group of weights, find alpha = geometric_mean(act_scales)^0.5
// clamped to [1/clip, clip] for stability.
__global__ void compute_awq_alphas(
    const float* __restrict__ act_scales,  // [in_features]
    float*       __restrict__ alphas,      // [n_groups] output
    int in_features, int group_size)
{
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    int n_groups = (in_features + group_size - 1) / group_size;
    if (group >= n_groups) return;

    int start = group * group_size;
    int end   = min(start + group_size, in_features);

    // Geometric mean of activation scales in this group
    float log_sum = 0.0f;
    int   count   = end - start;
    for (int i = start; i < end; i++)
        log_sum += logf(act_scales[i] + 1e-8f);

    float geo_mean = expf(log_sum / count);

    // Alpha = sqrt(geo_mean), clamped for stability
    float alpha = 1.0f / (sqrtf(geo_mean) + 1e-8f);
    alpha = fmaxf(alpha, 0.01f);
    alpha = fminf(alpha, 100.0f);
    alphas[group] = alpha;
}

// ── Step 3+4: Scale weights and quantize in one fused kernel ──────────────────
// weights: [out_features x in_features] fp16
// alphas:  [n_groups] — one per input-feature group
// dst:     [out_features x in_features] fp8 E4M3
// scales:  [out_features x n_groups] quantization scales
// inv_alphas: [n_groups] stored for runtime activation compensation
__global__ void awq_quantize_kernel(
    const __half*  __restrict__ weights,
    const float*   __restrict__ alphas,
    fp8_e4m3_t*    __restrict__ dst,
    float*         __restrict__ quant_scales,
    float*         __restrict__ inv_alphas,
    int out_features, int in_features, int group_size)
{
    int row   = blockIdx.y;                          // output channel
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    int n_groups = (in_features + group_size - 1) / group_size;
    if (group >= n_groups || row >= out_features) return;

    float alpha = alphas[group];

    // Store inv_alpha once (only row 0 needs to write it, others are same)
    if (row == 0) inv_alphas[group] = 1.0f / alpha;

    // Find max |w * alpha| in this group for this output row
    int start = group * group_size;
    int end   = min(start + group_size, in_features);

    float local_max = 0.0f;
    for (int i = start; i < end; i++) {
        float w = __half2float(weights[row * in_features + i]) * alpha;
        local_max = fmaxf(local_max, fabsf(w));
    }

    float scale = (local_max < 1e-8f) ? 1.0f : (448.0f / local_max);
    quant_scales[row * n_groups + group] = scale;

    // Quantize scaled weights
    for (int i = start; i < end; i++) {
        float w   = __half2float(weights[row * in_features + i]) * alpha;
        dst[row * in_features + i] = float_to_fp8_e4m3(w * scale);
    }
}

// ── Host launch ───────────────────────────────────────────────────────────────
QuantMatrix launch_awq_quantize(
    const __half* weights, int out_features, int in_features,
    const __half* acts,    int n_tokens,
    int group_size, cudaStream_t stream)
{
    int n_groups = (in_features + group_size - 1) / group_size;

    // Allocate temporaries
    float *act_scales, *alphas, *inv_alphas;
    cudaMalloc(&act_scales, in_features * sizeof(float));
    cudaMalloc(&alphas,     n_groups    * sizeof(float));
    cudaMalloc(&inv_alphas, n_groups    * sizeof(float));

    // Step 1: activation magnitudes
    int threads = 256;
    int blocks  = (in_features + threads - 1) / threads;
    compute_act_scales<<<blocks, threads, 0, stream>>>(
        acts, act_scales, n_tokens, in_features);

    // Step 2: per-group alpha
    blocks = (n_groups + threads - 1) / threads;
    compute_awq_alphas<<<blocks, threads, 0, stream>>>(
        act_scales, alphas, in_features, group_size);

    // Step 3+4: scale + quantize
    QuantMatrix qm;
    qm.rows = out_features; qm.cols = in_features; qm.bits = 8;
    snprintf(qm.format, 16, "E4M3");
    cudaMalloc(&qm.data, out_features * in_features * sizeof(fp8_e4m3_t));

    int total_scales = out_features * n_groups;
    cudaMalloc(&qm.params.scales, total_scales * sizeof(float));
    cudaMalloc(&qm.params.zeros,  n_groups     * sizeof(float)); // reuse for inv_alpha
    qm.params.group_size  = group_size;
    qm.params.granularity = QUANT_PER_GROUP;
    qm.params.n_scales    = total_scales;

    dim3 block2(32);
    dim3 grid2((n_groups + 31) / 32, out_features);
    awq_quantize_kernel<<<grid2, block2, 0, stream>>>(
        weights, alphas,
        (fp8_e4m3_t*)qm.data, qm.params.scales, inv_alphas,
        out_features, in_features, group_size);

    // Copy inv_alphas into zeros slot for runtime use
    cudaMemcpyAsync(qm.params.zeros, inv_alphas,
                    n_groups * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    cudaFree(act_scales);
    cudaFree(alphas);
    cudaFree(inv_alphas);

    return qm;
}

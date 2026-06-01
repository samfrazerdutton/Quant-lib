#include <cuda_runtime.h>
#include <math.h>

// ── RMS Norm ──────────────────────────────────────────────────────────────────
// Used in Llama instead of LayerNorm. Simpler: no mean subtraction.
// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
// One block per row (one token). Warp-reduce for the sum of squares.

__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int dim, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float smem[32];

    // Sum of squares
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
        ss += x[row*dim + i] * x[row*dim + i];

    // Warp reduce
    ss = warp_reduce_sum(ss);
    if (tid % 32 == 0) smem[tid/32] = ss;
    __syncthreads();

    // Final reduce across warps
    if (tid < 32) {
        ss = (tid < (blockDim.x/32)) ? smem[tid] : 0.0f;
        ss = warp_reduce_sum(ss);
        if (tid == 0) smem[0] = ss;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / dim + eps);

    for (int i = tid; i < dim; i += blockDim.x)
        out[row*dim + i] = x[row*dim + i] * rms_inv * weight[i];
}

void launch_rmsnorm(const float* x, const float* weight,
                    float* out, int rows, int dim,
                    float eps, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(rows);
    rmsnorm_kernel<<<grid, block, 0, stream>>>(x, weight, out, dim, eps);
}

#include <cuda_runtime.h>
#include <float.h>

// ── Warp-level reduction helpers ─────────────────────────────────────────────
// These use shuffle intrinsics — no shared memory needed, runs inside one warp
__device__ static float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ static float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ── Online softmax kernel ─────────────────────────────────────────────────────
// One block per row. Block size = 256 threads (8 warps).
// Uses the numerically stable: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// This is the exact pattern used inside every attention layer.
__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = input  + row * cols;
    float*       row_out = output + row * cols;

    // Shared memory for warp-level max and sum aggregation
    __shared__ float smem[32]; // one slot per warp (max 32 warps per block)

    int tid   = threadIdx.x;
    int nthrd = blockDim.x;
    int warp  = tid / 32;
    int lane  = tid % 32;
    int nwarp = nthrd / 32;

    // ── Pass 1: find row max ──────────────────────────────────────────────
    float thread_max = -FLT_MAX;
    for (int i = tid; i < cols; i += nthrd)
        thread_max = fmaxf(thread_max, row_in[i]);

    // Warp reduce
    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) smem[warp] = thread_max;
    __syncthreads();

    // First warp reduces across warps
    float block_max = -FLT_MAX;
    if (tid < nwarp) block_max = smem[tid];
    if (warp == 0)   block_max = warp_reduce_max(block_max);
    if (tid == 0)    smem[0]   = block_max;
    __syncthreads();

    float row_max = smem[0];

    // ── Pass 2: compute exp(x - max) and sum ─────────────────────────────
    float thread_sum = 0.0f;
    for (int i = tid; i < cols; i += nthrd) {
        float e = __expf(row_in[i] - row_max);  // __expf = fast hardware exp
        row_out[i] = e;
        thread_sum += e;
    }

    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) smem[warp] = thread_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < nwarp) block_sum = smem[tid];
    if (warp == 0)   block_sum = warp_reduce_sum(block_sum);
    if (tid == 0)    smem[0]   = block_sum;
    __syncthreads();

    float inv_sum = 1.0f / smem[0];

    // ── Pass 3: normalize ─────────────────────────────────────────────────
    for (int i = tid; i < cols; i += nthrd)
        row_out[i] *= inv_sum;
}

void launch_softmax(const float* input, float* output,
                    int rows, int cols, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(rows);
    softmax_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
}

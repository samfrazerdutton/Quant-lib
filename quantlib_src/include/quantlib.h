#pragma once
#include <cuda_runtime.h>
#include "quant_types.h"

#ifdef __cplusplus
extern "C" {
#endif

QuantMatrix quantlib_fp16_to_fp8_e4m3(const __half* src, int rows, int cols,
                                       QuantGranularity gran, int group_size,
                                       cudaStream_t stream);

QuantMatrix quantlib_fp16_to_fp8_e5m2(const __half* src, int rows, int cols,
                                       QuantGranularity gran, int group_size,
                                       cudaStream_t stream);

QuantMatrix quantlib_fp16_to_fp4    (const __half* src, int rows, int cols,
                                       QuantGranularity gran, int group_size,
                                       cudaStream_t stream);

void quantlib_fp8_to_fp16(const QuantMatrix* qm, __half* dst, cudaStream_t stream);
void quantlib_fp4_to_fp16(const QuantMatrix* qm, __half* dst, cudaStream_t stream);

void quantlib_fp8_gemm(const __half* A, const QuantMatrix* W,
                        __half* C, int M, int N, int K,
                        cudaStream_t stream);

void quantlib_fp4_gemm(const __half* A, const QuantMatrix* W,
                        __half* C, int M, int N, int K,
                        cudaStream_t stream);

QuantMatrix quantlib_awq_quantize(const __half* weights, int rows, int cols,
                                   const __half* calibration_acts,
                                   int n_calib_tokens,
                                   int group_size, int bits,
                                   cudaStream_t stream);

QuantMatrix quantlib_gptq_quantize(const __half* weights, int rows, int cols,
                                    const __half* calibration_acts,
                                    int n_calib_tokens,
                                    int group_size, int bits,
                                    cudaStream_t stream);

void  quantlib_free(QuantMatrix* qm);
float quantlib_quant_error(const __half* original, const __half* dequantized,
                            int n_elements);
void  quantlib_print_stats(const QuantMatrix* qm);

#ifdef __cplusplus
}
#endif

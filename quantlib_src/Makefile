NVCC      = nvcc
NVCCFLAGS = -O3 -arch=sm_75 --use_fast_math -lineinfo -Iinclude

KERNELS = kernels/fp8_convert.cu \
          kernels/fp4_convert.cu \
          kernels/fp8_gemm.cu    \
          kernels/fp4_gemm.cu    \
          kernels/awq.cu         \
          kernels/gptq.cu

LIB_SRC = src/quantlib.cu $(KERNELS)

all: libquant.a test_fp8 test_fp4 test_awq

# Static library — linked into llm_engine
libquant.a: $(LIB_SRC)
	$(NVCC) $(NVCCFLAGS) --lib $^ -o $@

test_fp8: libquant.a tests/test_fp8.cu
	$(NVCC) $(NVCCFLAGS) tests/test_fp8.cu -o $@ -L. -lquant -lm

test_fp4: libquant.a tests/test_fp4.cu
	$(NVCC) $(NVCCFLAGS) tests/test_fp4.cu -o $@ -L. -lquant -lm

test_awq: libquant.a tests/test_awq.cu
	$(NVCC) $(NVCCFLAGS) tests/test_awq.cu -o $@ -L. -lquant -lm

test_gemm: libquant.a tests/test_gemm.cu
	$(NVCC) $(NVCCFLAGS) tests/test_gemm.cu -o $@ -L. -lquant -lm

clean:
	rm -f libquant.a test_fp8 test_fp4 test_awq test_gemm

NVCC      = nvcc
NVCCFLAGS = -O3 -arch=sm_75 --use_fast_math -lineinfo -Iinclude -I../quantlib/include
QUANTLIB  = ../quantlib

all: test_matmul test_softmax test_attention test_kvcache test_forward test_real test_quant_infer

test_matmul: kernels/matmul.cu tests/test_matmul.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_softmax: kernels/softmax.cu tests/test_softmax.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_attention: kernels/rope.cu kernels/attention.cu tests/test_attention.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_kvcache: src/kv_cache.cu src/weights.cu tests/test_kvcache.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_forward: src/kv_cache.cu src/weights.cu src/forward.cu \
              kernels/rmsnorm.cu kernels/rope.cu kernels/attention.cu \
              kernels/feedforward.cu tests/test_forward.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

test_real: src/kv_cache.cu src/weights.cu src/forward.cu src/load_weights.cu \
           kernels/rmsnorm.cu kernels/rope.cu kernels/attention.cu \
           kernels/feedforward.cu tests/test_real_inference.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas

test_quant_infer: src/kv_cache.cu src/weights.cu src/forward.cu \
                  src/load_weights.cu src/quant_forward.cu \
                  kernels/rmsnorm.cu kernels/rope.cu kernels/attention.cu \
                  kernels/feedforward.cu tests/test_quant_infer.cu
	$(NVCC) $(NVCCFLAGS) -I$(QUANTLIB)/include $^ -o $@ \
	        -L$(QUANTLIB) -lquant -lcublas -lm

clean:
	rm -f test_matmul test_softmax test_attention test_kvcache \
	      test_forward test_real test_quant_infer

FP16_SRC = src/paged_kvcache.cu src/fp16_forward.cu \
           kernels/fp16_ops.cu kernels/flash_attention.cu

test_fp16_forward: $(FP16_SRC) tests/test_fp16_forward.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcublas -lm


test_continuous_batch: src/paged_kvcache.cu src/continuous_batch.cu tests/test_continuous_batch.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lm


SERVER_SRC = src/server.cu src/gguf_loader.cu src/continuous_batch.cu src/paged_kvcache.cu
SERVER_OBJ = $(SERVER_SRC:.cu=.o)

llm_server: $(SERVER_SRC) src/fp16_forward.cu kernels/fp16_ops.cu kernels/flash_attention.cu ../quantlib/libquant.a
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -L../quantlib -lquant -lcublas -lm -lpthread


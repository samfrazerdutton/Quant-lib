# CUDA LLM Inference Engine

A production-grade transformer inference engine built from scratch in C++/CUDA.
No PyTorch. No TensorFlow. Every kernel written and benchmarked by hand.

**Hardware:** NVIDIA GeForce RTX 2060 Max-Q (6GB VRAM, sm_75 Turing)

---

## Benchmark Results

### 1. Tiled GEMM — shared memory tiling + bank conflict padding

| Matrix Size        | Time (ms) | TFLOPS |
|--------------------|-----------|--------|
| 512 × 512 × 512    | 0.7283    | 0.369  |
| 1024 × 1024 × 1024 | 3.3276    | 0.645  |
| 2048 × 2048 × 2048 | 25.941    | 0.662  |
| 4096 × 4096 × 4096 | 208.574   | 0.659  |

### 2. Softmax — warp `__shfl_down_sync` reductions, numerically stable

| Shape       | Time (ms) | GB/s  | % of Peak (336 GB/s) |
|-------------|-----------|-------|----------------------|
| 512 × 512   | 0.0136    | 154.1 | 45.9%                |
| 1024 × 1024 | 0.0377    | 222.6 | 66.2%                |
| 2048 × 2048 | 0.1430    | 234.6 | 69.8%                |
| 4096 × 4096 | 0.6598    | 203.4 | 60.5%                |

### 3. RMSNorm — warp-reduce, runs every transformer layer

| Shape        | Time (ms) | GB/s  | % of Peak |
|--------------|-----------|-------|-----------|
| 2048 × 512   | 0.0390    | 322.9 | 96.1%     |
| 2048 × 1024  | 0.0755    | 333.4 | 99.2%     |
| 2048 × 2048  | 0.1494    | 337.0 | **100.3%**|
| 2048 × 4096  | 0.3757    | 268.0 | 79.8%     |

### 4. Multi-Head Attention — 32 heads, head_dim=128, O(n²) baseline

| seq_len | Time (ms) | GFLOPS |
|---------|-----------|--------|
| 64      | 0.6857    | 97.87  |
| 128     | 2.7028    | 99.32  |
| 256     | 10.8948   | 98.56  |
| 512     | 44.1217   | 97.34  |

### 5. RoPE — Rotary Position Embedding, isometry-preserving rotation

| seq × heads  | Time (ms) | GB/s  |
|--------------|-----------|-------|
| 128 × 32     | 0.0113    | 370.9 |
| 512 × 32     | 0.0729    | 230.0 |
| 1024 × 32    | 0.1429    | 234.9 |
| 2048 × 32    | 0.2842    | 236.1 |

### 6. SwiGLU FFN — `silu(gate) * up`, exact Llama activation

| rows × ffn_dim | Time (ms) | GB/s  |
|----------------|-----------|-------|
| 1 × 5632       | 0.0339    | 2.0   |
| 32 × 5632      | 0.0110    | 197.3 |
| 128 × 5632     | 0.0377    | 229.3 |
| 512 × 5632     | 0.1434    | 241.3 |

### 7. KV Cache — fused append kernel, all layers in one launch

| Config                  | Time (ms/tok) |
|-------------------------|---------------|
| 4 layers, kv=4, hd=64   | 0.2751        |
| 8 layers, kv=4, hd=64   | 0.5461        |
| 16 layers, kv=4, hd=64  | 1.0679        |
| 32 layers, kv=4, hd=64  | 0.7727        |

### 8. End-to-End Inference — TinyLlama-1.1B, real weights

| Model            | tok/s  | ms/tok | VRAM  |
|------------------|--------|--------|-------|
| FP32 (baseline)  | 48.4   | 20.6   | ~5 GB |
| FP8 quantized    | 21.9   | 45.6   | ~3 GB |

---

## Architecture

### Inference Kernels (`kernels/`)
- **matmul.cu** — tiled GEMM, 32×32 shared memory tiles, `+1` padding eliminates bank conflicts
- **softmax.cu** — numerically stable `exp(x - max)`, warp-level `__shfl_down_sync` reductions
- **attention.cu** — scaled dot-product attention, O(n²) baseline with causal masking
- **flash_attention.cu** — FlashAttention-2: tiled SRAM computation, eliminates O(n²) DRAM writes
- **rope.cu** — Rotary Position Embeddings, isometry-preserving (norms preserved to 1e-6)
- **rmsnorm.cu** — RMSNorm hitting 100% memory bandwidth at hidden_dim=2048
- **feedforward.cu** — SwiGLU: fused `silu(gate) * up` elementwise kernel
- **fp16_ops.cu** — FP16 variants of all ops for quantized forward path

### Engine (`src/`)
- **forward.cu** — full transformer forward pass with GQA support (handles k_proj < hidden_dim)
- **kv_cache.cu** — GPU KV cache, O(1) per-token append via single fused CUDA kernel
- **paged_kvcache.cu** — PagedAttention: block-table based memory management (vLLM algorithm)
- **continuous_batch.cu** — multi-request scheduler with VRAM recycling between sequences
- **fp16_forward.cu** — FP16 inference pipeline wired to Flash Attention
- **load_weights.cu** — binary weight loader, auto-detects GQA from tensor shapes
- **gguf_loader.cu** — GGUF v3 parser, loads Q4_K_M quantized models
- **server.cu** — multithreaded inference server, pthreads network listener

### Quantization Library (`quantlib/`)
- **fp8_convert.cu** — FP16→FP8 E4M3/E5M2, per-group/per-channel/per-tensor granularity
- **fp4_convert.cu** — FP16→FP4 E2M1 nibble packing, exact roundtrip verified
- **fp8_gemm.cu** — FP8 GEMM via PTX `mma.sync.aligned.m16n8k8` Tensor Core: **1,238 GFLOPS**
- **fp4_gemm.cu** — FP4 GEMM via PTX Tensor Core: **1,119 GFLOPS**
- **awq.cu** — Activation-aware Weight Quantization (Lin et al. 2023), salient channel protection verified
- **gptq.cu** — GPTQ (Frantar et al. 2022), Hessian-based second-order error propagation

---

## How to Build

**Requirements:** CUDA 13+, `nvcc`, GPU sm_75 or newer

```bash
# Download TinyLlama weights (638 MB, no login required)
mkdir -p models
wget -O models/tinyllama.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Build quantization library
cd quantlib && make libquant.a && cd ..

# Build everything
make all

# Run individual benchmarks
./test_matmul        # tiled GEMM correctness + sweep
./test_softmax       # warp-reduce bandwidth
./test_attention     # RoPE + MHA correctness
./test_kvcache       # paged KV cache
./test_forward       # tiny model forward pass
./test_real          # TinyLlama-1.1B real inference
./test_quant_infer   # FP32 vs FP8 comparison
./benchmark          # full benchmark table

# Run the inference server
./llm_server
```

## GPU Compatibility

Change `-arch=sm_75` in both Makefiles:

| Flag    | GPU Family      |
|---------|-----------------|
| sm_75   | RTX 20xx (Turing) |
| sm_86   | RTX 30xx (Ampere) |
| sm_89   | RTX 40xx (Ada)    |
| sm_80   | A100              |

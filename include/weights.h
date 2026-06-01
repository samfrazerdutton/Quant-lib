#pragma once
#include <cuda_runtime.h>
#include "quant_types.h"

// ── Weight tensor (one matrix) ────────────────────────────────────────────────
typedef struct {
    float* data;
    int    rows;
    int    cols;
    char   name[64];
} Tensor;

// ── Quantized weight matrix (from quantlib) ───────────────────────────────────
// Mirrors QuantMatrix but included here to avoid circular deps.
// Populated by quant_forward.cu after AWQ/FP8 conversion.
typedef struct {
    void*  data;          // GPU pointer (fp8 or fp4 packed)
    float* scales;        // per-group scales
    float* zeros;         // per-group zero points / inv_alphas (AWQ)
    int    rows;
    int    cols;
    int    bits;          // 8 or 4
    int    group_size;
    int    n_scales;
    char   format[16];    // "E4M3", "E5M2", "E2M1"
    int    valid;         // 1 = quantized data present
} QTensor;

// ── One transformer layer's weights ──────────────────────────────────────────
typedef struct {
    // FP32 originals (loaded from disk)
    Tensor wq, wk, wv, wo;
    Tensor w1, w2, w3;
    Tensor rms_attn, rms_ffn;

    // FP8 quantized versions (filled by quantize_model())
    QTensor qwq, qwk, qwv, qwo;
    QTensor qw1, qw2, qw3;
} LayerWeights;

// ── Full model weights ────────────────────────────────────────────────────────
typedef struct {
    int          n_layers;
    int          hidden_dim;
    int          n_heads;
    int          head_dim;
    int          ffn_dim;
    int          vocab_size;
    int          kv_dim;
    LayerWeights* layers;
    Tensor        tok_embeddings;
    Tensor        norm;
    Tensor        output;
} ModelWeights;

// ── Function declarations ─────────────────────────────────────────────────────
ModelWeights* weights_alloc(int n_layers, int hidden_dim, int n_heads,
                             int head_dim, int ffn_dim, int vocab_size);
void          weights_free(ModelWeights* w);
ModelWeights* weights_load_bin(const char* path, int n_layers, int hidden_dim,
                                int n_heads, int head_dim, int vocab_size);

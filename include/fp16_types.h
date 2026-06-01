#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ── FP16 tensor (GPU pointer) ─────────────────────────────────────────────────
typedef struct {
    __half* data;
    int     rows;
    int     cols;
    char    name[64];
} HalfTensor;

// ── FP16 forward state ────────────────────────────────────────────────────────
// All activations stored as fp16. Eliminates fp32<->fp16 conversion overhead.
typedef struct {
    __half* x;          // [hidden_dim]  current hidden state
    __half* x_norm;     // [hidden_dim]  after RMSNorm
    __half* q;          // [n_heads * head_dim]
    __half* k;          // [n_kv_heads * head_dim]
    __half* v;          // [n_kv_heads * head_dim]
    __half* attn_out;   // [n_heads * head_dim]
    __half* gate;       // [ffn_dim]
    __half* up;         // [ffn_dim]
    __half* ffn_mid;    // [ffn_dim]
    __half* ffn_out;    // [hidden_dim]
    __half* logits_fp16;// [vocab_size]
    float*  logits;     // [vocab_size] fp32 for sampling
    int     hidden_dim;
    int     ffn_dim;
    int     kv_dim;
    int     vocab_size;
} FP16State;

// ── FP16 model weights (one layer) ───────────────────────────────────────────
typedef struct {
    HalfTensor wq, wk, wv, wo;
    HalfTensor w1, w2, w3;
    HalfTensor rms_attn, rms_ffn;
} FP16LayerWeights;

// ── FP16 full model ───────────────────────────────────────────────────────────
typedef struct {
    int              n_layers;
    int              hidden_dim;
    int              n_heads;
    int              n_kv_heads;
    int              head_dim;
    int              ffn_dim;
    int              vocab_size;
    int              max_seq_len;
    float            rope_theta;
    FP16LayerWeights* layers;
    HalfTensor        tok_embeddings; // [vocab_size, hidden_dim]
    HalfTensor        norm;           // [hidden_dim]
    HalfTensor        output;         // [vocab_size, hidden_dim]
} FP16Model;

// ── Paged KV cache block ──────────────────────────────────────────────────────
#define PAGE_SIZE 16   // tokens per page

typedef struct {
    __half* k_data;    // [n_layers, PAGE_SIZE, n_kv_heads, head_dim]
    __half* v_data;
    int     block_id;
    int     ref_count;
} KVBlock;

// ── Sequence state for continuous batching ────────────────────────────────────
typedef enum {
    SEQ_PREFILL  = 0,
    SEQ_DECODE   = 1,
    SEQ_DONE     = 2
} SeqStatus;

typedef struct {
    int        seq_id;
    int*       tokens;         // host: full token history
    int        n_tokens;       // tokens generated so far
    int        max_tokens;     // generation limit
    int*       kv_block_ids;   // which KV blocks this seq owns
    int        n_blocks;
    SeqStatus  status;
    float      temperature;
    int        top_k;
} Sequence;

// ── Speculative decoding state ────────────────────────────────────────────────
typedef struct {
    FP16Model* draft_model;    // small fast model
    FP16Model* target_model;   // large accurate model
    int        n_draft_tokens; // how many to speculate ahead (typically 4-8)
    int*       draft_tokens;   // GPU buffer for draft token ids
    float*     draft_logprobs; // GPU buffer for draft log probs
} SpecState;

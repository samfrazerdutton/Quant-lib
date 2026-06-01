#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../include/weights.h"

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

static void upload_tensor(Tensor* t, int rows, int cols,
                          const float* data) {
    t->rows = rows;
    t->cols = cols;
    size_t n = (size_t)rows * cols;
    CUDA_CHECK(cudaMalloc(&t->data, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(t->data, data, n * sizeof(float),
                          cudaMemcpyHostToDevice));
}

static int assign_tensor(ModelWeights* w, const char* name,
                         int rows, int cols, const float* data) {
    if (strcmp(name, "model.embed_tokens.weight") == 0) {
        upload_tensor(&w->tok_embeddings, rows, cols, data); return 1;
    }
    if (strcmp(name, "model.norm.weight") == 0) {
        upload_tensor(&w->norm, rows, cols, data); return 1;
    }
    if (strcmp(name, "lm_head.weight") == 0) {
        upload_tensor(&w->output, rows, cols, data); return 1;
    }

    int layer = -1;
    char suffix[128];
    if (sscanf(name, "model.layers.%d.%127s", &layer, suffix) == 2
        && layer >= 0 && layer < w->n_layers) {
        LayerWeights* lw = &w->layers[layer];
        Tensor* t = NULL;

        if      (strcmp(suffix, "self_attn.q_proj.weight") == 0)           t = &lw->wq;
        else if (strcmp(suffix, "self_attn.k_proj.weight") == 0)           t = &lw->wk;
        else if (strcmp(suffix, "self_attn.v_proj.weight") == 0)           t = &lw->wv;
        else if (strcmp(suffix, "self_attn.o_proj.weight") == 0)           t = &lw->wo;
        else if (strcmp(suffix, "mlp.gate_proj.weight") == 0)              t = &lw->w1;
        else if (strcmp(suffix, "mlp.down_proj.weight") == 0)              t = &lw->w2;
        else if (strcmp(suffix, "mlp.up_proj.weight") == 0)                t = &lw->w3;
        else if (strcmp(suffix, "input_layernorm.weight") == 0)            t = &lw->rms_attn;
        else if (strcmp(suffix, "post_attention_layernorm.weight") == 0)   t = &lw->rms_ffn;

        if (t && t->data) {
            upload_tensor(t, rows, cols, data); return 1;
        }
    }
    return 0;
}

ModelWeights* weights_load_bin(const char* path,
                               int n_layers, int hidden_dim,
                               int n_heads, int ffn_dim, int vocab_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }

    ModelWeights* w = (ModelWeights*)calloc(1, sizeof(ModelWeights));
    w->n_layers   = n_layers;
    w->hidden_dim = hidden_dim;
    w->n_heads    = n_heads;
    w->head_dim   = hidden_dim / n_heads;
    w->ffn_dim    = ffn_dim;
    w->vocab_size = vocab_size;
    w->layers     = (LayerWeights*)calloc(n_layers, sizeof(LayerWeights));

    int hd = hidden_dim;

    for (int l = 0; l < n_layers; l++) {
        LayerWeights* lw = &w->layers[l];
        // Q is always [hidden_dim x hidden_dim]
        // K,V for GQA models are smaller but we allocate full size,
        // the loader will set actual rows/cols from the file
        CUDA_CHECK(cudaMalloc(&lw->wq.data,       (size_t)hd*hd*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lw->wk.data,       (size_t)hd*hd*sizeof(float))); // may be smaller
        CUDA_CHECK(cudaMalloc(&lw->wv.data,       (size_t)hd*hd*sizeof(float))); // may be smaller
        CUDA_CHECK(cudaMalloc(&lw->wo.data,       (size_t)hd*hd*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lw->w1.data,       (size_t)ffn_dim*hd*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lw->w2.data,       (size_t)hd*ffn_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lw->w3.data,       (size_t)ffn_dim*hd*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lw->rms_attn.data, (size_t)hd*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lw->rms_ffn.data,  (size_t)hd*sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&w->tok_embeddings.data,
                          (size_t)vocab_size * hd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w->norm.data,   (size_t)hd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w->output.data, (size_t)vocab_size * hd * sizeof(float)));

    uint32_t n_tensors;
    fread(&n_tensors, 4, 1, f);
    printf("Loading %u tensors...\n", n_tensors);

    int loaded = 0;
    for (uint32_t i = 0; i < n_tensors; i++) {
        uint32_t name_len;
        fread(&name_len, 4, 1, f);
        char name[256] = {0};
        fread(name, 1, name_len, f);
        uint32_t rows, cols;
        fread(&rows, 4, 1, f);
        fread(&cols, 4, 1, f);
        size_t n = (size_t)rows * cols;
        float* buf = (float*)malloc(n * sizeof(float));
        fread(buf, 4, n, f);
        if (assign_tensor(w, name, rows, cols, buf)) loaded++;
        free(buf);
    }
    fclose(f);

    printf("Loaded %d / %u tensors\n", loaded, n_tensors);

    // GQA detection: if k_proj rows < hidden_dim, store actual kv_dim
    w->kv_dim = w->layers[0].wk.rows;  // actual KV projection size
    printf("KV dim (GQA): %d  (Q dim: %d)\n", w->kv_dim, hd);

    size_t fm, tm;
    cudaMemGetInfo(&fm, &tm);
    printf("GPU: %.0f MB used / %.0f MB total\n\n",
           (tm-fm)/1e6, tm/1e6);
    return w;
}

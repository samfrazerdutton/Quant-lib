#include "weights.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// ── Helpers ───────────────────────────────────────────────────────────────────
static void alloc_tensor(Tensor* t, int rows, int cols, const char* name) {
    t->rows = rows;
    t->cols = cols;
    strncpy(t->name, name, 63);
    CUDA_CHECK(cudaMalloc(&t->data, (size_t)rows * cols * sizeof(float)));
}

static void free_tensor(Tensor* t) {
    if (t->data) { cudaFree(t->data); t->data = NULL; }
}

// Xavier uniform init on CPU then upload
static void init_random(Tensor* t) {
    int n = t->rows * t->cols;
    float* h = (float*)malloc(n * sizeof(float));
    float scale = sqrtf(6.0f / (t->rows + t->cols));
    for (int i = 0; i < n; i++)
        h[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    CUDA_CHECK(cudaMemcpy(t->data, h, n*sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h);
}

// ── Public API ────────────────────────────────────────────────────────────────
ModelWeights* weights_alloc_random(int n_layers, int hidden_dim,
                                   int n_heads, int ffn_dim,
                                   int vocab_size) {
    ModelWeights* w = (ModelWeights*)calloc(1, sizeof(ModelWeights));
    w->n_layers   = n_layers;
    w->hidden_dim = hidden_dim;
    w->n_heads    = n_heads;
    w->head_dim   = hidden_dim / n_heads;
    w->ffn_dim    = ffn_dim;
    w->vocab_size = vocab_size;
    w->layers     = (LayerWeights*)calloc(n_layers, sizeof(LayerWeights));

    int hd = hidden_dim;
    int kd = n_heads * w->head_dim;   // = hidden_dim for standard MHA

    size_t total_bytes = 0;

    for (int l = 0; l < n_layers; l++) {
        LayerWeights* lw = &w->layers[l];
        alloc_tensor(&lw->wq,       kd, hd, "wq");
        alloc_tensor(&lw->wk,       kd, hd, "wk");
        alloc_tensor(&lw->wv,       kd, hd, "wv");
        alloc_tensor(&lw->wo,       hd, kd, "wo");
        alloc_tensor(&lw->w1,  ffn_dim, hd, "w1");
        alloc_tensor(&lw->w2,       hd, ffn_dim, "w2");
        alloc_tensor(&lw->w3,  ffn_dim, hd, "w3");
        alloc_tensor(&lw->rms_attn, 1,  hd, "rms_attn");
        alloc_tensor(&lw->rms_ffn,  1,  hd, "rms_ffn");

        init_random(&lw->wq); init_random(&lw->wk);
        init_random(&lw->wv); init_random(&lw->wo);
        init_random(&lw->w1); init_random(&lw->w2);
        init_random(&lw->w3);

        // RMS norm weights init to 1
        float* ones = (float*)malloc(hd * sizeof(float));
        for (int i = 0; i < hd; i++) ones[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(lw->rms_attn.data, ones,
                              hd*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lw->rms_ffn.data,  ones,
                              hd*sizeof(float), cudaMemcpyHostToDevice));
        free(ones);

        total_bytes += (size_t)(4*kd*hd + ffn_dim*hd*3 + 2*hd) * sizeof(float);
        // approximate — exact sum computed below
    }

    alloc_tensor(&w->tok_embeddings, vocab_size, hd, "tok_embed");
    alloc_tensor(&w->norm,           1,          hd, "norm");
    alloc_tensor(&w->output,         vocab_size, hd, "lm_head");
    init_random(&w->tok_embeddings);
    init_random(&w->output);

    // Print memory usage
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Weights loaded. GPU memory: %.0f MB used / %.0f MB total\n",
           (total_mem - free_mem) / 1e6, total_mem / 1e6);

    return w;
}

void weights_free(ModelWeights* w) {
    for (int l = 0; l < w->n_layers; l++) {
        LayerWeights* lw = &w->layers[l];
        free_tensor(&lw->wq); free_tensor(&lw->wk);
        free_tensor(&lw->wv); free_tensor(&lw->wo);
        free_tensor(&lw->w1); free_tensor(&lw->w2);
        free_tensor(&lw->w3); free_tensor(&lw->rms_attn);
        free_tensor(&lw->rms_ffn);
    }
    free_tensor(&w->tok_embeddings);
    free_tensor(&w->norm);
    free_tensor(&w->output);
    free(w->layers);
    free(w);
}

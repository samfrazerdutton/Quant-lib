#pragma once
#include "weights.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

ModelWeights* load_gguf_weights(const char* filename, int n_layers, int hidden_dim, int n_heads, int head_dim, int ffn_dim, int vocab_size);

#ifdef __cplusplus
}
#endif

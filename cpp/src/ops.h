// The three basic building blocks. Deliberately naive: correctness first.
#pragma once
#include <cmath>
#include <cstring>

// Copy row `token` of the embedding table into out.  table: [vocab, dim]
inline void embed_lookup(float *out, const float *table, int token, int dim) {
    std::memcpy(out, table + (size_t)token * dim, sizeof(float) * dim);
}

// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
inline void rmsnorm(float *out, const float *x, const float *weight, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

// y = W x.  W is [rows, cols] row-major, x is [cols], y is [rows].
// Each output value is the dot product of one row of W with x.
inline void matvec(float *y, const float *W, const float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * cols;
        float acc = 0.f;
        for (int c = 0; c < cols; c++) acc += w[c] * x[c];
        y[r] = acc;
    }
}

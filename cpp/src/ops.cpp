#include "ops.h"

#include <cmath>

void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

void matvec(float* out, const float* W, const float* x, int out_dim, int in_dim) {
    for (int r = 0; r < out_dim; r++) {
        const float* row = W + (size_t)r * in_dim;
        float acc = 0.0f;
        for (int c = 0; c < in_dim; c++) acc += row[c] * x[c];
        out[r] = acc;
    }
}

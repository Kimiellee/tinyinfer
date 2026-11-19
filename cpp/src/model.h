// Loads the binary file written by python/export_weights.py.
// The file is mmap'ed, so "loading" is instant: the weight pointers below
// point straight into the file's bytes (nothing is copied).
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Must match HEADER_FMT in export_weights.py: "<4si" + "8i" + "2f" + "2i"
struct FileHeader {
    char magic[4];
    int32_t version;
    int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim, vocab_size, max_seq_len;
    float rope_theta, rms_eps;
    int32_t tied;   // 1 = output layer reuses the embedding matrix
    int32_t dtype;  // 0 = fp32, 1 = fp16
};
static_assert(sizeof(FileHeader) == 56, "FileHeader must match the Python struct layout");

constexpr size_t HEADER_SIZE = 256;

struct LayerWeights {
    const float *attn_norm;  // [dim]
    const float *wq;         // [n_heads*head_dim, dim]
    const float *wk;         // [n_kv_heads*head_dim, dim]
    const float *wv;         // [n_kv_heads*head_dim, dim]
    const float *wo;         // [dim, n_heads*head_dim]
    const float *ffn_norm;   // [dim]
    const float *w_gate;     // [hidden_dim, dim]
    const float *w_up;       // [hidden_dim, dim]
    const float *w_down;     // [dim, hidden_dim]
};

struct Model {
    FileHeader cfg{};
    void *map = nullptr;
    size_t map_size = 0;

    const float *tok_embed = nullptr;  // [vocab_size, dim]
    std::vector<LayerWeights> layers;
    const float *final_norm = nullptr; // [dim]
    const float *lm_head = nullptr;    // [vocab_size, dim] (== tok_embed when tied)

    bool load(const std::string &path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { perror("open"); return false; }
        struct stat st;
        fstat(fd, &st);
        map_size = (size_t)st.st_size;
        map = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map == MAP_FAILED) { perror("mmap"); map = nullptr; return false; }

        const char *base = (const char *)map;
        cfg = *(const FileHeader *)base;
        if (std::string(cfg.magic, 4) != "TINF") { fprintf(stderr, "bad magic\n"); return false; }
        if (cfg.dtype != 0) { fprintf(stderr, "Day 2 supports fp32 only (re-export without --dtype fp16)\n"); return false; }

        const size_t dim = cfg.dim, hid = cfg.hidden_dim, hd = cfg.head_dim;
        const size_t q_dim = (size_t)cfg.n_heads * hd, kv_dim = (size_t)cfg.n_kv_heads * hd;

        // Walk through the file in exactly the order of tensor_order() in Python.
        const float *p = (const float *)(base + HEADER_SIZE);
        auto take = [&](size_t n) { const float *r = p; p += n; return r; };

        tok_embed = take((size_t)cfg.vocab_size * dim);
        layers.resize(cfg.n_layers);
        for (auto &L : layers) {
            L.attn_norm = take(dim);
            L.wq = take(q_dim * dim);
            L.wk = take(kv_dim * dim);
            L.wv = take(kv_dim * dim);
            L.wo = take(dim * q_dim);
            L.ffn_norm = take(dim);
            L.w_gate = take(hid * dim);
            L.w_up = take(hid * dim);
            L.w_down = take(dim * hid);
        }
        final_norm = take(dim);
        lm_head = cfg.tied ? tok_embed : take((size_t)cfg.vocab_size * dim);

        size_t used = (size_t)((const char *)p - base);
        if (used != map_size) {
            fprintf(stderr, "size mismatch: walked %zu bytes, file has %zu\n", used, map_size);
            return false;
        }
        return true;
    }

    ~Model() { if (map) munmap(map, map_size); }
};

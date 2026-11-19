// Day 2: load the model, run embedding -> RMSNorm -> q/k/v projections for layer 0,
// and dump the results so python/compare.py can check them against Hugging Face.
//
// Usage: ./tinyinfer <model.bin> <prompt_ids.txt> <dump_dir>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "model.h"
#include "ops.h"

static void write_bin(const std::string &path, const std::vector<float> &v) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *)v.data(), v.size() * sizeof(float));
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " <model.bin> <prompt_ids.txt> <dump_dir>\n";
        return 1;
    }
    Model m;
    if (!m.load(argv[1])) return 1;
    const FileHeader &c = m.cfg;
    printf("dim=%d hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d tied=%d\n",
           c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab_size, c.tied);

    std::vector<int> ids;
    {
        std::ifstream f(argv[2]);
        int t;
        while (f >> t) ids.push_back(t);
    }
    printf("prompt has %zu tokens\n", ids.size());

    const int q_dim = c.n_heads * c.head_dim, kv_dim = c.n_kv_heads * c.head_dim;
    std::vector<float> x(c.dim), xn(c.dim), q(q_dim), k(kv_dim), v(kv_dim);
    std::vector<float> d_embed, d_norm, d_q, d_k, d_v;
    const LayerWeights &L0 = m.layers[0];

    for (int t : ids) {  // one token position at a time
        embed_lookup(x.data(), m.tok_embed, t, c.dim);
        rmsnorm(xn.data(), x.data(), L0.attn_norm, c.dim, c.rms_eps);
        matvec(q.data(), L0.wq, xn.data(), q_dim, c.dim);
        matvec(k.data(), L0.wk, xn.data(), kv_dim, c.dim);
        matvec(v.data(), L0.wv, xn.data(), kv_dim, c.dim);

        d_embed.insert(d_embed.end(), x.begin(), x.end());
        d_norm.insert(d_norm.end(), xn.begin(), xn.end());
        d_q.insert(d_q.end(), q.begin(), q.end());
        d_k.insert(d_k.end(), k.begin(), k.end());
        d_v.insert(d_v.end(), v.begin(), v.end());
    }

    std::string dir = argv[3];
    std::filesystem::create_directories(dir);
    write_bin(dir + "/embed.bin", d_embed);
    write_bin(dir + "/l0_attn_norm.bin", d_norm);
    write_bin(dir + "/l0_q_proj.bin", d_q);
    write_bin(dir + "/l0_k_proj.bin", d_k);
    write_bin(dir + "/l0_v_proj.bin", d_v);
    printf("wrote dumps to %s\n", dir.c_str());
    return 0;
}

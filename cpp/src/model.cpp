#include "model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

static const size_t HEADER_SIZE = 256;  // must match export_weights.py

Model::Model(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed");
    size_ = (size_t)st.st_size;
    base_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    parse_header();
    map_weights();
}

Model::~Model() {
    if (base_ && base_ != MAP_FAILED) munmap(base_, size_);
    if (fd_ >= 0) close(fd_);
}

void Model::parse_header() {
    const char* p = (const char*)base_;
    auto read = [&](auto& dst) {
        std::memcpy(&dst, p, sizeof(dst));
        p += sizeof(dst);
    };
    if (size_ < HEADER_SIZE || std::memcmp(p, "TINF", 4) != 0)
        throw std::runtime_error("bad magic: not a tinyinfer model file");
    p += 4;
    int version;
    read(version);
    if (version != 1) throw std::runtime_error("unsupported file version");
    // Same field order as HEADER_FMT in export_weights.py
    read(cfg.dim); read(cfg.hidden_dim); read(cfg.n_layers); read(cfg.n_heads);
    read(cfg.n_kv_heads); read(cfg.head_dim); read(cfg.vocab_size); read(cfg.max_seq_len);
    read(cfg.rope_theta); read(cfg.rms_eps);
    read(cfg.tied); read(cfg.dtype);
    if (cfg.dtype != 0) throw std::runtime_error("only fp32 files supported for now");
}

void Model::map_weights() {
    const char* start = (const char*)base_ + HEADER_SIZE;
    const float* p = (const float*)start;
    auto take = [&](size_t n) { const float* r = p; p += n; return r; };

    const size_t dim = cfg.dim, hid = cfg.hidden_dim;
    const size_t qd = (size_t)cfg.n_heads * cfg.head_dim;
    const size_t kvd = (size_t)cfg.n_kv_heads * cfg.head_dim;

    // Same order as tensor_order() in export_weights.py
    w.tok_emb = take((size_t)cfg.vocab_size * dim);
    w.layers.resize(cfg.n_layers);
    for (auto& l : w.layers) {
        l.attn_norm = take(dim);
        l.wq = take(qd * dim);
        l.wk = take(kvd * dim);
        l.wv = take(kvd * dim);
        l.wo = take(dim * qd);
        l.ffn_norm = take(dim);
        l.w_gate = take(hid * dim);
        l.w_up = take(hid * dim);
        l.w_down = take(dim * hid);
    }
    w.final_norm = take(dim);
    w.lm_head = cfg.tied ? w.tok_emb : take((size_t)cfg.vocab_size * dim);

    size_t used = HEADER_SIZE + (size_t)((const char*)p - start);
    if (used != size_)
        throw std::runtime_error("file size mismatch: expected " + std::to_string(used) +
                                 " bytes, got " + std::to_string(size_));
}

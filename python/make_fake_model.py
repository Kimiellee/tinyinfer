"""Create a tiny random model in the tinyinfer format + numpy reference activations.

Lets you test the C++ loader and ops WITHOUT downloading anything.
"""
import argparse
import struct
from pathlib import Path

import numpy as np

HEADER_FMT = "<4si" + "8i" + "2f" + "2i"   # keep in sync with export_weights.py
HEADER_SIZE = 256

C = dict(dim=16, hidden_dim=32, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=4,
         vocab_size=50, max_seq_len=64, rope_theta=10000.0, rms_eps=1e-5, tied=1)
IDS = [3, 17, 4, 42, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fake")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "ref").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    r = lambda *s: (rng.standard_normal(s) * 0.1).astype(np.float32)
    dim, hid, hd = C["dim"], C["hidden_dim"], C["head_dim"]
    qd, kvd = C["n_heads"] * hd, C["n_kv_heads"] * hd

    tensors = {"emb": r(C["vocab_size"], dim)}
    layers = []
    for _ in range(C["n_layers"]):
        layers.append(dict(
            attn_norm=(1 + r(dim)), wq=r(qd, dim), wk=r(kvd, dim), wv=r(kvd, dim),
            wo=r(dim, qd), ffn_norm=(1 + r(dim)), gate=r(hid, dim), up=r(hid, dim),
            down=r(dim, hid)))
    final_norm = 1 + r(dim)

    header = struct.pack(
        HEADER_FMT, b"TINF", 1, C["dim"], C["hidden_dim"], C["n_layers"], C["n_heads"],
        C["n_kv_heads"], C["head_dim"], C["vocab_size"], C["max_seq_len"],
        C["rope_theta"], C["rms_eps"], C["tied"], 0)
    with open(out / "model.bin", "wb") as f:
        f.write(header.ljust(HEADER_SIZE, b"\0"))
        f.write(tensors["emb"].tobytes())
        for l in layers:
            for k in ["attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "gate", "up", "down"]:
                f.write(l[k].tobytes())
        f.write(final_norm.tobytes())

    # numpy reference for the Day 2 tensors
    l0 = layers[0]
    x = tensors["emb"][IDS]
    normed = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + C["rms_eps"]) * l0["attn_norm"]
    ref = {"embed": x, "l0_attn_norm": normed.astype(np.float32),
           "l0_q_proj": normed @ l0["wq"].T, "l0_k_proj": normed @ l0["wk"].T,
           "l0_v_proj": normed @ l0["wv"].T}
    for name, arr in ref.items():
        np.save(out / "ref" / f"{name}.npy", arr.astype(np.float32))
    print(f"wrote {out}/model.bin and {out}/ref/*.npy; token ids: {' '.join(map(str, IDS))}")


if __name__ == "__main__":
    main()

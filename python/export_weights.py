"""Export a Llama-style HF model (e.g. SmolLM2) to the tinyinfer binary format.

File layout (little endian):
  [0, 256)   header (struct below, zero padded)
  [256, ...) tensors, row-major, contiguous, in the order given by tensor_order()

Header fields:
  magic "TINF", version
  int32: dim, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim, vocab_size, max_seq_len
  float32: rope_theta, rms_eps
  int32: tied_embeddings (1 = lm_head shares tok embeddings, no lm_head tensor stored)
  int32: dtype (0 = fp32, 1 = fp16)

A <out>.json manifest (name, shape, byte offset) is written next to the binary
so you can sanity check your C++ loader's offsets.
"""
import argparse
import json
import struct
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

MAGIC = b"TINF"
VERSION = 1
HEADER_SIZE = 256
HEADER_FMT = "<4si" + "8i" + "2f" + "2i"
DTYPES = {"fp32": (0, torch.float32), "fp16": (1, torch.float16)}


def tensor_order(n_layers, tied):
    names = ["model.embed_tokens.weight"]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        names += [
            p + "input_layernorm.weight",
            p + "self_attn.q_proj.weight",
            p + "self_attn.k_proj.weight",
            p + "self_attn.v_proj.weight",
            p + "self_attn.o_proj.weight",
            p + "post_attention_layernorm.weight",
            p + "mlp.gate_proj.weight",
            p + "mlp.up_proj.weight",
            p + "mlp.down_proj.weight",
        ]
    names.append("model.norm.weight")
    if not tied:
        names.append("lm_head.weight")
    return names


def expected_shape(name, c):
    dim, hid, hd = c["dim"], c["hidden_dim"], c["head_dim"]
    if name == "model.embed_tokens.weight" or name == "lm_head.weight":
        return (c["vocab_size"], dim)
    if name.endswith("layernorm.weight") or name == "model.norm.weight":
        return (dim,)
    if name.endswith("q_proj.weight"):
        return (c["n_heads"] * hd, dim)
    if name.endswith("k_proj.weight") or name.endswith("v_proj.weight"):
        return (c["n_kv_heads"] * hd, dim)
    if name.endswith("o_proj.weight"):
        return (dim, c["n_heads"] * hd)
    if name.endswith("gate_proj.weight") or name.endswith("up_proj.weight"):
        return (hid, dim)
    if name.endswith("down_proj.weight"):
        return (dim, hid)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--out", default="models/smollm2-360m.bin")
    ap.add_argument("--dtype", choices=DTYPES, default="fp32")
    args = ap.parse_args()

    path = Path(snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"]))
    hf = json.loads((path / "config.json").read_text())

    c = dict(
        dim=hf["hidden_size"],
        hidden_dim=hf["intermediate_size"],
        n_layers=hf["num_hidden_layers"],
        n_heads=hf["num_attention_heads"],
        n_kv_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
        vocab_size=hf["vocab_size"],
        max_seq_len=hf.get("max_position_embeddings", 2048),
        rope_theta=float(hf.get("rope_theta", 10000.0)),
        rms_eps=float(hf.get("rms_norm_eps", 1e-5)),
        tied=int(hf.get("tie_word_embeddings", False)),
    )
    c["head_dim"] = hf.get("head_dim", c["dim"] // c["n_heads"])
    print("config:", c)

    # Map tensor name -> safetensors file
    index = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                index[k] = f

    dtype_code, torch_dtype = DTYPES[args.dtype]
    header = struct.pack(
        HEADER_FMT, MAGIC, VERSION,
        c["dim"], c["hidden_dim"], c["n_layers"], c["n_heads"], c["n_kv_heads"],
        c["head_dim"], c["vocab_size"], c["max_seq_len"],
        c["rope_theta"], c["rms_eps"], c["tied"], dtype_code,
    )
    assert len(header) <= HEADER_SIZE

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with open(out, "wb") as fh:
        fh.write(header.ljust(HEADER_SIZE, b"\0"))
        for name in tensor_order(c["n_layers"], c["tied"]):
            with safe_open(index[name], framework="pt") as sf:
                t = sf.get_tensor(name)
            shape = tuple(t.shape)
            assert shape == expected_shape(name, c), f"{name}: {shape} != {expected_shape(name, c)}"
            offset = fh.tell()
            fh.write(t.to(torch.float32).to(torch_dtype).contiguous().numpy().tobytes())
            manifest.append({"name": name, "shape": list(shape), "offset": offset})

    out.with_suffix(".json").write_text(json.dumps({"config": c, "tensors": manifest}, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {len(manifest)} tensors")


if __name__ == "__main__":
    main()

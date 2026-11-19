"""Run the HF model in fp32 and dump intermediate activations + greedy generation.

These files are the test oracle for the C++ engine. Activations are saved as
float32 .npy with the batch dim removed: shape (seq_len, features).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="ref")
    ap.add_argument("--gen-tokens", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()

    acts = {}

    def hook(name):
        def fn(module, inputs, output):
            acts[name] = first(output).detach()[0].float().numpy()
        return fn

    m = model.model
    l0, last = m.layers[0], m.layers[-1]
    hooks = [
        m.embed_tokens.register_forward_hook(hook("embed")),
        l0.input_layernorm.register_forward_hook(hook("l0_attn_norm")),
        l0.self_attn.q_proj.register_forward_hook(hook("l0_q_proj")),
        l0.self_attn.k_proj.register_forward_hook(hook("l0_k_proj")),
        l0.self_attn.v_proj.register_forward_hook(hook("l0_v_proj")),
        l0.self_attn.register_forward_hook(hook("l0_attn_out")),   # after o_proj
        l0.register_forward_hook(hook("l0_out")),                   # full layer 0 output
        l0.mlp.register_forward_hook(hook("l0_mlp_out")),
        last.register_forward_hook(hook("last_layer_out")),         # before final norm
        m.norm.register_forward_hook(hook("final_norm")),
        model.lm_head.register_forward_hook(hook("logits")),
    ]

    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    for name, arr in acts.items():
        np.save(out / f"{name}.npy", arr.astype(np.float32))
        print(f"{name:16s} {arr.shape}")

    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False)
    prompt_ids = ids[0].tolist()
    gen_ids = gen[0].tolist()[len(prompt_ids):]
    (out / "generation.json").write_text(json.dumps({
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "greedy_ids": gen_ids,
        "greedy_text": tok.decode(gen_ids),
    }, indent=1))
    (out / "prompt_ids.txt").write_text(" ".join(map(str, prompt_ids)))
    print("prompt ids:", prompt_ids)
    print("greedy    :", repr(args.prompt + tok.decode(gen_ids)))


if __name__ == "__main__":
    main()

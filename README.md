# tinyinfer

From-scratch C++ LLM inference engine (CPU reference -> CUDA, INT8/INT4 quantization),
validated layer by layer against Hugging Face. Target model: SmolLM2-360M.

## Day 1: export + reference

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python python/export_weights.py --out models/smollm2-360m.bin     # add --dtype fp16 for half size
python python/reference.py --prompt "The capital of France is"    # writes ref/*.npy + ref/generation.json
```

## Notes for the C++ side

- Weights are stored exactly as in HF: linear weights are `[out_features, in_features]`
  row-major, so `y = W x` is a dot product of each row of W with x.
- **RoPE uses the HF "rotate_half" convention**: element `i` is paired with element
  `i + head_dim/2`, NOT with `i + 1` (that's the interleaved variant used by llama2.c).
- Tensor order is defined by `tensor_order()` in `export_weights.py`; offsets are in the `.json` manifest.
- SmolLM2 ties lm_head to the embedding matrix (`tied=1`), so there is no separate lm_head tensor.

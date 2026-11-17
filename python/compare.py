"""Compare activations dumped by the C++ engine against the HF reference.

C++ side: write each tensor as raw little-endian float32 to <cpp_dir>/<name>.bin
using the same names as ref/*.npy (embed, l0_attn_norm, l0_out, logits, ...).

Usage: python python/compare.py --ref ref --cpp cpp_dump
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ref")
    ap.add_argument("--cpp", default="cpp_dump")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    print(f"{'tensor':16s} {'max_abs':>10s} {'mean_abs':>10s}  status")
    for npy in sorted(Path(args.ref).glob("*.npy")):
        binf = Path(args.cpp) / f"{npy.stem}.bin"
        if not binf.exists():
            print(f"{npy.stem:16s} {'-':>10s} {'-':>10s}  (no C++ dump)")
            continue
        ref = np.load(npy)
        mine = np.fromfile(binf, dtype="<f4")
        if mine.size != ref.size:
            print(f"{npy.stem:16s} size mismatch: ref {ref.size}, cpp {mine.size}")
            continue
        d = np.abs(mine.reshape(ref.shape) - ref)
        status = "OK" if d.max() < args.tol else "MISMATCH"
        print(f"{npy.stem:16s} {d.max():10.2e} {d.mean():10.2e}  {status}")


if __name__ == "__main__":
    main()

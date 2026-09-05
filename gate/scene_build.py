"""Consolidates per-clip vision-tower feature caches into per-split arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA = Path("runs/gate/data")
TOWER = Path("runs/gate/tower")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "benchmark"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        ids = [str(s) for s in np.load(DATA / f"{split}.npz")["ids"]]
        keep, V, G = [], [], []
        for cid in ids:
            p = TOWER / f"{cid}.npy"
            if not p.is_file():
                continue
            f = np.load(p)
            V.append(f[:, 16]); G.append(f[:, :16].astype(np.float32).mean(1).astype(np.float16))
            keep.append(cid)
        np.savez(a.out / f"tower_{split}.npz", ids=np.asarray(keep),
                 V=np.stack(V), G=np.stack(G))
        print(f"[{split}] {len(keep)}/{len(ids)} clips", flush=True)

if __name__ == "__main__":
    main()

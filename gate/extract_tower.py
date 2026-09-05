"""Extracts frozen vision-tower features for gate training and inference."""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generator import data

TOWER = "checkpoints/student-dpo-v0/model"

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test", "benchmark"])
    ap.add_argument("--out-dir", type=Path, default=Path("runs/gate"))
    ap.add_argument("--model-path", default=TOWER)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=3, help="frames per tower call")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse", action="store_true",
                    help="walk the shard backwards (a second array meets the first)")
    a = ap.parse_args()

    entries: dict[str, dict] = {}
    for split in a.splits:
        for e in data.load_manifest(split):
            entries.setdefault(e["id"], e)
    todo = [e for i, (_, e) in enumerate(sorted(entries.items()))
            if i % a.num_shards == a.shard]
    out = a.out_dir / "tower"
    out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in todo if not (out / f"{e['id']}.npy").is_file()]
    if a.reverse:
        todo = todo[::-1]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[tower {a.shard}/{a.num_shards}] {len(todo)} clips", flush=True)
    if not todo:
        print(f"TOWER_SHARD_{a.shard}_OK", flush=True)
        return

    from gate.encoders import FastVLMTowerEncoder
    enc = FastVLMTowerEncoder(a.model_path, device="cuda")

    t0, n_err = time.time(), 0
    for i, e in enumerate(todo, 1):
        if (out / f"{e['id']}.npy").is_file():
            continue
        try:
            paths = data.frame_paths(e["frames_dir"])
            if not paths:
                raise FileNotFoundError("missing frames")
            imgs = [Image.open(p).convert("RGB") for p in paths]
            feats = np.concatenate(
                [enc.encode_frames(imgs[k:k + a.chunk])
                 for k in range(0, len(imgs), a.chunk)], axis=0)
            tmp = out / f".{e['id']}.tmp.npy"
            np.save(tmp, feats.astype(np.float16))
            tmp.rename(out / f"{e['id']}.npy")
        except Exception:
            n_err += 1
            print(f"ERR {e['id']}: {traceback.format_exc(limit=2)}", flush=True)
        if i % 50 == 0 or i == len(todo):
            print(f"[tower {a.shard}] {i}/{len(todo)} "
                  f"{i / (time.time() - t0):.2f} clip/s errs={n_err}", flush=True)
    print(f"TOWER_SHARD_{a.shard}_OK", flush=True)

if __name__ == "__main__":
    main()

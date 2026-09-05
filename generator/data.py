"""Record and label loading helpers."""
from __future__ import annotations

import json
import random
from pathlib import Path

LABEL_ROOT = Path("runs/relabel/v1sp6_2026-08-11/full")
SPLIT_ROOT = Path("runs/relabel/splits_2026-08-10")
EVAL30_SEED = 20260812

def load_labels(split: str) -> dict[str, dict]:

    out: dict[str, dict] = {}
    for shard in sorted((LABEL_ROOT / split).glob("shard-*.jsonl")):
        for line in shard.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("parse_ok") and isinstance(r.get("parsed"), dict):
                out[r["id"]] = r["parsed"]
    return out

def load_manifest(split: str) -> list[dict]:

    return json.loads((SPLIT_ROOT / f"{split}.json").read_text())

def frame_paths(frames_dir: str | Path) -> list[Path]:
    d = Path(frames_dir)
    paths = [d / f"{k}.jpg" for k in range(9)]
    return paths if all(p.is_file() for p in paths) else []

def eval30_entries() -> list[dict]:

    per_source = {"egoblind": 15, "wad": 10, "wrd": 5}
    bench = sorted(load_manifest("benchmark"), key=lambda e: e["id"])
    rng = random.Random(EVAL30_SEED)
    picked: list[dict] = []
    for src, n in per_source.items():
        pool = [e for e in bench if e["source"] == src]
        picked.extend(rng.sample(pool, n))
    return sorted(picked, key=lambda e: e["id"])

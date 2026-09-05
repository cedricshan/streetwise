"""Builds train/val/test JSONL from perception records and labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator import data
from generator.prompt import target_alert, target_json

def build_split(split: str, prep_dir: Path, out_path: Path,
                nocmc: bool = False) -> tuple[int, int]:
    labels = data.load_labels(split)
    manifest = data.load_manifest(split)
    n_ok = n_skip = 0
    with out_path.open("w") as fh:
        for e in manifest:
            lab = labels.get(e["id"])
            rec_path = prep_dir / "records" / f"{e['id']}.json"
            if lab is None or not rec_path.is_file():
                n_skip += 1
                continue
            rec = json.loads(rec_path.read_text())
            if "error" in rec:
                n_skip += 1
                continue
            if nocmc:
                rec = {**rec, **rec["nocmc"]}
            fh.write(json.dumps({
                "id": e["id"], "source": e["source"],
                "image": rec["marked_image"],
                "image_raw": str(Path(e["frames_dir"]) / "8.jpg"),
                "detections": rec["detections"],
                "motion": rec["motion"],
                "boxes": [t["box"] for t in rec.get("tracks", [])[:8]],
                "target_json": target_json(lab),
                "target_alert": target_alert(lab),
            }, ensure_ascii=False) + "\n")
            n_ok += 1
    return n_ok, n_skip

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-dir", type=Path,
                    default=Path("runs/prep_v1sp6"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/sft_v1sp6"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--nocmc", action="store_true",
                    help="ladder rung B: take detections/motion/marked image "
                         "from the record's ego-free arm")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        n_ok, n_skip = build_split(split, args.prep_dir,
                                   args.out_dir / f"{split}.jsonl",
                                   nocmc=args.nocmc)
        print(f"[build] {split}: {n_ok} rows ({n_skip} skipped)", flush=True)

if __name__ == "__main__":
    main()

"""Perception front-end: detection, depth, camera-motion compensation, tracking, ego-motion, and text serialization for each clip."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generator import data
from perception import serialize, track, vocab
from perception.depth import DepthEstimator, det_position
from perception.detect import Detector
from perception.cmc import EgoMotionEstimator, estimate_cmc

VERSIONS = {"detector": "yoloe-26x-seg", "depth": "moge-2-vitl-normal",
            "cmc": "lk-orb-v1", "prep": "v1sp6-prep-1"}

def process_clip(entry: dict, det: Detector, dep: DepthEstimator,
                 out_dir: Path) -> dict:
    paths = data.frame_paths(entry["frames_dir"])
    if not paths:
        return {"id": entry["id"], "source": entry["source"],
                "error": "missing frames"}
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    grays = [np.asarray(Image.open(p).convert("L")) for p in paths]
    H, W = imgs[0].shape[:2]

    per_frame_dets = det.detect_clip(paths)
    per_frame_pos = []
    for img, dets in zip(imgs, per_frame_dets):
        d = dep.infer(img)
        per_frame_pos.append([det_position(d["points"], d["mask"], dd.box)
                              for dd in dets])

    ego = EgoMotionEstimator(frame_width=W)
    cmcs = []
    for k in range(8):
        boxes = [d.box for d in per_frame_dets[k]] + \
                [d.box for d in per_frame_dets[k + 1]]
        c = estimate_cmc(grays[k], grays[k + 1], boxes)
        cmcs.append(c)
        ego.add(c, t=(k + 1) * track.DT, dt=track.DT)
    ego_state = ego.state(now=8 * track.DT)

    tracks = track.associate(per_frame_dets, per_frame_pos, cmcs)
    physics, ego_speed = track.analyze(tracks, ego_state.forward, (W, H))

    final = Image.open(paths[-1]).convert("RGB")
    marked_path = out_dir / "marked" / f"{entry['id']}.jpg"
    serialize.draw_marks(final, physics).save(marked_path, quality=92)

    tracks_b = track.associate(per_frame_dets, per_frame_pos, cmcs,
                               use_cmc=False)
    physics_b, _ = track.analyze(tracks_b, False, (W, H), ego_off=True)
    marked_b_path = out_dir / "marked_nocmc" / f"{entry['id']}.jpg"
    serialize.draw_marks(final, physics_b).save(marked_b_path, quality=92)

    return {
        "id": entry["id"], "source": entry["source"],
        "frames_dir": entry["frames_dir"],
        "detections": serialize.objects_text(physics),
        "motion": serialize.motion_text(physics, ego_state.describe(),
                                        ego_speed),
        "ego": {"desc": ego_state.describe(),
                "speed": round(ego_speed, 2),
                "quality": round(ego_state.quality, 3)},
        "tracks": physics,
        "marked_image": str(marked_path),
        "nocmc": {
            "detections": serialize.objects_text(physics_b),
            "motion": serialize.motion_text(physics_b, None, 0.0),
            "tracks": physics_b,
            "marked_image": str(marked_b_path),
        },
        "raw": {
            "dets": [[{"label": d.label, "conf": round(d.conf, 3),
                       "box": [round(v, 1) for v in d.box]} for d in fr]
                     for fr in per_frame_dets],
            "xyz": [[None if p is None else [round(v, 3) for v in p]
                     for p in fr] for fr in per_frame_pos],
            "cmc": [{"m": [round(v, 6) for v in c.matrix.ravel().tolist()],
                     "q": round(c.quality, 3), "method": c.method}
                    for c in cmcs],
        },
        "versions": VERSIONS,
    }

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test", "benchmark"])
    ap.add_argument("--manifest", type=Path, default=None,
                    help="JSON list of {id, source, frames_dir} — overrides --splits")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("runs/prep_v1sp6"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", type=str, default=None,
                    help="comma-separated clip ids (smoke tests)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    entries: dict[str, dict] = {}
    if args.manifest:
        for e in json.loads(args.manifest.read_text()):
            entries.setdefault(e["id"], e)
    else:
        for split in args.splits:
            for e in data.load_manifest(split):
                entries.setdefault(e["id"], e)
    todo = [e for i, (cid, e) in enumerate(sorted(entries.items()))
            if i % args.num_shards == args.shard]
    if args.ids:
        want = set(args.ids.split(","))
        todo = [e for e in entries.values() if e["id"] in want]
    if args.limit:
        todo = todo[:args.limit]

    for sub in ("records", "marked", "marked_nocmc"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        todo = [e for e in todo
                if not (args.out_dir / "records" / f"{e['id']}.json").is_file()]
    print(f"[prep shard {args.shard}/{args.num_shards}] {len(todo)} clips",
          flush=True)
    if not todo:
        return

    det = Detector(vocab.CLASSES, imgsz=args.imgsz)
    dep = DepthEstimator()
    t0 = time.time()
    n_err = 0
    for i, e in enumerate(todo, 1):
        try:
            rec = process_clip(e, det, dep, args.out_dir)
        except Exception:
            rec = {"id": e["id"], "source": e.get("source"),
                   "error": traceback.format_exc(limit=3)}
        if "error" in rec:
            n_err += 1
        (args.out_dir / "records" / f"{e['id']}.json").write_text(
            json.dumps(rec, ensure_ascii=False))
        if i % 25 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            print(f"[prep {args.shard}] {i}/{len(todo)} "
                  f"{rate:.2f} clip/s errs={n_err}", flush=True)
    print(f"[prep {args.shard}] DONE {len(todo)} clips errs={n_err}",
          flush=True)

if __name__ == "__main__":
    main()

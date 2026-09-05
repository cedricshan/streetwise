"""Generates JSON alerts for evaluation clips from a trained checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator import data
from generator.prompt import (build_frames_text, build_user_text,
                          parse_model_json)
from generator.sft import generate_one, load_image, resolve_model_path

def load_model(ckpt: Path, base_model: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if (ckpt / "model").is_dir():
        src = str(ckpt / "model")
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True)
    elif (ckpt / "adapter").is_dir():
        from peft import PeftModel
        base = resolve_model_path(base_model)
        tok = AutoTokenizer.from_pretrained(str(ckpt / "adapter"),
                                            trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True)
        model = PeftModel.from_pretrained(model, str(ckpt / "adapter"))
        model = model.merge_and_unload()
    elif base_model:
        base = resolve_model_path(base_model)
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True)
    else:
        raise SystemExit(f"no model found under {ckpt}")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--arm", choices=["json", "alert"], default="json")
    ap.add_argument("--content-edge", type=int, default=None)
    ap.add_argument("--no-marks", action="store_true",
                    help="use the raw final frame (no numbered boxes)")
    ap.add_argument("--pixelate-boxes", action="store_true",
                    help="low-pass detected regions (pixelate arm)")
    ap.add_argument("--blank-image", action="store_true",
                    help="text-only arm: replace the frame with a black image "
                         "of the same size (the visual token count is unchanged)")
    ap.add_argument("--drop-text", choices=["motion", "detections", "both"],
                    default=None,
                    help="input ablation: blank one or both text channels "
                         "(the slot is filled with 'none', the template is "
                         "unchanged so the prompt stays byte-compatible)")
    ap.add_argument("--frames", type=int, default=0,
                    help="ladder rung A: feed N raw clip frames and no "
                         "perception text (0 = normal single marked frame)")
    ap.add_argument("--nocmc", action="store_true",
                    help="ladder rung B: use the record's ego-free arm "
                         "(raw-IoU association, camera-relative motion tags, "
                         "no walker-state clause)")
    ap.add_argument("--prep-dir", type=Path,
                    default=Path("runs/prep_v1sp6"))
    ap.add_argument("--entries", type=Path, default=None,
                    help="JSON list of {id, source, frames_dir} — overrides eval30")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, tok = load_model(args.ckpt, args.base_model)
    image_processor = model.get_vision_tower().image_processor

    args.out.parent.mkdir(parents=True, exist_ok=True)
    entries = (json.loads(args.entries.read_text()) if args.entries
               else data.eval30_entries())
    rows = []
    for e in entries:
        rec_path = args.prep_dir / "records" / f"{e['id']}.json"
        rec = json.loads(rec_path.read_text())
        if args.nocmc:
            rec = {**rec, **rec["nocmc"]}
        image_path = (str(Path(e["frames_dir"]) / "8.jpg") if args.no_marks
                      else rec["marked_image"])
        if args.blank_image:
            from PIL import Image as _I
            _blk = Path("runs/bench300/blank")
            _blk.mkdir(parents=True, exist_ok=True)
            _w, _h = _I.open(image_path).size
            _p = _blk / f"black_{_w}x{_h}.png"
            if not _p.exists():
                _I.new("RGB", (_w, _h), (0, 0, 0)).save(_p)
            image_path = str(_p)
        if args.frames:
            fd = Path(e["frames_dir"])
            ex = {"id": e["id"],
                  "image_path": [str(fd / f"{k}.jpg") for k in range(args.frames)],
                  "user_text": build_frames_text(args.frames)}
        else:
            ex = {
                "id": e["id"],
                "image_path": image_path,
                "user_text": build_user_text(
                    "" if args.drop_text in ("detections", "both") else rec["detections"],
                    "" if args.drop_text in ("motion", "both") else rec["motion"],
                    alert_only=(args.arm == "alert")),
            }
        if args.pixelate_boxes:
            ex["pixelate_boxes"] = [t["box"] for t in rec.get("tracks", [])[:8]]
        t0 = time.time()
        out = generate_one(model, tok, image_processor, ex, args.content_edge)
        ms = int((time.time() - t0) * 1000)
        if args.arm == "json":
            parsed = parse_model_json(out)
            alert = (parsed or {}).get("alert", "")
            if not isinstance(alert, str):
                alert = str(alert)
        else:
            alert = out
        rows.append({"id": e["id"], "source": e["source"], "output": out,
                     "alert": alert, "latency_ms": ms})
        print(f"[infer] {e['id']}: {out[:120]}", flush=True)

    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[infer] wrote {len(rows)} rows -> {args.out}", flush=True)

if __name__ == "__main__":
    main()

"""Greedy teacher generation of training answers for sequence-level distillation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.prompt import build_user_text, parse_model_json
from generator.sft import generate_one, resolve_model_path

def merge(data: Path, out_dir: Path) -> None:
    gen = {}
    for p in sorted(out_dir.glob("gen-*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            gen[r["id"]] = r["gen"]
    rows = [json.loads(l) for l in data.open() if l.strip()]
    out_path = out_dir / "train.jsonl"
    n_ok = n_bad = 0
    with out_path.open("w") as fh:
        for r in rows:
            g = gen.get(r["id"], "")
            parsed = parse_model_json(g)
            if not parsed or not str(parsed.get("alert", "")).strip():
                n_bad += 1
                continue
            fh.write(json.dumps({**r, "target_json": g},
                                ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[merge] {n_ok} rows ({n_bad} dropped) -> {out_path}", flush=True)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--base-model", type=str, default="apple/FastVLM-7B")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge(args.data, args.out_dir)
        return

    out_path = args.out_dir / f"gen-{args.shard:02d}.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.open() if l.strip()}

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = resolve_model_path(args.base_model)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload().eval()
    image_processor = model.get_vision_tower().image_processor

    rows = [json.loads(l) for l in args.data.open() if l.strip()]
    rows = [r for i, r in enumerate(rows)
            if i % args.num_shards == args.shard and r["id"] not in done]
    print(f"[gen shard {args.shard}] {len(rows)} rows "
          f"({len(done)} done)", flush=True)

    t0 = time.time()
    with out_path.open("a") as fh:
        for i, r in enumerate(rows, 1):
            ex = {"image_path": r["image"],
                  "user_text": build_user_text(r["detections"], r["motion"])}
            g = generate_one(model, tok, image_processor, ex, None,
                             max_new_tokens=160)
            fh.write(json.dumps({"id": r["id"], "gen": g},
                                ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(rows):
                fh.flush()
                print(f"[gen {args.shard}] {i}/{len(rows)} "
                      f"{i/(time.time()-t0):.2f} row/s", flush=True)
    print(f"[gen {args.shard}] DONE", flush=True)

if __name__ == "__main__":
    main()

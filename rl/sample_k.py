"""Samples K completions per prompt from a checkpoint at fixed temperature."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.prompt import build_user_text, parse_model_json
from generator.sft import IMAGE_TOKEN_INDEX, load_image

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--entries", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--prep-dir", type=Path, default=Path("runs/prep_v1sp6"))
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    src = str(args.ckpt / "model") if (args.ckpt / "model").is_dir() else str(args.ckpt)
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    proc = model.get_vision_tower().image_processor

    entries = json.loads(args.entries.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fhs = [(args.out_dir / f"k{j}.jsonl").open("w") for j in range(args.k)]
    t0 = time.time()
    for i, e in enumerate(entries):
        rec = json.loads((args.prep_dir / "records" / f"{e['id']}.json").read_text())
        user_text = build_user_text(rec["detections"], rec["motion"])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "<image>\n" + user_text}],
            add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)
        ids = (tok(pre, add_special_tokens=False).input_ids + [IMAGE_TOKEN_INDEX]
               + tok(post, add_special_tokens=False).input_ids)
        in_ids = torch.tensor([ids], device=model.device)
        px = proc(images=load_image(rec["marked_image"], None), return_tensors="pt")["pixel_values"]
        px = px.to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(
                inputs=in_ids.expand(args.k, -1),
                attention_mask=torch.ones(args.k, in_ids.shape[1], device=model.device,
                                          dtype=torch.long),
                images=px.expand(args.k, *px.shape[1:]),
                do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        for j in range(args.k):
            out = " ".join(tok.decode(gen[j], skip_special_tokens=True).strip().split())
            parsed = parse_model_json(out) or {}
            alert = parsed.get("alert", "")
            fhs[j].write(json.dumps({"id": e["id"], "source": e.get("source", ""),
                                     "output": out, "alert": alert if isinstance(alert, str) else "",
                                     "k": j}, ensure_ascii=False) + "\n")
            fhs[j].flush()
        if (i + 1) % 10 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"[sample] {i+1}/{len(entries)} ({r:.2f} clips/s)", flush=True)
    for fh in fhs:
        fh.close()
    print(f"[sample] wrote {args.k} files -> {args.out_dir}", flush=True)

if __name__ == "__main__":
    main()

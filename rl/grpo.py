"""Dr. GRPO training of the student against the programmatic reward, with KL to the frozen initialization."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.policy_utils import comp_logprobs, encode_prompt
from generator.prompt import build_user_text, parse_model_json
from generator.sft import load_image
from rl import reward as RW

def sample_group(model, tok, prompt_ids, px, k, temperature, max_new):
    in_ids = torch.tensor([prompt_ids], device=model.device)
    model.config.use_cache = True
    with torch.no_grad():
        gen = model.generate(
            inputs=in_ids.expand(k, -1),
            attention_mask=torch.ones(k, in_ids.shape[1], device=model.device, dtype=torch.long),
            images=px.expand(k, *px.shape[1:]),
            do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
            max_new_tokens=max_new, eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id)
    model.config.use_cache = False
    comps, texts = [], []

    for g in gen:
        c = g.tolist()
        if tok.eos_token_id in c:
            c = c[:c.index(tok.eos_token_id) + 1]
        else:
            while c and c[-1] == tok.pad_token_id:
                c.pop()
        if len(c) >= 2:
            comps.append(c)
            texts.append(" ".join(tok.decode(c, skip_special_tokens=True).strip().split()))
    return comps, texts

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", type=Path, required=True, help="student init = reference policy")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/sft_v1sp6/train.jsonl"))
    ap.add_argument("--weights", type=Path, default=None, help="JSON {cov,act,clr,len} overriding reward.WEIGHTS")
    ap.add_argument("--n-prompts", type=int, default=6000)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.02, help="KL to the frozen init")
    ap.add_argument("--token-const", type=float, default=64.0, help="Dr.GRPO fixed length constant")
    ap.add_argument("--std-floor", type=float, default=1e-3, help="skip groups flatter than this")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=250, help="optimizer steps between checkpoints")
    ap.add_argument("--dev-size", type=int, default=200)
    ap.add_argument("--dev-every", type=int, default=250, help="optimizer steps between dev passes")
    ap.add_argument("--no-soft-match", action="store_true", help="reward: synonym table only")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights = json.loads(args.weights.read_text()) if args.weights else RW.WEIGHTS
    enc = None if args.no_soft_match else RW.Encoder()

    start_prompt, init_dir = 0, args.init
    cks = sorted(args.out_dir.glob("ckpt-*/model"), key=lambda p: int(p.parent.name.split("-")[1]))
    if cks:
        init_dir = cks[-1]
        start_prompt = int(cks[-1].parent.name.split("-")[1])
        print(f"[grpo] RESUME from {init_dir} (prompt {start_prompt})", flush=True)

    tok = AutoTokenizer.from_pretrained(str(init_dir), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        str(init_dir), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    student.config.use_cache = False
    for name, p in student.named_parameters():
        p.requires_grad = "vision_tower" not in name
    proc = student.get_vision_tower().image_processor

    ref = AutoModelForCausalLM.from_pretrained(
        str(args.init), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    ref.config.use_cache = False
    for p in ref.parameters():
        p.requires_grad = False

    rows = [json.loads(l) for l in args.data.open() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    dev_rows, rows = rows[:args.dev_size], rows[args.dev_size:]
    rows = rows[:args.n_prompts]
    todo = rows[start_prompt:]
    print(f"[grpo] {len(todo)} prompts to go ({start_prompt} consumed), K={args.k} "
          f"lr={args.lr} beta={args.beta} weights={weights}", flush=True)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step = start_prompt // args.prompts_per_step

    def prep(r):
        user_text = build_user_text(r["detections"], r["motion"])
        ids = encode_prompt(tok, user_text)
        px = proc(images=load_image(r["image"], None), return_tensors="pt")["pixel_values"]
        return ids, px.to(student.device, dtype=torch.bfloat16)

    def save(n_done):
        d = args.out_dir / f"ckpt-{n_done}" / "model"
        student.config.use_cache = True
        student.save_pretrained(d)
        tok.save_pretrained(d)
        student.config.use_cache = False
        print(f"[grpo] saved {d}", flush=True)

    @torch.no_grad()
    def dev_pass(tag):
        student.eval()
        acc = {k: 0.0 for k in ("cov", "act", "clr", "len", "gnd", "fmt")}
        tot, nw = 0.0, 0.0
        for r in dev_rows:
            ids, px = prep(r)
            in_ids = torch.tensor([ids], device=student.device)
            student.config.use_cache = True
            g = student.generate(inputs=in_ids, attention_mask=torch.ones_like(in_ids),
                                 images=px, do_sample=False,
                                 max_new_tokens=args.max_new_tokens,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            student.config.use_cache = False
            out = " ".join(tok.decode(g[0], skip_special_tokens=True).strip().split())
            c = RW.components(out, json.loads(r["target_json"]), enc, r["detections"])
            for k in acc:
                acc[k] += c[k]
            tot += RW.total(c, weights)
            nw += c["n_words"]
        n = len(dev_rows)
        line = {"tag": tag, "step": step, "R": round(tot / n, 4),
                **{k: round(v / n, 4) for k, v in acc.items()}, "words": round(nw / n, 2)}
        print(f"[dev] {json.dumps(line)}", flush=True)
        with (args.out_dir / "dev.jsonl").open("a") as fh:
            fh.write(json.dumps(line) + "\n")
        student.train()

    t0 = time.time()
    run = {"R": 0.0, "kl": 0.0, "n": 0, "skip": 0, "json_ok": 0, "samples": 0,
           "words": 0.0, "pg": 0.0, "loss": 0.0, "rstd": 0.0, "adv": 0.0,
           "logp": 0.0, "gnorm": 0.0, "toks": 0.0}
    opt.zero_grad()
    dev_pass("init")
    student.train()
    for i, r in enumerate(todo):
        pi = start_prompt + i
        gold = json.loads(r["target_json"])
        ids, px = prep(r)
        comps, texts = sample_group(student, tok, ids, px, args.k, args.temperature,
                                    args.max_new_tokens)
        if len(comps) < 2:
            continue
        R = []
        for t in texts:
            c = RW.components(t, gold, enc, r["detections"])
            R.append(RW.total(c, weights))
            run["json_ok"] += int(c["fmt"] > 0)
            run["words"] += c["n_words"]
        run["samples"] += len(texts)
        Rt = torch.tensor(R, device=student.device, dtype=torch.float32)
        run["R"] += float(Rt.mean())
        run["rstd"] += float(Rt.std())
        run["n"] += 1
        if float(Rt.std()) < args.std_floor:
            run["skip"] += 1
            continue
        adv = (Rt - Rt.mean()).detach()

        lp, mask = comp_logprobs(student, ids, comps, px, True)
        with torch.no_grad():
            lp_ref, _ = comp_logprobs(ref, ids, comps, px, False)
        pg = -(adv[:, None] * lp * mask).sum() / (len(comps) * args.token_const)
        d = (lp_ref - lp).clamp(-10, 10)
        kl = ((d.exp() - d - 1.0) * mask).sum() / mask.sum().clamp(min=1)
        loss = pg + args.beta * kl
        (loss / args.prompts_per_step).backward()
        run["kl"] += float(kl)
        run["pg"] += float(pg)
        run["loss"] += float(loss)
        run["adv"] += float(adv.abs().mean())
        run["logp"] += float((lp * mask).sum() / mask.sum().clamp(min=1))
        run["toks"] += float(mask.sum()) / len(comps)

        if (i + 1) % args.prompts_per_step == 0:
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / max(args.warmup_steps, 1))
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            run["gnorm"] += float(gn)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                n = max(run["n"], 1)
                rate = (i + 1) / (time.time() - t0)
                m = max(n - run["skip"], 1)
                line = {"step": step, "prompts": pi + 1,
                        "R": round(run["R"] / n, 4), "R_std": round(run["rstd"] / n, 4),
                        "loss": round(run["loss"] / m, 5), "pg": round(run["pg"] / m, 5),
                        "kl": round(run["kl"] / m, 5), "adv": round(run["adv"] / m, 4),
                        "logp": round(run["logp"] / m, 4),
                        "gnorm": round(run["gnorm"] / 10.0, 3),
                        "toks": round(run["toks"] / m, 1),
                        "skip": f"{run['skip']}/{n}",
                        "json": f"{run['json_ok']}/{run['samples']}",
                        "w": round(run["words"] / max(run["samples"], 1), 1),
                        "p_s": round(rate, 2),
                        "eta_h": round((len(todo) - i - 1) / max(rate, 1e-6) / 3600, 1)}
                print(f"[grpo] {json.dumps(line)}", flush=True)
                with (args.out_dir / "train.jsonl").open("a") as fh:
                    fh.write(json.dumps(line) + "\n")
                run = {k: 0.0 if isinstance(v, float) else 0 for k, v in run.items()}
            if step % args.dev_every == 0:
                dev_pass(f"step{step}")
            if step % args.save_every == 0:
                save(pi + 1)

    dev_pass("final")
    save(args.n_prompts)
    (args.out_dir / "train_meta.json").write_text(json.dumps({
        "init": str(args.init), "algo": "Dr.GRPO", "k": args.k, "lr": args.lr,
        "beta": args.beta, "temperature": args.temperature, "weights": weights,
        "token_const": args.token_const, "n_prompts": args.n_prompts,
        "soft_match": not args.no_soft_match}, indent=2))
    print("[grpo] DONE", flush=True)

if __name__ == "__main__":
    main()

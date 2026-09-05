"""Supervised fine-tuning: LoRA for the teacher, full-parameter for the student, loss on completion tokens only."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.prompt import build_frames_text, build_user_text

IMAGE_TOKEN_INDEX = -200
LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj")

import re

FIELD_WEIGHTS = {"scaffold": 0.3, "alert": 1.3, "hazards": 1.0,
                 "scene": 0.7, "danger": 3.0}
_SPAN_RES = {
    "alert": re.compile(r'^\{"alert":"(.*?)","hazards":'),
    "hazards": re.compile(r'"hazards":\[(.*)\],"scene":'),
    "scene": re.compile(r'"scene":"(.*?)","danger_level":'),
    "danger": re.compile(r'"danger_level":(\d+)\}$'),
}

def field_weight_per_char(target: str) -> list[float]:
    w = [FIELD_WEIGHTS["scaffold"]] * len(target)
    for name, rex in _SPAN_RES.items():
        m = rex.search(target)
        if m:
            for i in range(m.start(1), m.end(1)):
                w[i] = FIELD_WEIGHTS[name]
    return w

def resolve_model_path(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    root = os.environ.get("MODEL_ROOT")
    if root:
        cand = Path(root) / Path(model_id_or_path).name
        if cand.is_dir():
            return str(cand)
    return model_id_or_path

class RoundDataset(Dataset):
    def __init__(self, jsonl_path: Path, arm: str, limit: int | None = None,
                 no_marks: bool = False, frames: int = 0):
        rows = [json.loads(l) for l in jsonl_path.open() if l.strip()]
        key = "target_json" if arm == "json" else "target_alert"
        rows = [r for r in rows if r.get(key, "").strip()]
        self.rows = rows[:limit] if limit else rows
        self.arm = arm
        self.key = key
        self.img_key = "image_raw" if no_marks else "image"

        self.frames = frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        if self.frames:
            d = Path(r["image_raw"]).parent
            return {
                "id": r["id"],
                "image_path": [str(d / f"{k}.jpg") for k in range(self.frames)],
                "user_text": build_frames_text(self.frames),
                "target": r[self.key],
                "boxes": [],
            }
        return {
            "id": r["id"],
            "image_path": r.get(self.img_key) or r["image"],
            "user_text": build_user_text(
                "" if getattr(self, "drop_text", None) in ("detections", "both") else r["detections"],
                "" if getattr(self, "drop_text", None) in ("motion", "both") else r["motion"],
                alert_only=(self.arm == "alert")),
            "target": r[self.key],
            "boxes": r.get("boxes") or [],
        }

def load_image(path: str, content_edge: int | None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if content_edge and max(img.size) > content_edge:
        w, h = img.size
        s = content_edge / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                         Image.LANCZOS)
    return img

def pixelate_boxes(img: Image.Image, boxes: list, factor: int = 4) -> Image.Image:

    out = img.copy()
    W, H = out.size
    for b in boxes:
        x1, y1, x2, y2 = (max(0, int(b[0])), max(0, int(b[1])),
                          min(W, int(b[2])), min(H, int(b[3])))
        if x2 - x1 < factor * 2 or y2 - y1 < factor * 2:
            continue
        crop = out.crop((x1, y1, x2, y2))
        small = crop.resize((max(1, (x2 - x1) // factor),
                             max(1, (y2 - y1) // factor)), Image.BILINEAR)
        out.paste(small.resize((x2 - x1, y2 - y1), Image.BILINEAR), (x1, y1))
    return out

class Collator:

    def __init__(self, tok, image_processor, content_edge: int | None,
                 pixelate: bool = False, field_weights: bool = False):
        self.tok = tok
        self.image_processor = image_processor
        self.content_edge = content_edge
        self.pixelate = pixelate
        self.field_weights = field_weights
        self.pad_id = (tok.pad_token_id if tok.pad_token_id is not None
                       else tok.eos_token_id)

    def _encode(self, ex: dict) -> dict:
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": "<image>\n" + ex["user_text"]}],
            add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)
        pre_ids = self.tok(pre, add_special_tokens=False).input_ids
        post_ids = self.tok(post, add_special_tokens=False).input_ids
        comp_text = ex["target"] + "<|im_end|>"
        comp_ids = self.tok(comp_text, add_special_tokens=False).input_ids
        ids = pre_ids + [IMAGE_TOKEN_INDEX] + post_ids + comp_ids
        labels = [-100] * (len(pre_ids) + 1 + len(post_ids)) + comp_ids
        paths = ex["image_path"]
        if isinstance(paths, list):

            imgs = [load_image(p, self.content_edge) for p in paths]
            px = self.image_processor(images=imgs,
                                      return_tensors="pt")["pixel_values"]
        else:
            img = load_image(paths, self.content_edge)
            if self.pixelate and ex.get("boxes"):
                img = pixelate_boxes(img, ex["boxes"])
            px = self.image_processor(images=img,
                                      return_tensors="pt")["pixel_values"][0]
        out = {"ids": ids, "labels": labels, "px": px}
        if self.field_weights and ex["target"].startswith("{"):
            offs = self.tok(comp_text, add_special_tokens=False,
                            return_offsets_mapping=True).offset_mapping
            cw = field_weight_per_char(ex["target"])
            tw = []
            for a, b in offs:
                span = cw[a:min(b, len(cw))]
                tw.append(sum(span) / len(span) if span else 1.0)
            scale = len(tw) / max(sum(tw), 1e-6)
            out["tw"] = [w * scale for w in tw]
        return out

    def __call__(self, batch: list[dict]) -> dict:
        enc = [self._encode(ex) for ex in batch]
        max_len = max(len(e["ids"]) for e in enc)
        input_ids, labels, attn = [], [], []
        for e in enc:
            n = len(e["ids"])
            input_ids.append(torch.tensor(e["ids"] + [self.pad_id] * (max_len - n)))
            labels.append(torch.tensor(e["labels"] + [-100] * (max_len - n)))
            attn.append(torch.tensor([1] * n + [0] * (max_len - n)))
        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
            "images": torch.stack([e["px"] for e in enc]),
        }
        if self.field_weights:
            n_comp = [sum(1 for v in e["labels"] if v != -100) for e in enc]
            max_T = max(n_comp)
            comp_ids = torch.full((len(enc), max_T), -100, dtype=torch.long)
            weights = torch.zeros((len(enc), max_T))
            for b, (e, T) in enumerate(zip(enc, n_comp)):
                comp = [v for v in e["labels"] if v != -100]
                comp_ids[b, :T] = torch.tensor(comp)
                weights[b, :T] = torch.tensor(e.get("tw", [1.0] * T))
            out["comp_ids"] = comp_ids
            out["loss_weights"] = weights
            out["n_comp"] = torch.tensor(n_comp)
        return out

def make_weighted_trainer_cls():

    import torch.nn.functional as F
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            comp_ids = inputs.pop("comp_ids")
            weights = inputs.pop("loss_weights")
            n_comp = inputs.pop("n_comp")
            inputs.pop("labels", None)
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            images=inputs["images"])
            logits = outputs.logits
            n_img = logits.shape[1] - inputs["attention_mask"].shape[1] + 1
            attn_len = inputs["attention_mask"].sum(1)
            B, max_T = comp_ids.shape
            comp_logits = logits.new_zeros((B, max_T, logits.shape[-1]))
            for b in range(B):
                L = int(attn_len[b]) - 1 + n_img
                T = int(n_comp[b])
                comp_logits[b, :T] = logits[b, L - T - 1: L - 1]
            mask = comp_ids.ne(-100)
            ce = F.cross_entropy(comp_logits.transpose(1, 2).float(),
                                 comp_ids.clamp(min=0), reduction="none")
            loss = (ce * weights * mask).sum() / mask.sum().clamp(min=1)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer

@torch.inference_mode()
def generate_one(model, tok, image_processor, ex: dict,
                 content_edge: int | None, max_new_tokens: int = 160) -> str:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "<image>\n" + ex["user_text"]}],
        add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    paths = ex["image_path"]
    if isinstance(paths, list):
        imgs = [load_image(p, content_edge) for p in paths]
        px = image_processor(images=imgs,
                             return_tensors="pt")["pixel_values"].unsqueeze(0)
    else:
        img = load_image(paths, content_edge)
        if ex.get("pixelate_boxes"):
            img = pixelate_boxes(img, ex["pixelate_boxes"])
        px = image_processor(images=img, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=next(model.parameters()).dtype)

    out = model.generate(inputs=input_ids,
                         attention_mask=torch.ones_like(input_ids),
                         images=px, max_new_tokens=max_new_tokens,
                         do_sample=False, eos_token_id=tok.eos_token_id,
                         pad_token_id=tok.pad_token_id)
    return " ".join(tok.decode(out[0], skip_special_tokens=True).strip().split())

def write_report(out_dir: Path, model, tok, image_processor,
                 val_ds: RoundDataset, n: int, content_edge: int | None) -> None:
    model.eval()
    records = []
    for i in range(min(n, len(val_ds))):
        ex = val_ds[i]
        out = generate_one(model, tok, image_processor, ex, content_edge)
        records.append({"id": ex["id"], "target": ex["target"], "output": out})
        print(f"[report {i}] GT : {ex['target']}")
        print(f"[report {i}] out: {out}", flush=True)
    with (out_dir / "report_samples.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, default="apple/FastVLM-7B")
    ap.add_argument("--arm", choices=["json", "alert"], default="json")
    ap.add_argument("--tune", choices=["lora", "full"], default=None,
                    help="default: lora for 7B, full for 0.5B")
    ap.add_argument("--content-edge", type=int, default=None,
                    help="downscale image content to this longest edge first")
    ap.add_argument("--no-marks", action="store_true",
                    help="train on the raw final frame (no numbered boxes)")
    ap.add_argument("--pixelate-boxes", action="store_true",
                    help="exploration: low-pass detected regions, keep rest sharp")
    ap.add_argument("--frames", type=int, default=0,
                    help="ladder rung A: feed N raw clip frames and no "
                         "perception text (0 = normal single marked frame)")
    ap.add_argument("--drop-text", choices=["motion", "detections", "both"],
                    default=None,
                    help="input ablation: train with one or both text channels "
                         "blanked (slot filled with 'none'; template unchanged)")
    ap.add_argument("--field-weights", action="store_true",
                    help="exploration: per-field weighted CE on the JSON target")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--resume-adapter", type=Path, default=None,
                    help="continue training from an existing LoRA adapter")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-4 lora, 1e-5 full")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--report-samples", type=int, default=8)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    tune = args.tune or ("full" if "0.5B" in args.model else "lora")
    lr = args.lr or (1e-5 if tune == "full" else 1e-4)
    resolved = resolve_model_path(args.model)
    print(f"[sft] model={resolved} tune={tune} arm={args.arm} lr={lr} "
          f"content_edge={args.content_edge}", flush=True)

    tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        resolved, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    image_processor = model.get_vision_tower().image_processor
    model.config.use_cache = False
    model.enable_input_require_grads()

    if tune == "lora":
        from peft import LoraConfig, PeftModel, get_peft_model
        if args.resume_adapter:
            print(f"[sft] resuming from adapter {args.resume_adapter}",
                  flush=True)
            model = PeftModel.from_pretrained(
                model, str(args.resume_adapter), is_trainable=True)
        else:
            targets = [
                name for name, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and ".layers." in name
                and "vision_tower" not in name and "mm_projector" not in name
                and name.rsplit(".", 1)[-1] in LORA_SUFFIXES
            ]
            print(f"[sft] LoRA on {len(targets)} backbone Linears", flush=True)
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                target_modules=targets, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    else:
        for name, p in model.named_parameters():
            p.requires_grad = "vision_tower" not in name
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[sft] full finetune: {n_train/1e6:.0f}M trainable "
              f"(vision tower frozen)", flush=True)

    train_ds = RoundDataset(args.data_dir / "train.jsonl", args.arm,
                            args.max_train, args.no_marks, args.frames)
    val_ds = RoundDataset(args.data_dir / "val.jsonl", args.arm, args.max_val,
                          args.no_marks, args.frames)
    train_ds.drop_text = val_ds.drop_text = args.drop_text
    print(f"[sft] train {len(train_ds)} / val {len(val_ds)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=6,
        dataloader_persistent_workers=True,
        report_to=[],
        seed=17,
    )
    collator = Collator(tok, image_processor, args.content_edge,
                        pixelate=args.pixelate_boxes,
                        field_weights=args.field_weights)
    trainer_cls = make_weighted_trainer_cls() if args.field_weights else Trainer
    trainer = trainer_cls(model=model, args=targs,
                          train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=collator)

    metrics0 = trainer.evaluate()
    print(f"[sft] pre-train val loss : {metrics0.get('eval_loss'):.4f}",
          flush=True)
    trainer.train()
    metrics1 = trainer.evaluate()
    print(f"[sft] post-train val loss: {metrics1.get('eval_loss'):.4f}",
          flush=True)
    (args.out_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history))

    save_dir = args.out_dir / ("adapter" if tune == "lora" else "model")
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    (args.out_dir / "train_meta.json").write_text(json.dumps({
        "model": resolved, "tune": tune, "arm": args.arm, "lr": lr,
        "content_edge": args.content_edge, "no_marks": args.no_marks,
        "drop_text": args.drop_text, "frames": args.frames,
        "pixelate_boxes": args.pixelate_boxes,
        "field_weights": args.field_weights,
        "data_dir": str(args.data_dir),
        "pre_val_loss": metrics0.get("eval_loss"),
        "post_val_loss": metrics1.get("eval_loss"),
        "train_size": len(train_ds), "val_size": len(val_ds),
        "epochs": args.epochs,
        "resume_adapter": str(args.resume_adapter) if args.resume_adapter else None,
        "lora_r": args.lora_r if tune == "lora" else None}, indent=2))

    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    write_report(args.out_dir, model, tok, image_processor, val_ds,
                 args.report_samples, args.content_edge)
    print(f"[sft] done -> {args.out_dir}", flush=True)

if __name__ == "__main__":
    main()

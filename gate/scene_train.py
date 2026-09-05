"""Trains and evaluates the scene gate."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gate import metrics as MET
from gate.scene_model import SceneGate

DATA = Path("runs/gate/data")
TOWER = Path("runs/gate/tower")
CKPT = Path("runs/gate/ckpt")
BENCH_IDS = {e["id"] for e in json.loads(
    Path("runs/relabel/splits_2026-08-10/benchmark.json").read_text())}
GLOBAL_TOKEN = 16

def load(split: str, exclude: set[str] | None = None, grid: bool = False,
         data_dir: Path = DATA):

    d = np.load(data_dir / f"{split}.npz")
    t = np.load(data_dir / f"tower_{split}.npz")
    row = {str(c): i for i, c in enumerate(t["ids"])}
    ids = [str(s) for s in d["ids"]]
    keep = [i for i, c in enumerate(ids) if c in row and not (exclude and c in exclude)]
    tk = np.asarray([row[ids[i]] for i in keep]); keep = np.asarray(keep)
    return {
        "ids": [ids[i] for i in keep],
        "V": torch.from_numpy((t["G"] if grid else t["V"])[tk]),
        "S": torch.from_numpy(d["S"][keep].astype(np.float32)),
        "y": torch.from_numpy(d["y"][keep].astype(np.float32)),
        "onset": d["onset"][keep].astype(np.int64),
        "danger": d["danger"][keep].astype(np.int64),
    }

def fit_pca(V: torch.Tensor, k: int):

    X = V.reshape(-1, V.shape[-1]).float()
    mean = X.mean(0)
    Xc = X - mean

    idx = torch.randperm(Xc.shape[0])[:40000]
    U, Sv, Vh = torch.linalg.svd(Xc[idx], full_matrices=False)
    comp = Vh[:k].T.contiguous()
    proj = Xc[idx] @ comp
    return mean, comp, proj.std(0).clamp(min=1e-3)

@torch.no_grad()
def predict(model, d, bs=512):

    model.eval()
    out, hs, xs = [], [], []
    for i in range(0, len(d["ids"]), bs):
        V, S = d["V"][i:i + bs], d["S"][i:i + bs]
        lg, H = model(V, S)
        out.append(lg); hs.append(H); xs.append(model.embed(V, S))
    return torch.cat(out).numpy(), torch.cat(hs).numpy(), torch.cat(xs).numpy()

def similarity_scales(H: np.ndarray, n: int = 4000, seed: int = 0):

    rng = np.random.default_rng(seed)
    Hn = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-9)
    same = (Hn[:, 1:] * Hn[:, :-1]).sum(-1).reshape(-1)
    a = rng.integers(0, H.shape[0], n); b = rng.integers(0, H.shape[0], n)
    ta = rng.integers(0, H.shape[1], n); tb = rng.integers(0, H.shape[1], n)
    diff = (Hn[a, ta] * Hn[b, tb]).sum(-1)
    return float(np.mean(same)), float(np.mean(diff))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=CKPT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--d-vis", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-symbolic", action="store_true")
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--grid", action="store_true",
                    help="mean of the 16 grid tokens instead of the global token")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--smoke", action="store_true",
                    help="carve val/test/bench out of whatever train clips are cached")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(8)

    tr = load("train", exclude=BENCH_IDS, grid=a.grid, data_dir=a.data)
    if a.smoke:
        n = len(tr["ids"]); perm = np.random.permutation(n)
        cut = lambda sl: {k: (v[sl] if isinstance(v, (np.ndarray, torch.Tensor))
                              else [v[i] for i in sl]) for k, v in tr.items()}
        va, te, be = cut(perm[:n // 5]), cut(perm[n // 5:2 * n // 5]), cut(perm[2 * n // 5:3 * n // 5])
        tr = cut(perm[3 * n // 5:])
    else:
        va = load("val", grid=a.grid, data_dir=a.data)
        te = load("test", exclude=BENCH_IDS, grid=a.grid, data_dir=a.data)
        be = load("benchmark", grid=a.grid, data_dir=a.data)
    print(f"train {len(tr['ids'])} | val {len(va['ids'])} | test {len(te['ids'])} "
          f"| bench {len(be['ids'])}", flush=True)

    pm, pc, ps = fit_pca(tr["V"], a.d_vis)
    Sf = tr["S"].reshape(-1, tr["S"].shape[-1])
    model = SceneGate(d_vis=a.d_vis, d_sym=tr["S"].shape[-1], hidden=a.hidden,
                      use_vision=not a.no_vision, use_symbolic=not a.no_symbolic,
                      temporal=not a.no_temporal,
                      pca_mean=pm, pca_comp=pc, sym_mean=Sf.mean(0),
                      sym_std=Sf.std(0).clamp(min=1e-3))
    model.pca_scale.copy_(ps)
    print(f"params {model.n_params()}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    n = len(tr["ids"]); steps = max(1, n // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.epochs * steps)
    best, best_state, hist = -1.0, None, []
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0; t0 = time.time()
        for b in range(steps):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            lg, _ = model(tr["V"][idx], tr["S"][idx])
            loss = Fn.binary_cross_entropy_with_logits(lg, tr["y"][idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += float(loss)
        lg, _, _ = predict(model, va)
        m = MET.summarize(lg, va["y"].numpy(), va["onset"])
        comp = 0.5 * m["pr_auc"] + 0.5 * m["clip_roc_auc"]
        hist.append({"epoch": ep, "loss": tot / steps, "val_frame_pr": m["pr_auc"],
                     "val_frame_roc": m["roc_auc"], "val_clip_roc": m["clip_roc_auc"]})
        if comp > best:
            best = comp; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:3d} loss {tot/steps:.4f} | val frame-ROC {m['roc_auc']:.4f} "
              f"PR {m['pr_auc']:.4f} clip-ROC {m['clip_roc_auc']:.4f} | "
              f"{time.time()-t0:.1f}s", flush=True)

    model.load_state_dict(best_state)
    out = a.out / a.tag; out.mkdir(parents=True, exist_ok=True)
    res = {"tag": a.tag, "args": {k: (str(v) if isinstance(v, Path) else v)
                                 for k, v in vars(a).items()},
           "params": model.n_params(), "history": hist}
    _, Htr, Xtr = predict(model, tr)
    res["sim_same"], res["sim_diff"] = similarity_scales(Htr)
    res["xsim_same"], res["xsim_diff"] = similarity_scales(Xtr)
    print(f"similarity anchors  hidden: same-clip {res['sim_same']:.4f} "
          f"different-clip {res['sim_diff']:.4f} | appearance: "
          f"{res['xsim_same']:.4f} / {res['xsim_diff']:.4f}", flush=True)
    for name, d in [("val", va), ("test", te), ("benchmark", be)]:
        lg, H, X = predict(model, d)
        m = MET.summarize(lg, d["y"].numpy(), d["onset"]); res[name] = m
        np.savez_compressed(out / f"preds_{name}.npz", ids=np.asarray(d["ids"]),
                            logit=lg, H=H.astype(np.float16), X=X.astype(np.float16),
                            y=d["y"].numpy(), onset=d["onset"], danger=d["danger"])
        print(f"[{name}] frame-ROC {m['roc_auc']:.4f} PR {m['pr_auc']:.4f} "
              f"clip-ROC {m['clip_roc_auc']:.4f}", flush=True)
        for row in m["timing_warm"]:
            print(f"    FA {row['fa_rate']:.3f} -> hit {row['hit_rate']:.3f} "
                  f"med-err {row['median_err_s']:+.2f}s  ±1frame {row['within_1frame']:.3f}",
                  flush=True)
    torch.save({"state_dict": model.state_dict(), "config": res["args"],
                "sim_same": res["sim_same"], "sim_diff": res["sim_diff"],
                "xsim_same": res["xsim_same"], "xsim_diff": res["xsim_diff"],
                "d_sym": int(tr["S"].shape[-1])}, out / "gate.pt")
    (out / "report.json").write_text(json.dumps(res, indent=1))
    print("saved", out, flush=True)

if __name__ == "__main__":
    main()

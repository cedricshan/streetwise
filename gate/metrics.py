"""Gate evaluation metrics."""
from __future__ import annotations

import numpy as np

DT = 0.375

def _auc(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))

def frame_metrics(scores: np.ndarray, y: np.ndarray) -> dict:

    s, t = scores.reshape(-1), y.reshape(-1)
    roc, pr = _auc(t, s)
    order = np.argsort(-s)
    ts = t[order]
    tp = np.cumsum(ts)
    fp = np.cumsum(1 - ts)
    P, N = ts.sum(), len(ts) - ts.sum()
    ba = 0.5 * (tp / max(P, 1) + 1 - fp / max(N, 1))
    return {"roc_auc": roc, "pr_auc": pr, "best_ba": float(ba.max()),
            "pos_rate": float(t.mean())}

def first_fire(scores: np.ndarray, thr: float) -> np.ndarray:

    over = scores >= thr
    idx = np.argmax(over, axis=1)
    return np.where(over.any(axis=1), idx, -1)

def timing_at(scores: np.ndarray, onset: np.ndarray, thr: float,
              min_onset: int = 1) -> dict:

    ft = first_fire(scores, thr)
    neg = onset < 0
    pos = onset >= min_onset
    fa = float((ft[neg] >= 0).mean()) if neg.any() else float("nan")
    hit_mask = pos & (ft >= 0)
    hit = float(hit_mask.sum() / max(pos.sum(), 1))
    err = (ft[hit_mask] + 1 - onset[hit_mask]) * DT
    return {
        "thr": float(thr), "fa_rate": fa, "hit_rate": hit,
        "miss_rate": 1.0 - hit,
        "median_err_s": float(np.median(err)) if err.size else float("nan"),
        "mean_abs_err_s": float(np.abs(err).mean()) if err.size else float("nan"),
        "within_1frame": float((np.abs(err) <= DT + 1e-6).mean()) if err.size else float("nan"),
        "early_rate": float((err < -DT - 1e-6).mean()) if err.size else float("nan"),
        "late_rate": float((err > DT + 1e-6).mean()) if err.size else float("nan"),
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
    }

def thr_for_fa(scores: np.ndarray, onset: np.ndarray, target_fa: float) -> float:

    neg = scores[onset < 0]
    if neg.size == 0:
        return 0.5
    peak = neg.max(axis=1)
    q = np.quantile(peak, 1.0 - target_fa)
    return float(np.nextafter(q, np.inf))

def timing_curve(scores: np.ndarray, onset: np.ndarray,
                 fas=(0.02, 0.05, 0.10, 0.20, 0.30),
                 min_onset: int = 1) -> list[dict]:
    return [timing_at(scores, onset, thr_for_fa(scores, onset, f), min_onset)
            for f in fas]

def clip_auc(scores: np.ndarray, onset: np.ndarray) -> dict:

    roc, pr = _auc((onset >= 0).astype(int), scores.max(axis=1))
    return {"clip_roc_auc": roc, "clip_pr_auc": pr}

def summarize(scores: np.ndarray, y: np.ndarray, onset: np.ndarray) -> dict:
    out = frame_metrics(scores, y)
    out.update(clip_auc(scores, onset))
    out["timing"] = timing_curve(scores, onset)
    out["timing_warm"] = timing_curve(scores, onset, min_onset=3)
    return out

def certify_threshold(scores: np.ndarray, onset: np.ndarray,
                      alpha: float = 0.10, delta: float = 0.05) -> dict:

    from scipy.stats import binom
    pos = scores[onset >= 1]
    if pos.size == 0:
        return {"alpha": alpha, "delta": delta, "threshold": float("nan")}
    peak = np.sort(pos.max(axis=1))
    n = len(peak)
    grid = np.unique(np.concatenate([[-np.inf], peak]))
    best = float(grid[0])
    for thr in grid:
        miss = float((peak < thr).mean())
        if binom.cdf(int(round(miss * n)), n, alpha) <= delta:
            best = float(thr)
        else:
            break
    return {"alpha": alpha, "delta": delta, "threshold": best,
            "n_pos": int(n),
            "empirical_miss": float((peak < best).mean())}

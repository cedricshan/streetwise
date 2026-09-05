"""Metric depth estimation and 3D positioning of detections."""
from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "Ruicheng/moge-2-vitl-normal"

class DepthEstimator:
    def __init__(self, device: str = "cuda"):
        from moge.model.v2 import MoGeModel
        self.model = MoGeModel.from_pretrained(MODEL_ID).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def infer(self, img_rgb: np.ndarray) -> dict:

        t = torch.tensor(img_rgb / 255.0, dtype=torch.float32,
                         device=self.device).permute(2, 0, 1)
        out = self.model.infer(t)
        return {
            "points": out["points"].cpu().numpy(),
            "depth": out["depth"].cpu().numpy(),
            "mask": out["mask"].cpu().numpy(),
        }

def det_position(points: np.ndarray, mask: np.ndarray,
                 box: tuple[float, float, float, float]) -> tuple[float, float, float] | None:

    h, w = mask.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 + 0.2 * bw)); cx2 = int(min(w, x2 - 0.2 * bw))
    cy1 = int(max(0, y1 + 0.2 * bh)); cy2 = int(min(h, y2 - 0.2 * bh))
    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = (int(max(0, x1)), int(max(0, y1)),
                              int(min(w, x2)), int(min(h, y2)))
    m = mask[cy1:cy2, cx1:cx2]
    if m.sum() < 8:
        return None
    p = points[cy1:cy2, cx1:cx2][m]
    med = np.median(p, axis=0)
    if not np.isfinite(med).all() or med[2] <= 0.05:
        return None
    return float(med[0]), float(med[1]), float(med[2])

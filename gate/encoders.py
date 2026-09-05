"""Frozen FastVLM vision-tower frame encoder with grid pooling."""
from __future__ import annotations

import math

import numpy as np

def grid_pool(tokens: np.ndarray, grid: int = 4) -> np.ndarray:

    n, d = tokens.shape
    side = int(math.isqrt(n))
    out = np.empty((grid * grid + 1, d), dtype=tokens.dtype)
    if side * side == n and side >= grid:
        sq = tokens.reshape(side, side, d)
        edges = np.linspace(0, side, grid + 1).round().astype(int)
        idx = 0
        for i in range(grid):
            for j in range(grid):
                block = sq[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
                out[idx] = block.reshape(-1, d).mean(axis=0)
                idx += 1
    else:
        chunks = np.array_split(tokens, grid * grid, axis=0)
        for idx, chunk in enumerate(chunks):
            out[idx] = chunk.mean(axis=0) if len(chunk) else 0.0
    out[-1] = tokens.mean(axis=0)
    return out

class FastVLMTowerEncoder:

    def __init__(self, model_path: str = "apple/FastVLM-0.5B",
                 device: str = "cuda", grid: int = 4) -> None:
        import gc
        import torch
        from transformers import AutoModelForCausalLM

        self._torch = torch
        self.grid = grid
        self.device = device
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16)
        tower = model.get_vision_tower()
        tower.to(device).eval()
        for p in tower.parameters():
            p.requires_grad_(False)
        self.tower = tower
        self.image_processor = tower.image_processor
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    @property
    def dim(self) -> int:
        return 3072

    def encode_frames(self, images: list) -> np.ndarray:
        torch = self._torch
        pixel = self.image_processor(images=images, return_tensors="pt")[
            "pixel_values"].to(self.device, dtype=torch.float16)
        with torch.no_grad():
            feats = self.tower(pixel)
        feats = feats.float().cpu().numpy()
        return np.stack([grid_pool(f, self.grid) for f in feats]).astype(np.float16)

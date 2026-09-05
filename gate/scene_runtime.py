"""Streaming scene-gate runtime."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .scene_model import SceneGate, SceneMemory

@dataclass
class SceneOutput:
    speak: bool
    p: float
    novelty: float
    hidden: torch.Tensor

class SceneStreamingGate:

    def __init__(self, model: SceneGate, theta: float, tau: float,
                 sim_same: float, sim_diff: float, lookback: int | None = 9):
        self.m = model.eval()
        self.mem = SceneMemory(theta, tau, sim_same, sim_diff)
        self.lookback = lookback
        self.h = None
        self.xs: deque = deque(maxlen=lookback or 1)

    @classmethod
    def from_checkpoint(cls, d: Path, theta: float, tau: float = 10.0,
                        lookback: int | None = 9):
        blob = torch.load(d / "gate.pt", map_location="cpu", weights_only=False)
        c = blob["config"]
        m = SceneGate(d_vis=c["d_vis"], d_sym=blob["d_sym"], hidden=c["hidden"],
                      use_vision=not c["no_vision"], use_symbolic=not c["no_symbolic"],
                      temporal=not c["no_temporal"])
        m.load_state_dict(blob["state_dict"])
        return cls(m, theta, tau, blob["sim_same"], blob["sim_diff"], lookback)

    def reset(self):
        self.h = None
        self.xs.clear()
        self.mem.reset()

    @torch.no_grad()
    def observe(self, v: np.ndarray, s: np.ndarray, t: float) -> SceneOutput:
        V = torch.from_numpy(np.asarray(v, dtype=np.float32)).view(1, -1)
        S = torch.from_numpy(np.asarray(s, dtype=np.float32)).view(1, -1)
        x = self.m.embed(V, S)
        if self.lookback:
            self.xs.append(x)
            h = None
            for xj in self.xs:
                h = self.m.step(xj, h)
            self.h = h
        else:
            self.h = self.m.step(x, self.h)
        p = float(torch.sigmoid(self.m.head(self.h)).item())
        speak, n = self.mem.decide(p, self.h[0], t)
        return SceneOutput(speak=speak, p=p, novelty=n, hidden=self.h[0])

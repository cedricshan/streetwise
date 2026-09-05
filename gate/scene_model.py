"""GRU scene gate over per-frame features, with the novelty-decayed speaking memory."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

class SceneGate(nn.Module):
    def __init__(self, d_vis: int = 64, d_sym: int = 80, hidden: int = 64,
                 use_vision: bool = True, use_symbolic: bool = True,
                 temporal: bool = True,
                 pca_mean=None, pca_comp=None, sym_mean=None, sym_std=None):
        super().__init__()
        self.use_vision, self.use_symbolic, self.temporal = use_vision, use_symbolic, temporal
        d_in = (d_vis if use_vision else 0) + (d_sym if use_symbolic else 0)
        self.hidden = hidden
        if temporal:
            self.rnn = nn.GRUCell(d_in, hidden)
        else:
            self.rnn = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

        self.register_buffer("pca_mean", torch.zeros(3072) if pca_mean is None
                             else torch.as_tensor(pca_mean, dtype=torch.float32))
        self.register_buffer("pca_comp", torch.zeros(3072, d_vis) if pca_comp is None
                             else torch.as_tensor(pca_comp, dtype=torch.float32))
        self.register_buffer("pca_scale", torch.ones(d_vis))
        self.register_buffer("sym_mean", torch.zeros(d_sym) if sym_mean is None
                             else torch.as_tensor(sym_mean, dtype=torch.float32))
        self.register_buffer("sym_std", torch.ones(d_sym) if sym_std is None
                             else torch.as_tensor(sym_std, dtype=torch.float32))

    def embed(self, V, S):

        parts = []
        if self.use_vision:
            parts.append(((V.float() - self.pca_mean) @ self.pca_comp) / self.pca_scale)
        if self.use_symbolic:
            parts.append((S.float() - self.sym_mean) / self.sym_std)
        return torch.cat(parts, dim=-1)

    def step(self, x_t, h):

        if self.temporal:
            if h is None:
                h = x_t.new_zeros(x_t.shape[0], self.hidden)
            return self.rnn(x_t, h)
        return self.rnn(x_t)

    def forward(self, V, S):

        x = self.embed(V, S)
        h, hs = None, []
        for t in range(x.shape[1]):
            h = self.step(x[:, t], h)
            hs.append(h)
        H = torch.stack(hs, dim=1)
        return self.head(H).squeeze(-1), H

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

class SceneMemory:

    def __init__(self, theta: float, tau: float = 10.0,
                 sim_same: float = 1.0, sim_diff: float = 0.0):
        self.theta, self.tau = theta, tau
        self.sim_same, self.sim_diff = sim_same, sim_diff
        self.m = None
        self.t_spoken = None

    def reset(self):
        self.m, self.t_spoken = None, None

    def similarity(self, h, m) -> float:
        c = float((h * m).sum() / (h.norm() * m.norm() + 1e-9))
        s = (c - self.sim_diff) / max(self.sim_same - self.sim_diff, 1e-6)
        return min(max(s, 0.0), 1.0)

    def novelty(self, h, t: float) -> float:
        if self.m is None:
            return 1.0
        return 1.0 - self.similarity(h, self.m) * math.exp(-(t - self.t_spoken) / self.tau)

    def decide(self, p: float, h, t: float) -> tuple[bool, float]:
        n = self.novelty(h, t)
        speak = p * n >= self.theta
        if speak:
            self.m, self.t_spoken = h.detach().clone(), t
        return speak, n

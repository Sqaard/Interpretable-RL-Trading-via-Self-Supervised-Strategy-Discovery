"""Pure numeric helpers shared by Stage 0.1 execution controllers."""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / max(float(temperature), EPS)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / max(float(exp_z.sum()), EPS)


def normalize_simplex(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= EPS:
        out = np.zeros_like(w)
        out[-1] = 1.0
        return out
    return w / total


def normalize_stock_simplex(weights: np.ndarray) -> np.ndarray:
    """Normalize stock-only weights without sending fallback mass to cash."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= EPS:
        return np.full_like(w, 1.0 / max(len(w), 1), dtype=np.float64)
    return w / total


def sigmoid_scalar(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(float(x), -30.0, 30.0))))


def smoothstep(x: float) -> float:
    z = float(np.clip(x, 0.0, 1.0))
    return z * z * (3.0 - 2.0 * z)


def deadzone_scale(gap: float, eps: float, tau: float) -> float:
    if tau <= EPS:
        return 0.0 if gap < eps else 1.0
    return smoothstep((gap - eps) / tau)


def rank01(values: np.ndarray) -> np.ndarray:
    """Return stable cross-sectional ranks scaled to [0, 1]."""
    x = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = x.size
    if n <= 1:
        return np.zeros_like(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / float(n - 1)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector to the probability simplex."""
    x = np.asarray(v, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Simplex projection expects a 1-D vector.")
    n = x.shape[0]
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        out = np.zeros_like(x)
        out[-1] = 1.0
        return out
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(x - theta, 0.0)


def cap_and_redistribute(weights: np.ndarray, cap: float, budget: float | None = None) -> np.ndarray:
    """Cap non-negative weights and redistribute excess to uncapped names."""
    w = np.asarray(weights, dtype=np.float64)
    if budget is None:
        budget = float(w.sum())
    budget = max(float(budget), 0.0)
    if w.size == 0 or budget <= EPS:
        return np.zeros_like(w, dtype=np.float64)
    cap = max(float(cap), EPS)
    if cap * w.size + EPS < budget:
        cap = budget / w.size
    out = np.maximum(w, 0.0)
    for _ in range(16):
        over = out > cap
        if not np.any(over):
            break
        excess = float(np.sum(out[over] - cap))
        out[over] = cap
        under = ~over
        capacity = np.maximum(cap - out[under], 0.0)
        capacity_sum = float(capacity.sum())
        if capacity_sum <= EPS:
            break
        out[under] += excess * capacity / capacity_sum
    total = float(out.sum())
    if total <= EPS:
        return np.full_like(out, budget / len(out), dtype=np.float64)
    return out * (budget / total)

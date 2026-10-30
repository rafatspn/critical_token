"""Causal running statistics over a trace's response tokens (index r = 0 .. n-1).

Every function only reads entries before r (or r + 1 .. r + k for look-ahead),
so values at r are unchanged when the trace is truncated after r (+ k).
"""
from __future__ import annotations

import numpy as np

WINDOW = 64
WARMUP = 8


def window_zscore(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """z-score of x[r] against x[max(0, r - window) : r]; NaN with fewer than 2 finite earlier values."""
    out = np.full(len(x), np.nan)
    for r in range(len(x)):
        earlier = x[max(0, r - window): r]
        earlier = earlier[np.isfinite(earlier)]
        if len(earlier) >= 2:
            std = earlier.std()
            if std > 0:
                out[r] = (x[r] - earlier.mean()) / std
    return out


def warmup(n: int, minimum: int = WARMUP) -> np.ndarray:
    """True while fewer than ``minimum`` earlier response tokens exist."""
    return np.arange(n) < minimum


def ema_delta(x: np.ndarray, half_life: float = 16.0) -> np.ndarray:
    """x[r] minus the exponential moving average of x[0 .. r-1]; NaN at r = 0."""
    alpha = 1 - 2 ** (-1 / half_life)
    out = np.full(len(x), np.nan)
    ema = np.nan
    for r in range(len(x)):
        if r > 0:
            ema = x[r - 1] if r == 1 else alpha * x[r - 1] + (1 - alpha) * ema
            out[r] = x[r] - ema
    return out


def prev_delta(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[1:] = x[1:] - x[:-1]
    return out


def future(x: np.ndarray, k: int, how: str) -> np.ndarray:
    """Mean or max of x[r+1 .. r+k] over the entries that exist; NaN if none."""
    out = np.full(len(x), np.nan)
    for r in range(len(x) - 1):
        chunk = x[r + 1: r + 1 + k]
        out[r] = chunk.mean() if how == "mean" else chunk.max()
    return out


def window_phi(u: np.ndarray, strengths: np.ndarray, frobenius: float, window: int) -> np.ndarray:
    """Φ-Align (paper Eq. 4) of the covariance of rows u[max(0, j-window+1) .. j], for each row j.

    ``u`` holds each state in the head's eigenbasis (rows x d, float64); the
    window covariance is built from running sums of u and u², never as a matrix:
    tr(W_c Σ W_cᵀ) = Σ_j s_j² Var(u_j) and tr(Σ) = Σ_j Var(u_j).
    """
    s1 = np.cumsum(u, axis=0)
    s2 = np.cumsum(u * u, axis=0)
    zero = np.zeros((1, u.shape[1]))
    s1, s2 = np.vstack([zero, s1]), np.vstack([zero, s2])
    out = np.full(len(u), np.nan)
    for j in range(len(u)):
        lo = max(0, j - window + 1)
        n = j + 1 - lo
        if n < 2:
            continue
        mean = (s1[j + 1] - s1[lo]) / n
        var = np.maximum((s2[j + 1] - s2[lo]) / n - mean * mean, 0.0)
        total = var.sum()
        if total > 0:
            out[j] = float((strengths * var).sum() / (total * frobenius))
    return out


def lagged_delta(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[lag:] = x[lag:] - x[:-lag]
    return out

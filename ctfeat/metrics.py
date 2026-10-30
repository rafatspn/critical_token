"""Detector metrics with problem-level resampling.

Resampling problems with replacement is equivalent to weighting every row by its
problem's resample count, so each metric is written for row weights over a single
sort of the scores; each bootstrap resample then costs O(n).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

METRICS = ("auroc", "auprc", "fa_per_1000_at_50recall")


@dataclass
class Ranked:
    """Scores sorted once, descending, with ties collapsed into bins."""
    bins: np.ndarray       # bin of each row (0 = highest score)
    n_bins: int
    order: np.ndarray      # row indices in descending score order

    @classmethod
    def build(cls, score: np.ndarray) -> Ranked:
        score = np.asarray(score, dtype=np.float64)
        order = np.argsort(-score, kind="stable")
        _, inverse = np.unique(-score, return_inverse=True)
        return cls(bins=inverse, n_bins=int(inverse.max()) + 1 if len(inverse) else 0, order=order)


def binary_metrics(ranked: Ranked, y: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Weighted AUROC, average precision, and false alarms per 1,000 negatives at 50% recall."""
    pos = np.bincount(ranked.bins, weights=w * y, minlength=ranked.n_bins)
    neg = np.bincount(ranked.bins, weights=w * (1 - y), minlength=ranked.n_bins)
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return {name: math.nan for name in METRICS}
    neg_above = np.cumsum(neg) - neg                 # negatives scored strictly higher
    auroc = 1 - float((pos * (neg_above + 0.5 * neg)).sum() / (P * N))
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    auprc = float((pos / P * tp / np.maximum(tp + fp, 1e-12)).sum())
    first = int(np.searchsorted(tp, 0.5 * P - 1e-9))
    fa = float(fp[min(first, len(fp) - 1)] / N * 1000)
    return {"auroc": auroc, "auprc": auprc, "fa_per_1000_at_50recall": fa}


def graded_metrics(score: np.ndarray, target: np.ndarray, w: np.ndarray, *, ranks: tuple[np.ndarray, np.ndarray] | None = None,
                   threshold: float = 0.3, top: float = 0.01, precision: bool = True) -> dict[str, float]:
    """Weighted Spearman (Pearson on fixed global ranks) and precision@1% (share of the top 1% with target ≥ threshold)."""
    rs, rt = ranks if ranks is not None else (_rank(score), _rank(target))
    W = w.sum()
    if W <= 0:
        return {"spearman": math.nan, "precision_at_1pct": math.nan}
    ms, mt = (w * rs).sum() / W, (w * rt).sum() / W
    cov = (w * (rs - ms) * (rt - mt)).sum()
    var = math.sqrt((w * (rs - ms) ** 2).sum() * (w * (rt - mt) ** 2).sum())
    rho = float(cov / var) if var > 0 else math.nan
    if not precision:
        return {"spearman": rho}
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(w[order])
    cut = int(np.searchsorted(cum, top * W)) + 1
    chosen = order[:cut]
    share = float((w[chosen] * (target[chosen] >= threshold)).sum() / max(w[chosen].sum(), 1e-12))
    return {"spearman": rho, "precision_at_1pct": share}


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


def problem_index(problems: np.ndarray) -> tuple[np.ndarray, int]:
    _, index = np.unique(problems, return_inverse=True)
    return index, int(index.max()) + 1


@dataclass
class Bootstrap:
    """Problem-level bootstrap resamples as per-problem counts; the same seed gives paired resamples."""
    counts: np.ndarray     # (n_boot, n_problems)
    problems: np.ndarray   # sorted unique problem ids

    @classmethod
    def draw(cls, problems: np.ndarray, n_boot: int, seed: int) -> Bootstrap:
        unique = np.unique(problems)
        rng = np.random.default_rng(seed)
        counts = np.stack([np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique)) for _ in range(n_boot)])
        return cls(counts=counts.astype(np.float64), problems=unique)

    def weights(self, problems: np.ndarray):
        """Yield row weights for each resample, for rows with the given problem ids."""
        index = np.searchsorted(self.problems, problems)
        for counts in self.counts:
            yield counts[index]


def ci(samples: np.ndarray) -> tuple[float, float]:
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return math.nan, math.nan
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else math.nan


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Returns (adjusted p-values, rejected) for the BH procedure."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    values = p[finite]
    order = np.argsort(values)
    m = len(values)
    adjusted = values[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1)
    result = np.empty(m)
    result[order] = adjusted
    out[finite] = result
    return out, np.nan_to_num(out, nan=1.0) <= q

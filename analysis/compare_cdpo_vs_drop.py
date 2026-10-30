"""Compare intervening at cDPO's critical token vs the sharpest-drop token.

On wrong traces where ``crit_cdpo != crit_drop``, compare at each token:
  - pass@1 / pass@64 after replacement (64 sampled alternatives x 1 rollout,
    the original token excluded; cDPO's own metric);
  - pass@1 / pass@64 of freely resampling the token (64 rollouts from the
    prefix just before it, i.e. V_before).
No rollouts are run; everything comes from the finished dataset.

Usage:
    python analysis/compare_cdpo_vs_drop.py [dataset_dir] [--out analysis/results]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_DATASET = "outputs/study_qwen25_7b_math500_exhaustive/dataset"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (as in cDPO's get_pass_at_k)."""
    k = min(k, n)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test (McNemar's exact test for binary outcomes); ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(diff: np.ndarray, seed: int = 0, n_boot: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired comparison of per-trace scores a (drop) vs b (cDPO)."""
    diff = a - b
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    low, high = bootstrap_ci(diff)
    return {
        "mean_drop": float(a.mean()), "mean_cdpo": float(b.mean()), "mean_diff": float(diff.mean()),
        "ci_low": low, "ci_high": high, "drop_wins": wins, "cdpo_wins": losses,
        "ties": int(len(diff) - wins - losses), "p_sign": sign_test(wins, losses),
    }


def load(dataset: Path) -> list[dict[str, float]]:
    columns = ["trace_id", "pos", "without_sampled_n_correct", "without_sampled_n_total",
               "before_n_correct", "before_n_total", "with_n_correct", "with_n_total"]
    rows = {(r["trace_id"], r["pos"]): r for r in pq.read_table(dataset / "positions.parquet", columns=columns).to_pylist()}
    traces = [json.loads(line) for line in (dataset / "traces.jsonl").open()]
    records = []
    for trace in traces:
        cdpo, drop = trace.get("crit_cdpo"), trace.get("crit_drop")
        if trace["correct"] or cdpo is None or drop is None or cdpo == drop:
            continue
        record: dict[str, float] = {"trace_id": trace["trace_id"], "cdpo": cdpo, "drop": drop}
        for name, pos in (("cdpo", cdpo), ("drop", drop)):
            row = rows[(trace["trace_id"], pos)]
            if not row["without_sampled_n_total"]:
                raise ValueError(f"no swap result at {trace['trace_id']}:{pos} ({name})")
            for kind, prefix in (("replace", "without_sampled"), ("resample", "before")):
                n, c = int(row[f"{prefix}_n_total"]), int(row[f"{prefix}_n_correct"])
                record[f"{kind}_p1_{name}"] = c / n
                record[f"{kind}_p64_{name}"] = pass_at_k(n, c, 64)
            record[f"v_with_{name}"] = row["with_n_correct"] / row["with_n_total"]
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--out", default="analysis/results")
    args = parser.parse_args()
    dataset, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    traces = [json.loads(line) for line in (dataset / "traces.jsonl").open()]
    n_wrong = sum(not t["correct"] and t.get("crit_cdpo") is not None and t.get("crit_drop") is not None for t in traces)
    records = load(dataset)
    column = lambda key: np.array([r[key] for r in records], dtype=float)

    results = {}
    for kind in ("replace", "resample"):
        for metric in ("p1", "p64"):
            results[f"{kind}_{metric}"] = compare(column(f"{kind}_{metric}_drop"), column(f"{kind}_{metric}_cdpo"))
    # Does excluding the original token beat free resampling, at each token?
    for name in ("cdpo", "drop"):
        for metric in ("p1", "p64"):
            results[f"replace_vs_resample_{metric}_at_{name}"] = compare(column(f"replace_{metric}_{name}"), column(f"resample_{metric}_{name}"))
    lag = column("drop") - column("cdpo")

    labels = {"p1": "pass@1", "p64": "pass@64"}
    lines = [
        "# cDPO critical token vs sharpest-drop token", "",
        f"Dataset: `{dataset}`", "",
        f"Wrong traces with both labels: {n_wrong}; different tokens: {len(records)} ({len(records) / n_wrong:.0%}). "
        f"`crit_drop` comes before `crit_cdpo` in {int((lag < 0).sum())} of them (median lag {np.median(lag):+.0f} tokens).", "",
        "Replacement = 64 sampled alternatives (original token excluded) x 1 rollout. "
        "Resample = 64 rollouts from the prefix just before the token (V_before). "
        "Differences are drop − cDPO, paired per trace; 95% bootstrap CI; exact two-sided sign test on wins vs losses.", "",
        "## Drop vs cDPO", "",
        "| intervention | metric | at drop | at cDPO | diff [95% CI] | drop wins / cDPO wins / ties | p (sign) |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind in ("replace", "resample"):
        for metric in ("p1", "p64"):
            r = results[f"{kind}_{metric}"]
            lines.append(
                f"| {kind} | {labels[metric]} | {r['mean_drop']:.3f} | {r['mean_cdpo']:.3f} | "
                f"{r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                f"{r['drop_wins']} / {r['cdpo_wins']} / {r['ties']} | {r['p_sign']:.2g} |"
            )
    lines += [
        "", "## Replacement vs free resampling at the same token", "",
        "| token | metric | replace | resample | diff [95% CI] | replace wins / resample wins / ties | p (sign) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("cdpo", "drop"):
        for metric in ("p1", "p64"):
            r = results[f"replace_vs_resample_{metric}_at_{name}"]
            lines.append(
                f"| {name} | {labels[metric]} | {r['mean_drop']:.3f} | {r['mean_cdpo']:.3f} | "
                f"{r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                f"{r['drop_wins']} / {r['cdpo_wins']} / {r['ties']} | {r['p_sign']:.2g} |"
            )
    (out / "cdpo_vs_drop.md").write_text("\n".join(lines) + "\n")
    with (out / "cdpo_vs_drop_per_trace.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

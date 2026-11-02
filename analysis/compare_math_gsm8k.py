"""Side-by-side MATH vs GSM8K tables for steps 1 and 2 and the subsidiary analysis (results only).

    python analysis/compare_math_gsm8k.py [--out comparison_math_gsm8k.md]

Everything is read from finished outputs: each run's dataset (traces, positions, meta),
step 2 results (tier_models.csv, tier_gains.csv) and the subsidiary per-trace records
written by analysis/compare_cdpo_vs_drop.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from compare_cdpo_vs_drop import compare  # noqa: E402

RUNS = {
    "MATH": {"dataset": "outputs/study_qwen25_7b_math500_exhaustive/dataset", "results": "outputs/signals/results",
             "subsidiary": "analysis/results"},
    "GSM8K": {"dataset": "outputs/dataset_gsm8k", "results": "outputs/results_gsm8k", "subsidiary": "analysis/results_gsm8k"},
}
LABELS = ["is_crit_cdpo", "is_crit_drop", "is_fixable", "is_crit_good", "is_noncritical", "is_consequence", "is_ambiguous"]
STAGES = ("generate", "select", "values", "swap", "label")
TIER2 = ("T1+T2avg", "T1+T2top8", "T1+T2top32")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *["| " + " | ".join(row) + " |" for row in rows]]


# ---------------------------------------------------------------- step 1

def step1(run: dict[str, str]) -> dict[str, str]:
    dataset = REPO / run["dataset"]
    traces = [json.loads(line) for line in (dataset / "traces.jsonl").open() if line.strip()]
    meta = json.loads((dataset / "meta.json").read_text())
    positions = pq.read_table(dataset / "positions.parquet", columns=[*LABELS, "entropy_T1", "entropy_decile", "engine_mismatch"]).to_pandas()
    wrong = [t for t in traces if not t["correct"]]
    both = [t for t in wrong if t.get("crit_drop") is not None and t.get("crit_cdpo") is not None]
    lag = np.array([t["crit_drop"] - t["crit_cdpo"] for t in both], dtype=float)
    critical = positions[positions.is_crit_cdpo | positions.is_crit_drop | positions.is_crit_good]
    noncritical = positions[positions.is_noncritical]
    share = lambda mask: f"{float(np.mean(mask)):.1%}" if len(mask) else "n/a"
    stat = lambda values, f: f"{f(values):.4f}" if len(values) else "n/a"
    out: dict[str, str] = {
        "problems": str(len({t["problem_id"] for t in traces})),
        "wrong traces / right traces": f"{len(wrong)} / {len(traces) - len(wrong)}",
        "positions": str(len(positions)),
    }
    out.update({f"`{label}`": str(int(positions[label].sum())) for label in LABELS})
    out.update({
        "wrong traces with both labels": str(len(both)),
        "`crit_drop` = `crit_cdpo`": share(lag == 0),
        "`crit_drop` before `crit_cdpo`": share(lag < 0),
        "lag `crit_drop − crit_cdpo`: mean / median": f"{lag.mean():.2f} / {np.median(lag):.1f}" if len(lag) else "n/a",
        "fixable at `crit_cdpo`": share(positions.loc[positions.is_crit_cdpo, "is_fixable"]),
        "fixable at `crit_drop`": share(positions.loc[positions.is_crit_drop, "is_fixable"]),
        "critical entropy mean / median": f"{stat(critical.entropy_T1, statistics.fmean)} / {stat(critical.entropy_T1, statistics.median)}",
        "non-critical entropy mean / median": f"{stat(noncritical.entropy_T1, statistics.fmean)} / {stat(noncritical.entropy_T1, statistics.median)}",
        "critical tokens in bottom two entropy quintiles": share(critical.entropy_decile <= 3),
        "`crit_drop` in bottom two entropy quintiles": share(positions.loc[positions.is_crit_drop, "entropy_decile"] <= 3),
        "`crit_cdpo` in bottom two entropy quintiles": share(positions.loc[positions.is_crit_cdpo, "entropy_decile"] <= 3),
    })
    mismatch = positions.dropna(subset=["engine_mismatch"])
    crit_mismatch = critical.dropna(subset=["engine_mismatch"])
    out["`engine_mismatch`: all / critical positions"] = f"{share(mismatch.engine_mismatch.astype(bool))} / {share(crit_mismatch.engine_mismatch.astype(bool))}"
    tokens, hours = meta.get("generated_tokens_by_stage", {}), meta.get("gpu_hours_by_stage", {})
    for stage in STAGES:
        out[f"compute `{stage}`: tokens / GPU-hours"] = f"{int(tokens.get(stage, 0)):,} / {float(hours.get(stage, 0)):.1f}"
    out["compute total: tokens / GPU-hours"] = f"{int(meta.get('generated_tokens_total', 0)):,} / {sum(map(float, hours.values())):.1f}"
    return out


# ---------------------------------------------------------------- step 2

def cell(row: pd.Series | None) -> str:
    return "n/a" if row is None else f"{row['value']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def pick(tier: pd.DataFrame, model: str, delay: str, target: str, pool: str, metric: str, subset: str = "all") -> pd.Series | None:
    sel = tier[(tier.model == model) & (tier.delay == delay) & (tier.target == target) & (tier.pool == pool)
               & (tier.subset == subset) & (tier.mismatch == "with_mismatch") & (tier.metric == metric)]
    return sel.iloc[0] if len(sel) else None


def best_tier2(tier: pd.DataFrame, delay: str, target: str, metric: str) -> str | None:
    pool = "wrong_positions" if target == "B" else "all_other"
    scored = [(row["value"], name) for name in TIER2 if (row := pick(tier, name, delay, target, pool, metric)) is not None
              and np.isfinite(row["value"])]
    return max(scored)[1] if scored else None


def step2_rows(tiers: dict[str, pd.DataFrame], target: str, delay: str) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    metric = "spearman" if target == "B" else "auroc"
    pool = "wrong_positions" if target == "B" else "all_other"
    models: list[tuple[str, Callable[[str], str | None]]] = [
        ("entropy (`ent__pre`)", lambda name: "baseline:ent__pre"),
        ("−log p(chosen)", lambda name: "baseline:neg_logp_chosen__pre"),
        ("Tier 1", lambda name: "T1"),
        ("best Tier 2", lambda name: best_tier2(tiers[name], delay, target, metric)),
        ("Tier 3 (T1+T3)", lambda name: "T1+T3"),
        ("all tiers", lambda name: "all"),
    ]
    main_rows, fork_rows, alarm_rows = [], [], []
    for label, model_of in models:
        names = {run: model_of(run) for run in tiers}
        # Baselines are scored at pre only; at post they are the same rows.
        lookup = {run: (names[run], "pre" if (names[run] or "").startswith("baseline:") else delay) for run in tiers}
        main_rows.append([label, *[f"{cell(pick(tiers[run], names[run], lookup[run][1], target, pool, metric))}"
                                   + (f" ({names[run]})" if label == "best Tier 2" and names[run] else "") for run in tiers]])
        if target != "B":
            fork_rows.append([label, *[cell(pick(tiers[run], names[run], lookup[run][1], target, "neg_benign_fork", "auroc")) for run in tiers]])
            alarm_rows.append([label, *[cell(pick(tiers[run], names[run], lookup[run][1], target, "all_other", "fa_per_1000_at_50recall"))
                                        for run in tiers]])
    return main_rows, fork_rows, alarm_rows


def gains_rows(gains: dict[str, pd.DataFrame], target: str, delay: str) -> list[list[str]]:
    rows = []
    for model in (*TIER2, "T1+T3", "all"):
        cells = []
        for run, table_ in gains.items():
            sel = table_[(table_.model == model) & (table_.delay == delay) & (table_.target == target)]
            cells.append("n/a" if sel.empty else f"{sel.iloc[0].auroc_gain_over_T1:+.3f} [{sel.iloc[0].ci_low:+.3f}, {sel.iloc[0].ci_high:+.3f}]")
        rows.append([model, *cells])
    return rows


# ---------------------------------------------------------------- subsidiary

def subsidiary(run: dict[str, str]) -> dict[str, dict[str, float]] | None:
    path = REPO / run["subsidiary"] / "cdpo_vs_drop_per_trace.jsonl"
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.open() if line.strip()]
    column = lambda key: np.array([r[key] for r in records], dtype=float)
    out = {"n": {"traces": len(records)}}
    for kind in ("replace", "resample"):
        for metric in ("p1", "p64"):
            out[f"{kind}_{metric}"] = compare(column(f"{kind}_{metric}_drop"), column(f"{kind}_{metric}_cdpo"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "comparison_math_gsm8k.md")
    parser.add_argument("--runs", type=Path, help="JSON overriding RUNS (for testing)")
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text()) if args.runs else RUNS
    names = list(runs)
    lines = ["# MATH vs GSM8K: step 1, step 2 and the subsidiary analysis", "",
             "Results only. Sources: " + "; ".join(f"{name}: `{run['dataset']}`, `{run['results']}`, `{run['subsidiary']}`"
                                                   for name, run in runs.items()), ""]

    lines += ["## Step 1", ""]
    s1 = {name: step1(run) for name, run in runs.items()}
    lines += table(["", *names], [[key, *[s1[name].get(key, "n/a") for name in names]] for key in s1[names[0]]])

    tiers = {name: pd.read_csv(REPO / run["results"] / "tier_models.csv") for name, run in runs.items()}
    gains = {name: pd.read_csv(REPO / run["results"] / "tier_gains.csv") for name, run in runs.items()}
    lines += ["", "## Step 2", "",
              "Pool 1 (all other positions in the target's traces), all positions, engine-mismatch positions included; "
              "values with 95% problem-level bootstrap CIs. Targets A/C/D: AUROC; B: Spearman with `drop`. "
              "Baselines are scored at `pre` and repeated at `post`. Best Tier 2 is the highest of "
              + ", ".join(f"`{m}`" for m in TIER2) + " on each dataset (variant in brackets)."]
    titles = {"A": "A (`is_crit_drop`)", "B": "B (`drop`, graded)", "C": "C (`is_crit_cdpo`)", "D": "D (`is_crit_good`, right traces)"}
    for target, title in titles.items():
        for delay in ("pre", "post"):
            main_rows, fork_rows, alarm_rows = step2_rows(tiers, target, delay)
            lines += ["", f"### Target {title}, delay `{delay}`", ""]
            lines += table(["model", *[f"{n} {'Spearman' if target == 'B' else 'AUROC'}" for n in names]], main_rows)
            if target != "B":
                lines += ["", "AUROC gain over Tier 1 (paired bootstrap, pool 1):", ""]
                lines += table(["model", *names], gains_rows(gains, target, delay))
                lines += ["", "AUROC against benign forks (`neg_benign_fork` pool):", ""]
                lines += table(["model", *names], fork_rows)
                lines += ["", "False alarms per 1,000 tokens at 50% recall (pool 1):", ""]
                lines += table(["model", *names], alarm_rows)

    lines += ["", "## Subsidiary: sharpest-drop vs cDPO token (traces where the two differ)", "",
              "Replacement = 64 sampled alternatives × 1 rollout; resample = 64 rollouts from the prefix before the token. "
              "Diff = drop − cDPO, paired per trace, 95% bootstrap CI.", ""]
    subs = {name: subsidiary(run) for name, run in runs.items()}
    rows = [["traces", *[str(subs[n]["n"]["traces"]) if subs[n] else "n/a" for n in names]]]
    for kind in ("replace", "resample"):
        for metric, label in (("p1", "pass@1"), ("p64", "pass@64")):
            row = [f"{kind} {label}: drop / cDPO / diff [CI]"]
            for name in names:
                r = subs[name][f"{kind}_{metric}"] if subs[name] else None
                row.append("n/a" if r is None else f"{r['mean_drop']:.3f} / {r['mean_cdpo']:.3f} / {r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
            rows.append(row)
    lines += table(["", *names], rows)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

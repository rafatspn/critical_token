"""Evaluation of online critical-token signals (Section 4).

    python -m ctfeat.evaluate --dataset dataset/ --features features/ --out results/

Every statistic is grouped by problem_id: CV folds and bootstrap resamples (1,000) are
drawn over problems.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .features import ATTN_STATS
from .metrics import METRICS, Bootstrap, Ranked, _rank, benjamini_hochberg, binary_metrics, ci, graded_metrics, two_sided_p
from .models import GroupedLogistic, GroupedRidge, TopHeads, folds, out_of_fold, preprocess
from .preflight import check_labels
from .registry import DELAYS, allowed, delay_of

LABELS = ["is_crit_cdpo", "is_crit_drop", "is_fixable", "is_crit_good", "is_noncritical", "is_consequence", "is_ambiguous"]
TARGETS = {"A": "is_crit_drop", "C": "is_crit_cdpo", "D": "is_crit_good"}
BASELINES = ("ent__pre", "pmax__pre", "neg_logp_chosen__pre", "ent_z__pre", "random")
HEAD_STATS_MAIN = ("att_ent", "lookback_ratio", "att_recent16", "att_sink")
PROBE_LAYERS = (4, 8, 12, 16, 20, 24, 28)


# ---------------------------------------------------------------- inputs

def load_inputs(dataset: Path, features: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    check_labels(dataset)
    traces = []
    with (dataset / "traces.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            traces.append({key: row.get(key) for key in ("trace_id", "problem_id", "correct", "crit_drop", "crit_cdpo", "response_ids")})
    columns = ["trace_id", "pos", "v_before", "v_with", "entropy_T1", "token_type", "engine_mismatch", *LABELS]
    labels = pq.read_table(dataset / "positions.parquet", columns=columns).to_pandas()
    info = pd.DataFrame([{k: t[k] for k in ("trace_id", "problem_id", "correct")} for t in traces])
    labels = labels.merge(info, on="trace_id")
    labels["drop"] = labels["v_before"] - labels["v_with"]
    frame = labels
    for table in ("tier1", "tier2", "tier3_derived"):
        path = features / f"{table}.parquet"
        if path.exists():
            frame = frame.merge(pq.read_table(path).to_pandas(), on=["trace_id", "pos"], how="inner")
    frame = frame.sort_values(["trace_id", "pos"]).reset_index(drop=True)
    frame["neg_logp_chosen__pre"] = -frame["logp_chosen__pre"]
    frame["random"] = np.random.default_rng(0).random(len(frame))
    negatives = pq.read_table(dataset / "negatives.parquet").to_pandas()
    return frame, traces, negatives


def feature_columns(frame: pd.DataFrame, manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_column = {item["column"]: item for item in manifest}
    return {column: by_column[column] for column in frame.columns if column in by_column}


def model_matrix(frame: pd.DataFrame, columns: list[str], n_layers: int) -> np.ndarray:
    """Feature matrix; +inf ('never', e.g. commit layers) becomes n_layers + 1."""
    X = frame[columns].to_numpy(dtype=np.float64, copy=True)
    X[np.isposinf(X)] = n_layers + 1
    X[np.isneginf(X)] = np.nan
    return X


def load_head_matrix(features: Path, frame: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Per-head attention features for every frame row (float16).

    Row statistics per (layer, head, stat) at pre (row r) and post (row r+1), and attention
    received per (layer, head) at delays 4/16/64. Returns (matrix, column names, head id per
    column), where head id = (layer - 1) * heads + head.
    """
    blocks = []
    names: list[str] = []
    head_of = np.zeros(0, dtype=int)
    for trace_id, group in frame.groupby("trace_id", sort=False):
        safe = trace_id.replace(":", "_").replace("/", "_")
        stats = np.load(features / "attn" / f"{safe}.npy", mmap_mode="r")             # rows x layers x heads x stats
        received = np.load(features / "attn_recv" / f"{safe}.npy", mmap_mode="r")     # tokens x delays x layers x heads
        pos = group["pos"].to_numpy()
        parts = [np.asarray(stats[pos + offset]).reshape(len(pos), -1) for offset in (0, 1)]
        parts.append(np.asarray(received[pos]).reshape(len(pos), -1))
        blocks.append(np.concatenate(parts, 1).astype(np.float16))
        if not names:
            _, layers, heads, n_stats = stats.shape
            n_delays = received.shape[1]
            for delay in ("pre", "post"):
                for l in range(layers):
                    for h in range(heads):
                        names += [f"{ATTN_STATS[s]}_L{l + 1}_H{h}__{delay}" for s in range(n_stats)]
            for d, k in enumerate((4, 16, 64)[:n_delays]):
                for l in range(layers):
                    names += [f"att_recv_L{l + 1}_H{h}__k{k}" for h in range(heads)]
            head_of = np.array([(int(n.split("_L")[1].split("_H")[0]) - 1) * heads + int(n.split("_H")[1].split("__")[0]) for n in names])
    return np.concatenate(blocks), names, head_of


def load_hidden(features: Path, frame: pd.DataFrame, layer_index: int, delay: str) -> np.ndarray:
    out = []
    for trace_id, group in frame.groupby("trace_id", sort=False):
        hidden = np.load(features / "hidden" / f"{trace_id.replace(':', '_').replace('/', '_')}.npy", mmap_mode="r")
        rows = group["pos"].to_numpy() + (1 if delay == "post" else 0)
        out.append(np.asarray(hidden[layer_index, rows], dtype=np.float32))
    return np.concatenate(out)


# ---------------------------------------------------------------- pools and metrics

def pools_for(frame: pd.DataFrame, target: str, negatives: pd.DataFrame) -> dict[str, np.ndarray]:
    """Negative masks per comparison pool (over all frame rows) for a binary target."""
    label = TARGETS[target]
    population = ~frame["correct"] if target in ("A", "C") else frame["correct"]
    pos = frame[label].to_numpy(bool) & population.to_numpy(bool)
    inside = population.to_numpy(bool)
    pools = {"all_other": inside & ~pos,
             "noncritical": frame["is_noncritical"].to_numpy(bool) & inside & ~pos}
    keys = pd.MultiIndex.from_frame(frame[["trace_id", "pos"]])
    right = frame["correct"].to_numpy(bool)
    for group, chunk in negatives[negatives["crit_label_kind"] == label].groupby("group"):
        wanted = pd.MultiIndex.from_frame(chunk[["neg_trace_id", "neg_pos"]].drop_duplicates().set_axis(["trace_id", "pos"], axis=1))
        # Pools stay inside the target's population, except the correct_trace group of
        # targets A/C, whose negatives are right-trace positions by construction.
        scope = right if (group == "correct_trace" and target in ("A", "C")) else inside
        pools[f"neg_{group}"] = keys.isin(wanted) & scope & ~pos
    return pools


def evaluate_binary(score: np.ndarray, frame: pd.DataFrame, target: str, pools: dict[str, np.ndarray], boot: Bootstrap,
                    ent_median: float, extra: dict[str, Any]) -> list[dict[str, Any]]:
    """AUROC / AUPRC / false alarms per pool, overall and on the low-entropy subset, with and without engine_mismatch."""
    label = TARGETS[target]
    population = ~frame["correct"] if target in ("A", "C") else frame["correct"]
    pos = frame[label].to_numpy(bool) & population.to_numpy(bool)
    low = frame["ent__pre"].to_numpy() < ent_median
    clean = ~frame["engine_mismatch"].fillna(False).to_numpy(bool)
    problems = frame["problem_id"].astype(str).to_numpy()
    rows = []
    for pool, neg in pools.items():
        for subset, mask in (("all", np.ones(len(frame), bool)), ("low_entropy", low)):
            for mismatch, keep in (("with_mismatch", np.ones(len(frame), bool)), ("no_mismatch", clean)):
                sel = (pos | neg) & mask & keep & np.isfinite(score)
                if pos[sel].sum() == 0 or neg[sel].sum() == 0:
                    continue
                y, s, g = pos[sel].astype(float), score[sel], problems[sel]
                ranked = Ranked.build(s)
                point = binary_metrics(ranked, y, np.ones(len(y)))
                samples = np.array([list(binary_metrics(ranked, y, w).values()) for w in boot.weights(g)])
                for index, metric in enumerate(METRICS):
                    low_ci, high_ci = ci(samples[:, index])
                    rows.append({**extra, "target": target, "pool": pool, "subset": subset, "mismatch": mismatch, "metric": metric,
                                 "value": point[metric], "ci_low": low_ci, "ci_high": high_ci,
                                 "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())})
    return rows


def evaluate_graded(score: np.ndarray, frame: pd.DataFrame, boot: Bootstrap, extra: dict[str, Any]) -> list[dict[str, Any]]:
    wrong = ~frame["correct"].to_numpy(bool)
    clean = ~frame["engine_mismatch"].fillna(False).to_numpy(bool)
    rows = []
    for mismatch, keep in (("with_mismatch", np.ones(len(frame), bool)), ("no_mismatch", clean)):
        sel = wrong & keep & np.isfinite(score)
        s, t, g = score[sel], frame["drop"].to_numpy()[sel], frame["problem_id"].astype(str).to_numpy()[sel]
        point = graded_metrics(s, t, np.ones(len(s)))
        samples = np.array([[v for v in graded_metrics(s, t, w).values()] for w in boot.weights(g)])
        for index, metric in enumerate(point):
            low_ci, high_ci = ci(samples[:, index])
            rows.append({**extra, "target": "B", "pool": "wrong_positions", "subset": "all", "mismatch": mismatch,
                         "metric": metric, "value": point[metric], "ci_low": low_ci, "ci_high": high_ci, "n": int(sel.sum())})
    return rows


# ---------------------------------------------------------------- 4.3.1 univariate screen

def _screen_chunk(block: np.ndarray, names: list[str], y: np.ndarray, drop: np.ndarray, drop_rank: np.ndarray,
                  counts: np.ndarray, index: np.ndarray) -> list[tuple[str, float, np.ndarray, float, np.ndarray]]:
    """Point AUROC / Spearman and their bootstrap samples for a block of columns."""
    out = []
    for j, column in enumerate(names):
        x = block[:, j].astype(np.float64)
        finite = np.isfinite(x)
        if finite.sum() < 10:
            continue
        top = x[finite].max()
        x = np.where(np.isposinf(x), top + 1, x)
        x = np.where(np.isfinite(x), x, np.median(x[np.isfinite(x)]))
        if x.std() == 0:
            continue
        ranked, ranks = Ranked.build(x), (_rank(x), drop_rank)
        auc = binary_metrics(ranked, y, np.ones(len(y)))["auroc"]
        rho = graded_metrics(x, drop, np.ones(len(x)), ranks=ranks, precision=False)["spearman"]
        auc_boot = np.empty(len(counts))
        rho_boot = np.empty(len(counts))
        for b, problem_counts in enumerate(counts):
            w = problem_counts[index]
            auc_boot[b] = binary_metrics(ranked, y, w)["auroc"]
            rho_boot[b] = graded_metrics(x, drop, w, ranks=ranks, precision=False)["spearman"]
        out.append((column, auc, auc_boot, rho, rho_boot))
    return out


def univariate_screen(frame: pd.DataFrame, columns: dict[str, np.ndarray], meta: dict[str, dict[str, Any]],
                      boot: Bootstrap, n_jobs: int | None = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AUROC (target A, pool 1) and Spearman with drop (target B) per scalar feature, sign-flipped so higher
    means more critical, with problem-level bootstrap 95% CIs; p-values from the bootstrap SE (Wald),
    Benjamini–Hochberg at q = 0.05 across features."""
    from joblib import Parallel, delayed

    wrong = ~frame["correct"].to_numpy(bool)
    y = frame["is_crit_drop"].to_numpy(bool)[wrong].astype(float)
    drop = frame["drop"].to_numpy(dtype=float)[wrong]
    problems = frame["problem_id"].astype(str).to_numpy()[wrong]
    index = np.searchsorted(boot.problems, problems)
    drop_rank = _rank(drop)
    names = list(columns)
    chunks = [names[i: i + 200] for i in range(0, len(names), 200)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_screen_chunk)(np.column_stack([np.asarray(columns[c])[wrong] for c in chunk]), chunk, y, drop, drop_rank, boot.counts, index)
        for chunk in chunks)
    rows_a, rows_b = [], []
    for column, auc, auc_boot, rho, rho_boot in (item for chunk in results for item in chunk):
        info = meta.get(column, {})
        base = {"feature": column, "tier": info.get("tier"), "delay": delay_of(column) if "__" in column else None,
                "family": info.get("family"), "layer": info.get("layer"), "head": info.get("head")}
        for rows, key, point, samples, null in ((rows_a, "auroc", auc, auc_boot, 0.5), (rows_b, "spearman", rho, rho_boot, 0.0)):
            sign = 1 if point >= null else -1
            flip = (lambda v: v) if sign == 1 else ((lambda v: 1 - v) if key == "auroc" else (lambda v: -v))
            se = float(np.nanstd(samples, ddof=1))
            low, high = ci(flip(samples))
            rows.append({**base, key: flip(point), "sign": sign, "se": se, "ci_low": low, "ci_high": high,
                         "p": two_sided_p((point - null) / se) if se > 0 else math.nan})
    out = []
    for rows, key in ((rows_a, "auroc"), (rows_b, "spearman")):
        table = pd.DataFrame(rows)
        table["q"], table["significant"] = benjamini_hochberg(table["p"].to_numpy())
        out.append(table.sort_values(key, ascending=False).reset_index(drop=True))
    return out[0], out[1]


# ---------------------------------------------------------------- 4.3.2 tier models

def tier_feature_sets(meta: dict[str, dict[str, Any]], limit: str) -> dict[str, list[str]]:
    """Columns per tier allowed at a delay limit (tier-1 identity columns included)."""
    pick = lambda tiers: [c for c, info in meta.items() if info["tier"] in tiers and allowed(info["delay"], limit)
                          and info["name"] != "warmup" and info["name"] != "phi_warmup"]
    t1 = pick({1})
    t2 = [c for c in pick({2})]
    t3 = [c for c in pick({3})]
    return {"T1": t1, "T1+T2avg": t1 + t2, "T1+T3": t1 + t3, "all": t1 + t2 + t3}


def build_models(frame: pd.DataFrame, meta: dict[str, dict[str, Any]], limit: str,
                 head_matrix: tuple[np.ndarray, list[str], np.ndarray] | None, n_layers: int) -> dict[str, tuple[np.ndarray, list]]:
    """Feature matrix and preprocessing per tier model, using only features allowed at ``limit``.

    T1+T2top{8,32} add every per-head column (row statistics, and attention received at
    delays ≤ limit) of the H heads chosen inside each training fold.
    """
    models: dict[str, tuple[np.ndarray, list]] = {name: (model_matrix(frame, cols, n_layers), preprocess())
                                                  for name, cols in tier_feature_sets(meta, limit).items()}
    if head_matrix is not None:
        heads, names, head_of = head_matrix
        keep = np.array([i for i, name in enumerate(names) if allowed(delay_of(name), limit)])
        base = models["T1"][0]
        extra = heads[:, keep].astype(np.float64)
        for H in (8, 32):
            models[f"T1+T2top{H}"] = (np.concatenate([base, extra], 1),
                                      [("select", TopHeads(base.shape[1], head_of[keep], H)), *preprocess()])
    return models


def run_tier_models(frame: pd.DataFrame, meta: dict[str, dict[str, Any]], negatives: pd.DataFrame, features: Path,
                    boot: Bootstrap, n_layers: int, seed: int, n_jobs: int | None,
                    head_matrix: tuple[np.ndarray, list[str], np.ndarray] | None) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    groups = frame["problem_id"].astype(str).to_numpy()
    splits = folds(groups, 5, seed)
    ent_median = float(np.nanmedian(frame["ent__pre"]))
    rows, oof = [], {}
    for limit in ("pre", "post"):
        models = build_models(frame, meta, limit, head_matrix, n_layers)
        for target in ("A", "C", "D", "B"):
            train_mask = (~frame["correct"] if target != "D" else frame["correct"]).to_numpy(bool)
            for name, (X, steps) in models.items():
                if target == "B":
                    y = frame["drop"].to_numpy(dtype=float)
                    score = out_of_fold_masked(X, np.nan_to_num(y), groups, GroupedRidge(seed=seed), steps, splits, train_mask)
                    rows += evaluate_graded(score, frame, boot, {"model": name, "delay": limit})
                else:
                    y = frame[TARGETS[target]].to_numpy(bool).astype(int)
                    score = out_of_fold_masked(X, y, groups, GroupedLogistic(seed=seed, n_jobs=n_jobs), steps, splits, train_mask)
                    rows += evaluate_binary(score, frame, target, pools_for(frame, target, negatives), boot, ent_median,
                                            {"model": name, "delay": limit})
                oof[(name, limit, target)] = score
                print(f"tier model {name} @ {limit} target {target} done", flush=True)
    return pd.DataFrame(rows), oof


def out_of_fold_masked(X: np.ndarray, y: np.ndarray, groups: np.ndarray, estimator: Any, steps: list,
                       splits: list[tuple[np.ndarray, np.ndarray]], train_mask: np.ndarray) -> np.ndarray:
    """OOF scores where training uses only ``train_mask`` rows of the training fold; all test-fold rows are scored."""
    masked = [(train[train_mask[train]], test) for train, test in splits]
    return out_of_fold(X, y, groups, estimator, steps, masked)


def tier_gains(oof: dict[tuple[str, str, str], np.ndarray], frame: pd.DataFrame, boot: Bootstrap) -> pd.DataFrame:
    """Paired-bootstrap AUROC gain of each model over Tier 1 (pool 1)."""
    rows = []
    for (name, limit, target), score in oof.items():
        if name == "T1" or target == "B":
            continue
        base = oof[("T1", limit, target)]
        population = (~frame["correct"] if target in ("A", "C") else frame["correct"]).to_numpy(bool)
        sel = population & np.isfinite(score) & np.isfinite(base)
        y = frame[TARGETS[target]].to_numpy(bool)[sel].astype(float)
        g = frame["problem_id"].astype(str).to_numpy()[sel]
        r_model, r_base = Ranked.build(score[sel]), Ranked.build(base[sel])
        one = np.ones(len(y))
        point = binary_metrics(r_model, y, one)["auroc"] - binary_metrics(r_base, y, one)["auroc"]
        samples = np.array([binary_metrics(r_model, y, w)["auroc"] - binary_metrics(r_base, y, w)["auroc"] for w in boot.weights(g)])
        low, high = ci(samples)
        rows.append({"model": name, "delay": limit, "target": target, "auroc_gain_over_T1": point, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4.3.3 probes

def run_probes(frame: pd.DataFrame, features: Path, hidden_layers: list[int], boot: Bootstrap, seed: int, n_jobs: int | None) -> pd.DataFrame:
    wrong = ~frame["correct"].to_numpy(bool)
    sub = frame[wrong].reset_index(drop=True)
    groups = sub["problem_id"].astype(str).to_numpy()
    splits = folds(groups, 5, seed)
    rows = []
    for index, layer in enumerate(hidden_layers):
        for delay in ("pre", "post"):
            X = load_hidden(features, sub, index, delay).astype(np.float64)
            steps_layer = preprocess(pca=256)
            for target in ("A", "C"):
                y = sub[TARGETS[target]].to_numpy(bool).astype(int)
                score = out_of_fold(X, y, groups, GroupedLogistic(seed=seed, n_jobs=n_jobs), steps_layer, splits)
                ok = np.isfinite(score)
                ranked, yy = Ranked.build(score[ok]), y[ok].astype(float)
                point = binary_metrics(ranked, yy, np.ones(len(yy)))["auroc"]
                low, high = ci(np.array([binary_metrics(ranked, yy, w)["auroc"] for w in boot.weights(groups[ok])]))
                row = {"layer": layer, "delay": delay, "target": target, "auroc": point, "ci_low": low, "ci_high": high}
                if target == "A":
                    probe_score = score[ok]
                    probe_drop = sub["drop"].to_numpy()[ok]
                    probe_groups = groups[ok]
                    row["spearman_drop"] = graded_metrics(probe_score, probe_drop, np.ones(int(ok.sum())))["spearman"]
                    samples = np.array([graded_metrics(probe_score, probe_drop, w)["spearman"]
                                        for w in boot.weights(probe_groups)])
                    row["spearman_ci_low"], row["spearman_ci_high"] = ci(samples)
                rows.append(row)
            print(f"probe layer {layer} {delay} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4.3.4 delay curve, 4.3.5 breakdowns

def run_delay_curve(frame: pd.DataFrame, meta: dict[str, dict[str, Any]], boot: Bootstrap, n_layers: int, seed: int, n_jobs: int | None,
                    head_matrix: tuple[np.ndarray, list[str], np.ndarray] | None = None) -> pd.DataFrame:
    """Best cross-validated model at each delay, with one line per tier family.

    Tier 2 chooses among head-averaged, top-8 and top-32 variants independently for
    targets A and B. ``chosen_model`` records the winning concrete variant.
    """
    groups = frame["problem_id"].astype(str).to_numpy()
    splits = folds(groups, 5, seed)
    wrong = (~frame["correct"]).to_numpy(bool)
    rows = []
    for limit in DELAYS:
        candidates_a: dict[str, dict[str, Any]] = {}
        candidates_b: dict[str, dict[str, Any]] = {}
        for name, (X, steps) in build_models(frame, meta, limit, head_matrix, n_layers).items():
            y = frame["is_crit_drop"].to_numpy(bool).astype(int)
            score = out_of_fold_masked(X, y, groups, GroupedLogistic(seed=seed, n_jobs=n_jobs), steps, splits, wrong)
            ok = wrong & np.isfinite(score)
            ranked = Ranked.build(score[ok])
            ya, ga = y[ok].astype(float), groups[ok]
            point = binary_metrics(ranked, ya, np.ones(len(ya)))["auroc"]
            low, high = ci(np.array([binary_metrics(ranked, ya, w)["auroc"] for w in boot.weights(ga)]))
            candidates_a[name] = {"delay": limit, "chosen_model": name, "target": "A", "metric": "auroc",
                                  "value": point, "ci_low": low, "ci_high": high}
            drop = np.nan_to_num(frame["drop"].to_numpy(dtype=float))
            score_b = out_of_fold_masked(X, drop, groups, GroupedRidge(seed=seed), steps, splits, wrong)
            ok_b = wrong & np.isfinite(score_b)
            sb, tb = score_b[ok_b], drop[ok_b]
            point_b = graded_metrics(sb, tb, np.ones(len(sb)))["spearman"]
            low_b, high_b = ci(np.array([graded_metrics(sb, tb, w)["spearman"] for w in boot.weights(groups[ok_b])]))
            candidates_b[name] = {"delay": limit, "chosen_model": name, "target": "B", "metric": "spearman",
                                  "value": point_b, "ci_low": low_b, "ci_high": high_b}
            print(f"delay curve {limit} {name} done", flush=True)
        families = {
            "Tier 1": ["T1"],
            "Tier 2": ["T1+T2avg", "T1+T2top8", "T1+T2top32"],
            "Tier 3": ["T1+T3"],
            "All": ["all"],
        }
        for target_candidates in (candidates_a, candidates_b):
            for tier, names in families.items():
                available = [target_candidates[name] for name in names if name in target_candidates]
                finite = [row for row in available if np.isfinite(row["value"])]
                if not finite:
                    continue
                best = max(finite, key=lambda row: row["value"])
                rows.append({**best, "tier": tier})
    return pd.DataFrame(rows)


def run_breakdowns(frame: pd.DataFrame, oof: dict[tuple[str, str, str], np.ndarray], tier_table: pd.DataFrame,
                   boot: Bootstrap) -> pd.DataFrame:
    wrong = (~frame["correct"]).to_numpy(bool)
    quintile = pd.qcut(frame["ent__pre"].rank(method="first"), 5, labels=False).to_numpy()
    y = frame["is_crit_drop"].to_numpy(bool).astype(float)
    rows = []
    for limit in ("pre", "post"):
        sel = tier_table[(tier_table.delay == limit) & (tier_table.target == "A") & (tier_table.pool == "all_other")
                         & (tier_table.subset == "all") & (tier_table.mismatch == "with_mismatch") & (tier_table.metric == "auroc")]
        sel = sel[sel.model.isin({key[0] for key in oof}) & sel.value.notna()]
        if sel.empty:
            continue
        best = sel.sort_values("value", ascending=False).iloc[0]["model"]
        score = oof[(best, limit, "A")]
        for kind, values in (("token_type", frame["token_type"].to_numpy()), ("entropy_quintile", quintile)):
            for value in pd.unique(values):
                mask = wrong & (values == value) & np.isfinite(score)
                if 0 < y[mask].sum() < mask.sum():
                    ranked = Ranked.build(score[mask])
                    auc = binary_metrics(ranked, y[mask], np.ones(mask.sum()))["auroc"]
                    groups = frame["problem_id"].astype(str).to_numpy()[mask]
                    low, high = ci(np.array([binary_metrics(ranked, y[mask], w)["auroc"] for w in boot.weights(groups)]))
                    rows.append({"delay": limit, "model": best, "breakdown": kind, "value": str(value), "auroc": auc,
                                 "ci_low": low, "ci_high": high, "n_pos": int(y[mask].sum()), "n": int(mask.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def heatmaps(univ: pd.DataFrame, frame: pd.DataFrame, meta: dict[str, dict[str, Any]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figures / f"{name}.{ext}", dpi=150)
        plt.close(fig)

    tier3 = univ[(univ.tier == 3) & univ.layer.notna() & (univ.delay == "pre")].copy()
    tier3["stat"] = tier3.feature.str.replace(r"_L\d+__pre$", "", regex=True)
    tier3 = tier3[~tier3.stat.str.startswith("dla_")]
    if len(tier3):
        grid = tier3.pivot_table(index="stat", columns="layer", values="auroc")
        fig, ax = plt.subplots(figsize=(12, 0.35 * len(grid) + 2))
        image = ax.imshow(grid.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(grid)), grid.index)
        ax.set_xticks(range(len(grid.columns)), [int(c) for c in grid.columns])
        ax.set_xlabel("layer")
        ax.set_title("Tier 3 derived features: AUROC (target A, pool 1, sign-flipped, pre)")
        fig.colorbar(image)
        save(fig, "heatmap_tier3_layer_stat")
    wrong = ~frame["correct"].to_numpy(bool)
    y = frame["is_crit_drop"].to_numpy(bool)
    layers = sorted({int(info["layer"]) for c, info in meta.items() if info["name"].startswith("dla_attn_L")})
    if layers:
        diff = np.array([[np.nanmean(frame[f"dla_{kind}_L{l}__pre"].to_numpy()[wrong & y]) - np.nanmean(frame[f"dla_{kind}_L{l}__pre"].to_numpy()[wrong & ~y])
                          for kind in ("attn", "mlp")] for l in layers])
        fig, ax = plt.subplots(figsize=(4, 0.25 * len(layers) + 2))
        bound = np.nanmax(np.abs(diff)) or 1
        image = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_yticks(range(len(layers)), layers)
        ax.set_xticks([0, 1], ["attention", "MLP"])
        ax.set_ylabel("layer")
        ax.set_title("DLA: mean at positives − negatives")
        fig.colorbar(image)
        save(fig, "heatmap_dla")
    heads = univ[univ.feature.str.contains(r"_H\d+__pre$", regex=True)].copy()
    if len(heads):
        heads["stat"] = heads.feature.str.replace(r"_L\d+_H\d+__pre$", "", regex=True)
        heads["L"] = heads.feature.str.extract(r"_L(\d+)_H")[0].astype(int)
        heads["H"] = heads.feature.str.extract(r"_H(\d+)__")[0].astype(int)
        for stat in HEAD_STATS_MAIN:
            grid = heads[heads.stat == stat].pivot_table(index="L", columns="H", values="auroc")
            if grid.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 7))
            image = ax.imshow(grid.to_numpy(), aspect="auto", cmap="viridis")
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_title(f"{stat}: AUROC (target A, pool 1, sign-flipped, pre)")
            fig.colorbar(image)
            save(fig, f"heatmap_heads_{stat}")


def line_figures(probes: pd.DataFrame, delay_curve: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if len(probes):
        fig, ax = plt.subplots(figsize=(7, 4))
        for (target, delay), chunk in probes.groupby(["target", "delay"]):
            ax.errorbar(chunk.layer, chunk.auroc, yerr=[chunk.auroc - chunk.ci_low, chunk.ci_high - chunk.auroc], label=f"{target} {delay}", capsize=2)
        ax.set_xlabel("layer")
        ax.set_ylabel("AUROC")
        ax.legend()
        ax.set_title("Hidden-state probes by layer")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figures / f"probes_by_layer.{ext}", dpi=150)
        plt.close(fig)
    if len(delay_curve):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, target in zip(axes, ("A", "B")):
            chunk = delay_curve[delay_curve.target == target]
            for tier, line in chunk.groupby("tier"):
                line = line.set_index("delay").loc[[d for d in DELAYS if d in set(line.delay)]]
                ax.errorbar(range(len(line)), line.value, yerr=[line.value - line.ci_low, line.ci_high - line.value], label=tier, capsize=2)
                ax.set_xticks(range(len(line)), line.index)
            ax.set_title(f"Delay curve, target {target} ({'AUROC' if target == 'A' else 'Spearman'})")
            ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figures / f"delay_curve.{ext}", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip", nargs="*", default=[], choices=["heads", "probes", "delay", "trajectories"])
    args = parser.parse_args(argv)
    run(args.dataset, args.features, args.out, n_boot=args.n_boot, seed=args.seed, n_jobs=args.n_jobs, skip=set(args.skip))


def run(dataset: Path, features: Path, out: Path, *, n_boot: int = 1000, seed: int = 0, n_jobs: int | None = -1, skip: set[str] = frozenset()) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame, traces, negatives = load_inputs(dataset, features)
    manifest = json.loads((features / "manifest.json").read_text())
    extract_meta = json.loads((features / "extract_meta.json").read_text())
    meta = feature_columns(frame, manifest)
    manifest_by_column = {item["column"]: item for item in manifest}
    n_layers = max([int(info["layer"]) for info in meta.values() if info.get("layer")] or [0])
    boot = Bootstrap.draw(frame["problem_id"].astype(str).to_numpy(), n_boot, seed)

    head_matrix = None if "heads" in skip else load_head_matrix(features, frame)
    scalar = {column: frame[column].to_numpy() for column in meta}
    scalar.update({name: frame[name].to_numpy() for name in ("neg_logp_chosen__pre", "random")})
    if head_matrix is not None:
        scalar.update({name: head_matrix[0][:, index] for index, name in enumerate(head_matrix[1])})
    univ_a, univ_b = univariate_screen(frame, scalar, manifest_by_column, boot, n_jobs)
    if head_matrix is not None:
        # Top-ranked per-head features become frame columns so the trajectories can plot them.
        position = {name: index for index, name in enumerate(head_matrix[1])}
        for name in [*univ_a.feature.head(20), *univ_b.feature.head(20)]:
            if name in position and name not in frame:
                frame[name] = head_matrix[0][:, position[name]].astype(np.float32)
    univ_a.to_csv(out / "univariate_A.csv", index=False)
    univ_b.to_csv(out / "univariate_B.csv", index=False)
    heatmaps(univ_a, frame, meta, out)
    print("univariate screen done", flush=True)

    ent_median = float(np.nanmedian(frame["ent__pre"]))
    baseline_rows = []
    for name in BASELINES:
        score = frame[name].to_numpy(dtype=float)
        for target in ("A", "C", "D"):
            baseline_rows += evaluate_binary(score, frame, target, pools_for(frame, target, negatives), boot, ent_median,
                                             {"model": f"baseline:{name}", "delay": delay_of(name) if "__" in name else "pre"})
        baseline_rows += evaluate_graded(score, frame, boot, {"model": f"baseline:{name}", "delay": "pre"})
    tier_table, oof = run_tier_models(frame, meta, negatives, features, boot, n_layers, seed, n_jobs, head_matrix)
    tier_table = pd.concat([pd.DataFrame(baseline_rows), tier_table], ignore_index=True)
    tier_table.to_csv(out / "tier_models.csv", index=False)
    gains = tier_gains(oof, frame, boot)
    gains.to_csv(out / "tier_gains.csv", index=False)
    run_breakdowns(frame, oof, tier_table, boot).to_csv(out / "breakdowns.csv", index=False)

    probes = pd.DataFrame() if "probes" in skip else run_probes(frame, features, extract_meta["hidden_layers"], boot, seed, n_jobs)
    probes.to_csv(out / "probes_by_layer.csv", index=False)
    delay_curve = pd.DataFrame() if "delay" in skip else run_delay_curve(frame, meta, boot, n_layers, seed, n_jobs, head_matrix)
    delay_curve.to_csv(out / "delay_curve.csv", index=False)
    line_figures(probes, delay_curve, out)

    if "trajectories" not in skip:
        from .trajectories import run_trajectories
        run_trajectories(frame, traces, univ_a, features, extract_meta["hidden_layers"], out, seed=seed, n_layers=n_layers)

    (out / "meta.json").write_text(json.dumps({
        "seed": seed, "n_boot": n_boot, "n_problems": int(frame["problem_id"].nunique()),
        "model": extract_meta.get("model"), "model_revision": extract_meta.get("model_revision"),
        "versions": {name: _version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "matplotlib")},
    }, indent=1))
    from .report import write_report
    write_report(out, frame, extract_meta)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()

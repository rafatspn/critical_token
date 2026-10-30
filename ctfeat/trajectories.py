"""Event-aligned trajectories (Section 4.3.7). Labels only align traces; they are never features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import benjamini_hochberg

WINDOW = 64
FIXED = ["ent__pre", "pmax__pre", "logp_chosen__pre", "ll_kl_final_L8__pre", "ll_kl_final_L16__pre", "ll_kl_final_L24__pre",
         "commit_layer__pre", "phi_tok_ref__pre", "phi_win_64__pre", "phi_win_64_delta16__pre"]
BAND_BOOT = 200


def _series(frame: pd.DataFrame, column: str) -> dict[str, np.ndarray]:
    return {trace_id: group.sort_values("pos")[column].to_numpy(dtype=float) for trace_id, group in frame.groupby("trace_id")}


def _events(frame: pd.DataFrame, traces: list[dict[str, Any]], seed: int) -> dict[str, list[tuple[str, int, str]]]:
    """(trace_id, position, problem_id) per event type, with a separately position-matched
    right-trace control for each critical-event type."""
    lengths = frame.groupby("trace_id")["pos"].max().to_dict()
    info = {t["trace_id"]: t for t in traces}
    wrong = [t for t in traces if not t["correct"] and t["trace_id"] in lengths]
    right = [t for t in traces if t["correct"] and t["trace_id"] in lengths]
    rng = np.random.default_rng(seed)
    events: dict[str, list[tuple[str, int, str]]] = {
        "crit_drop": [], "crit_cdpo": [], "control_crit_drop": [], "control_crit_cdpo": [],
    }
    for trace in wrong:
        for kind in ("crit_drop", "crit_cdpo"):
            if trace.get(kind) is None:
                continue
            r = int(trace[kind])
            events[kind].append((trace["trace_id"], r, str(trace["problem_id"])))
            candidates = [t for t in right if lengths[t["trace_id"]] >= max(0, r - 16)]
            if candidates:
                pick = candidates[rng.integers(len(candidates))]
                hi = min(r + 16, lengths[pick["trace_id"]])
                lo = min(max(0, r - 16), hi)
                events[f"control_{kind}"].append(
                    (pick["trace_id"], int(rng.integers(lo, hi + 1)), str(info[pick["trace_id"]]["problem_id"]))
                )
    return events


def _aligned(series: dict[str, np.ndarray], events: list[tuple[str, int, str]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-WINDOW, WINDOW + 1)
    out = np.full((len(events), len(offsets)), np.nan)
    for row, (trace_id, pos, _) in enumerate(events):
        values = series[trace_id]
        idx = pos + offsets
        ok = (idx >= 0) & (idx < len(values))
        out[row, ok] = values[idx[ok]]
    return out, np.array([problem for _, _, problem in events])


def _band(matrix: np.ndarray, problems: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(problems)
    per_problem = np.stack([np.nanmean(matrix[problems == p], 0) for p in unique])
    rng = np.random.default_rng(seed)
    boots = np.stack([np.nanmean(per_problem[rng.integers(0, len(unique), len(unique))], 0) for _ in range(BAND_BOOT)])
    return np.nanmean(matrix, 0), np.nanpercentile(boots, 2.5, 0), np.nanpercentile(boots, 97.5, 0)


def run_trajectories(frame: pd.DataFrame, traces: list[dict[str, Any]], univariate: pd.DataFrame, features: Path,
                     hidden_layers: list[int], out: Path, *, seed: int = 0, n_layers: int | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures" / "aligned"
    figures.mkdir(parents=True, exist_ok=True)
    # Every feature column (<name>__<delay>) gets a segment-means row; the top features and the fixed list are plotted.
    feature_columns = [c for c in frame.columns if "__" in c and pd.api.types.is_numeric_dtype(frame[c])]
    frame = _finite_layers(frame, feature_columns, n_layers)
    top = [f for f in univariate.feature.tolist() if f in frame.columns][:20]
    columns = list(dict.fromkeys([*top, *[c for c in FIXED if c in frame.columns], "v_with"]))
    events = _events(frame, traces, seed)
    offsets = np.arange(-WINDOW, WINDOW + 1)

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figures / f"{name}.{ext}", dpi=120)
        plt.close(fig)

    for column in columns:
        series = _series(frame, column)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
        for ax, kind in zip(axes, ("crit_drop", "crit_cdpo")):
            for label, ev, color in ((kind, events[kind], "C3"),
                                     ("control (right traces)", events[f"control_{kind}"], "C0")):
                if not ev:
                    continue
                mean, low, high = _band(*_aligned(series, ev), seed)
                ax.plot(offsets, mean, color=color, label=label)
                ax.fill_between(offsets, low, high, color=color, alpha=0.2)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel(f"tokens from {kind}")
            ax.set_title(column)
            ax.legend(fontsize=7)
        save(fig, f"aligned_{column}")

    _between_label_stretch(frame, traces, columns, events, figures, save, plt, seed)
    _segments(frame, traces, list(dict.fromkeys([*feature_columns, "v_with"]))).to_csv(out / "segments.csv", index=False)
    _hidden_drift(frame, traces, features, hidden_layers, figures, save, plt)


def _finite_layers(frame: pd.DataFrame, columns: list[str], n_layers: int | None) -> pd.DataFrame:
    """+inf ('never', e.g. chosen_commit_layer) becomes n_layers + 1, as in the models, so means stay finite."""
    infinite = [c for c in columns if np.isposinf(frame[c].to_numpy(dtype=float)).any()]
    if not infinite:
        return frame
    if n_layers is None:
        raise ValueError(f"columns {infinite[:3]} hold +inf; pass n_layers")
    return frame.assign(**{c: frame[c].replace(np.inf, n_layers + 1) for c in infinite})


def _between_label_stretch(frame, traces, columns, events, figures, save, plt, seed) -> None:
    """Segment crit_drop → crit_cdpo rescaled to 20 bins, with 32 tokens either side; right-trace control at a matched r."""
    info = [t for t in traces if not t["correct"] and t.get("crit_drop") is not None and t.get("crit_cdpo") is not None
            and t["crit_drop"] < t["crit_cdpo"]]
    right = {t["trace_id"]: t for t in traces if t["correct"]}
    lengths = frame.groupby("trace_id")["pos"].max().to_dict()
    controls_for = _stretch_control_matches(info, right, lengths, seed)
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows), squeeze=False)
    x_axis = np.concatenate([np.arange(-32, 0) / 32 - 0.0, np.linspace(0, 1, 20), 1 + np.arange(1, 33) / 32])
    for ax, column in zip(axes.flat, columns):
        series = _series(frame, column)
        curves, controls = [], []
        for trace in info:
            values = series.get(trace["trace_id"])
            if values is None:
                continue
            curves.append(_stretch(values, trace["crit_drop"], trace["crit_cdpo"]))
            match = controls_for.get(trace["trace_id"])
            if match is not None and match[0] in series:
                pick, start, end = match
                controls.append(_stretch(series[pick], start, end))
        for data, label, color in ((curves, "wrong: drop → cDPO", "C3"), (controls, "right: matched r", "C0")):
            if data:
                ax.plot(x_axis, np.nanmean(np.stack(data), 0), color=color, label=label)
        ax.axvline(0, color="k", lw=0.5)
        ax.axvline(1, color="k", lw=0.5)
        ax.set_title(column, fontsize=8)
    axes.flat[0].legend(fontsize=7)
    save(fig, "between_label_stretch")


def _stretch_control_matches(info: list[dict[str, Any]], right: dict[str, dict[str, Any]],
                             lengths: dict[str, int], seed: int) -> dict[str, tuple[str, int, int]]:
    """One complete same-length right-trace segment per wrong trace, reused for every feature."""
    rng = np.random.default_rng(seed)
    controls: dict[str, tuple[str, int, int]] = {}
    for trace in info:
        start = int(trace["crit_drop"])
        end = start + int(trace["crit_cdpo"] - trace["crit_drop"])
        fitting = [tid for tid in right if lengths.get(tid, -1) >= end + 32]
        if fitting:
            controls[trace["trace_id"]] = (fitting[rng.integers(len(fitting))], start, end)
    return controls


def _stretch(values: np.ndarray, start: int, end: int) -> np.ndarray:
    before = np.full(32, np.nan)
    idx = np.arange(start - 32, start)
    ok = (idx >= 0) & (idx < len(values))
    before[ok] = values[idx[ok]]
    middle = np.full(20, np.nan)
    if end > start:
        for b in range(20):
            lo, hi = start + int(b * (end - start) / 20), start + int((b + 1) * (end - start) / 20)
            chunk = values[lo: max(hi, lo + 1)] if lo < len(values) else []
            middle[b] = np.nanmean(chunk) if len(chunk) else np.nan
    after = np.full(32, np.nan)
    idx = np.arange(end + 1, end + 33)
    ok = idx < len(values)
    after[ok] = values[idx[ok]]
    return np.concatenate([before, middle, after])


def _segments(frame: pd.DataFrame, traces: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Per-feature segment means (before drop / between / after cDPO; event tokens excluded): the mean
    within each trace, then over the traces of each problem, compared pairwise across problems."""
    from scipy.stats import wilcoxon

    info = {t["trace_id"]: t for t in traces if not t["correct"] and t.get("crit_drop") is not None
            and t.get("crit_cdpo") is not None and t["crit_drop"] < t["crit_cdpo"]}
    sub = frame[frame.trace_id.isin(info)]
    pos = sub["pos"].to_numpy()
    drop = sub["trace_id"].map(lambda t: info[t]["crit_drop"]).to_numpy()
    cdpo = sub["trace_id"].map(lambda t: info[t]["crit_cdpo"]).to_numpy()
    segment = np.select([pos < drop, (pos > drop) & (pos < cdpo), pos > cdpo], ["before", "between", "after"], default="")
    keep = segment != ""
    values = sub.loc[keep, columns].astype(float)
    values["trace_id"], values["segment"] = sub["trace_id"].to_numpy()[keep], segment[keep]
    per_trace = values.groupby(["trace_id", "segment"]).mean().reset_index()
    per_trace["problem_id"] = per_trace["trace_id"].map(lambda t: info[t]["problem_id"])
    per_problem = per_trace.drop(columns="trace_id").groupby(["problem_id", "segment"]).mean()
    rows = []
    for column in columns:
        table = per_problem[column].unstack("segment")
        for a, b in (("before", "between"), ("between", "after"), ("before", "after")):
            if a not in table or b not in table:
                continue
            diff = (table[b] - table[a]).dropna()
            if len(diff) < 5 or np.allclose(diff, 0):
                continue
            p = float(wilcoxon(diff).pvalue)
            sd = diff.std(ddof=1)
            rows.append({"feature": column, "seg_from": a, "seg_to": b, "mean_from": float(table[a].mean()), "mean_to": float(table[b].mean()),
                         "mean_diff": float(diff.mean()), "effect_size_dz": float(diff.mean() / sd) if sd > 0 else np.nan,
                         "n_problems": int(len(diff)), "p": p})
    table = pd.DataFrame(rows)
    if len(table):
        table["q"], table["significant"] = benjamini_hochberg(table["p"].to_numpy())
    return table


def _hidden_drift(frame, traces, features, hidden_layers, figures, save, plt) -> None:
    """cos(h_post[r], h_post[crit_drop]) over r = drop − 64 .. drop + 64, for layers 8/16/24 (when saved)."""
    layers = [layer for layer in (8, 16, 24) if layer in hidden_layers]
    if not layers:
        return
    offsets = np.arange(-WINDOW, WINDOW + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for layer in layers:
        index = hidden_layers.index(layer)
        curves = []
        for trace in traces:
            if trace["correct"] or trace.get("crit_drop") is None:
                continue
            path = features / "hidden" / f"{trace['trace_id'].replace(':', '_').replace('/', '_')}.npy"
            if not path.exists():
                continue
            h = np.load(path, mmap_mode="r")[index].astype(np.float32)[1:]        # h_post for r = 0 .. L-1
            r = int(trace["crit_drop"])
            ref = h[r]
            curve = np.full(len(offsets), np.nan)
            idx = r + offsets
            ok = (idx >= 0) & (idx < len(h))
            rows = h[idx[ok]]
            curve[ok] = rows @ ref / (np.linalg.norm(rows, axis=1) * np.linalg.norm(ref) + 1e-12)
            curves.append(curve)
        if curves:
            ax.plot(offsets, np.nanmean(np.stack(curves), 0), label=f"layer {layer}")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("tokens from crit_drop")
    ax.set_ylabel("cos(h_post, h_post at drop)")
    ax.legend()
    save(fig, "hidden_drift")

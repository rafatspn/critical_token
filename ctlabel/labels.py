from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, read_parquet, write_jsonl, write_parquet


def wilson_interval(n_correct: int, n_total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n_total <= 0:
        return float("nan"), float("nan")
    p = n_correct / n_total
    denominator = 1 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def difference_interval(a_correct: int, a_total: int, b_correct: int, b_total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Normal 95% interval for p(a) - p(b)."""
    if a_total <= 0 or b_total <= 0:
        return float("nan"), float("nan")
    a, b = a_correct / a_total, b_correct / b_total
    difference = a - b
    error = z * math.sqrt(a * (1 - a) / a_total + b * (1 - b) / b_total)
    return difference - error, difference + error


def _count(row: dict[str, Any] | None, prefix: str) -> tuple[int, int] | None:
    if row is None or row.get(f"{prefix}_n_total") in (None, 0):
        return None
    return int(row[f"{prefix}_n_correct"]), int(row[f"{prefix}_n_total"])


def _ratio(count: tuple[int, int] | None) -> float | None:
    return None if count is None else count[0] / count[1]


def label_position(
    *, correct_trace: bool, position: int, crit_cdpo: int | None, crit_good_candidate: int | None,
    with_count: tuple[int, int] | None, before_count: tuple[int, int] | None,
    sampled_count: tuple[int, int] | None, alt2_count: tuple[int, int] | None, alt3_count: tuple[int, int] | None,
    tau_crit: float, zero_frac: float,
) -> dict[str, Any]:
    with_value, before_value = _ratio(with_count), _ratio(before_count)
    drop = None if with_value is None or before_value is None else before_value - with_value
    drop_ci = (float("nan"), float("nan")) if with_count is None or before_count is None else difference_interval(before_count[0], before_count[1], with_count[0], with_count[1])
    is_crit_drop = bool(drop is not None and drop >= tau_crit and drop_ci[0] > 0)
    improvements = []
    for alternative in (sampled_count, alt2_count, alt3_count):
        if alternative is not None and with_count is not None:
            effect = _ratio(alternative) - with_value
            interval = difference_interval(alternative[0], alternative[1], with_count[0], with_count[1])
            improvements.append((effect, interval))
    is_fixable = any(effect >= tau_crit and interval[0] > 0 for effect, interval in improvements)
    best_improvement = max(improvements, key=lambda item: item[0]) if improvements else (None, (None, None))
    a2, a3 = _ratio(alt2_count), _ratio(alt3_count)
    cft_good = with_value is not None and with_value >= 0.7 and a2 is not None and a3 is not None and a2 <= 0.3 and a3 <= 0.3
    rise_good = crit_good_candidate == position and drop is not None and -drop >= tau_crit
    result = {
        "v_with": with_value, "v_before": before_value, "drop": drop,
        "with_ci_low": wilson_interval(*with_count)[0] if with_count else None,
        "with_ci_high": wilson_interval(*with_count)[1] if with_count else None,
        "before_ci_low": wilson_interval(*before_count)[0] if before_count else None,
        "before_ci_high": wilson_interval(*before_count)[1] if before_count else None,
        "drop_ci_low": drop_ci[0] if drop is not None else None,
        "drop_ci_high": drop_ci[1] if drop is not None else None,
        "fixable_effect": best_improvement[0],
        "fixable_ci_low": best_improvement[1][0], "fixable_ci_high": best_improvement[1][1],
        "is_crit_cdpo": bool(not correct_trace and crit_cdpo == position),
        "is_crit_drop": bool(not correct_trace and is_crit_drop),
        "is_fixable": bool(is_fixable),
        "is_crit_good": bool(correct_trace and (cft_good or rise_good)),
        "is_consequence": bool(not correct_trace and crit_cdpo is not None and position > crit_cdpo and with_value is not None and with_value <= zero_frac),
    }
    for name, count in (("without_sampled", sampled_count), ("alt2", alt2_count), ("alt3", alt3_count)):
        interval = wilson_interval(*count) if count else (None, None)
        result[f"v_{name}"] = _ratio(count)
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = interval
    return result


def assign_entropy_deciles(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: (float(rows[index]["entropy_T1"]), rows[index]["trace_id"], rows[index]["pos"]))
    for rank, index in enumerate(ordered):
        rows[index]["entropy_decile"] = min(9, rank * 10 // max(1, len(rows)))


def finalize_position_labels(row: dict[str, Any], *, before_critical: bool, tau_null: float) -> dict[str, Any]:
    """Add mutually usable null/ambiguous labels once trace context is known."""
    interval_inside = row["drop_ci_low"] is not None and row["drop_ci_low"] >= -tau_null and row["drop_ci_high"] <= tau_null
    row["is_noncritical"] = bool(interval_inside and before_critical and row["v_with"] is not None and row["v_with"] > row["zero_frac"])
    row["is_ambiguous"] = not any(row[key] for key in ("is_crit_cdpo", "is_crit_drop", "is_fixable", "is_crit_good", "is_noncritical", "is_consequence"))
    return row


def sample_negatives(positions: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """Draw matched negatives for every critical position.

    Uses only ``positions.parquet`` columns (no rollouts), so later analysis can
    redraw negatives with other matching rules.
    """
    critical_kinds = ["is_crit_cdpo", "is_crit_drop", "is_fixable", "is_crit_good"]
    critical_keys = {(row["trace_id"], int(row["pos"])) for row in positions if any(row[kind] for kind in critical_kinds)}
    noncritical = [row for row in positions if row["is_noncritical"] and (row["trace_id"], int(row["pos"])) not in critical_keys]
    consequences = [row for row in positions if row["is_consequence"] and (row["trace_id"], int(row["pos"])) not in critical_keys]
    output = []
    for critical in positions:
        for kind in critical_kinds:
            if not critical[kind]:
                continue
            groups = {
                "benign_fork": [row for row in noncritical if row["model"] == critical["model"] and row["entropy_decile"] == critical["entropy_decile"]],
                "same_type": [row for row in noncritical if row["model"] == critical["model"] and row["token_type"] == critical["token_type"]],
                "nearby": [row for row in noncritical if row["trace_id"] == critical["trace_id"] and abs(int(row["pos"]) - int(critical["pos"])) <= 64],
                "consequence": [row for row in consequences if row["trace_id"] == critical["trace_id"]],
                "correct_trace": [row for row in noncritical if row["model"] == critical["model"] and row["correct_trace"] and row["token_type"] == critical["token_type"] and row["entropy_decile"] == critical["entropy_decile"]],
                "random": [row for row in noncritical if row["model"] == critical["model"]],
            }
            for group, candidates in groups.items():
                rng = random.Random(f"{seed}:{critical['trace_id']}:{critical['pos']}:{kind}:{group}")
                for negative in rng.sample(candidates, min(k, len(candidates))):
                    output.append({
                        "crit_trace_id": critical["trace_id"], "crit_pos": int(critical["pos"]),
                        "crit_label_kind": kind, "group": group, "neg_trace_id": negative["trace_id"],
                        "neg_pos": int(negative["pos"]),
                    })
    return output


def negatives_from_positions(positions_path: str | Path, k: int = 3, seed: int = 33) -> list[dict[str, Any]]:
    """Standalone entry point: redraw negatives from a saved ``positions.parquet``."""
    return sample_negatives(read_parquet(positions_path), k, seed)


def run_label(config: dict[str, Any]) -> None:
    out, dataset_out = Path(config["out_dir"]), Path(config["out_dir"]) / "dataset"
    traces_list = list(read_jsonl(out / "traces.jsonl"))
    traces = {row["trace_id"]: row for row in traces_list}
    stats = read_parquet(out / "token_stats.parquet")
    value_rows = read_parquet(out / "values.parquet")
    values = {(row["trace_id"], int(row["pos"])): (int(row["n_correct"]), int(row["n_total"])) for row in value_rows}
    swaps = {(row["trace_id"], int(row["pos"])): row for row in read_parquet(out / "swap.parquet")}
    positions = []
    for stat in stats:
        trace, position = traces[stat["trace_id"]], int(stat["pos"])
        swap = swaps.get((stat["trace_id"], position))
        with_count = _count(swap, "with") or values.get((stat["trace_id"], position))
        before_count = _count(swap, "before") or values.get((stat["trace_id"], position - 1))
        labels = label_position(
            correct_trace=bool(trace["correct"]), position=position, crit_cdpo=trace.get("crit_cdpo"),
            crit_good_candidate=trace.get("crit_good_candidate"), with_count=with_count, before_count=before_count,
            sampled_count=_count(swap, "without_sampled"), alt2_count=_count(swap, "alt2"), alt3_count=_count(swap, "alt3"),
            tau_crit=float(config["tau_crit"]), zero_frac=float(config["zero_frac"]),
        )
        row = dict(stat)
        row.update(swap or {})
        row.update(labels)
        if with_count is not None:
            row.setdefault("with_n_correct", with_count[0])
            row.setdefault("with_n_total", with_count[1])
        if before_count is not None:
            row.setdefault("before_n_correct", before_count[0])
            row.setdefault("before_n_total", before_count[1])
        row["correct_trace"] = bool(trace["correct"])
        row["model"] = trace["model"]
        row["zero_frac"] = float(config["zero_frac"])
        row.setdefault("incl_prob", None)
        row.setdefault("stratum", None)
        row.setdefault("swap_done", False)
        positions.append(row)
    critical_by_trace: dict[str, list[int]] = defaultdict(list)
    for row in positions:
        if row["is_crit_cdpo"] or row["is_crit_drop"] or row["is_fixable"] or row["is_crit_good"]:
            critical_by_trace[row["trace_id"]].append(int(row["pos"]))
    for row in positions:
        before_critical = not critical_by_trace[row["trace_id"]] or int(row["pos"]) < min(critical_by_trace[row["trace_id"]])
        finalize_position_labels(row, before_critical=before_critical, tau_null=float(config["tau_null"]))
    assign_entropy_deciles(positions)
    negatives = sample_negatives(positions, int(config["k_neg"]), int(config["seed"]))
    dataset_out.mkdir(parents=True, exist_ok=True)
    write_parquet(dataset_out / "positions.parquet", positions)
    from transformers import AutoTokenizer
    from .datasets import load_problems
    from .tracefields import ensure_trace_fields
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=config["trust_remote_code"])
    questions = {str(row["id"]): row["question"] for row in load_problems(config["dataset"], config.get("input_path"))}
    write_jsonl(dataset_out / "traces.jsonl", [ensure_trace_fields(trace, tokenizer, questions) for trace in traces_list])
    if negatives:
        write_parquet(dataset_out / "negatives.parquet", negatives)
    else:
        # Preserve a loadable schema even when a tiny smoke run has no eligible negative.
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema = pa.schema([
            ("crit_trace_id", pa.string()), ("crit_pos", pa.int32()), ("crit_label_kind", pa.string()),
            ("group", pa.string()), ("neg_trace_id", pa.string()), ("neg_pos", pa.int32()),
        ])
        pq.write_table(pa.Table.from_pylist([], schema=schema), dataset_out / "negatives.parquet")


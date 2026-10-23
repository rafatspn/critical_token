from __future__ import annotations

import importlib.metadata
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, read_parquet, write_json


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_report(config: dict[str, Any]) -> None:
    out, dataset = Path(config["out_dir"]), Path(config["out_dir"]) / "dataset"
    traces = list(read_jsonl(dataset / "traces.jsonl"))
    positions = read_parquet(dataset / "positions.parquet")
    costs = list(read_jsonl(out / "cost.jsonl"))
    select_diagnostics = None
    if (out / "select_meta.json").exists():
        select_diagnostics = json.loads((out / "select_meta.json").read_text(encoding="utf-8"))
    labels = ["is_crit_cdpo", "is_crit_drop", "is_fixable", "is_crit_good", "is_noncritical", "is_consequence", "is_ambiguous"]
    counts = {label: sum(bool(row[label]) for row in positions) for label in labels}
    wrong = [trace for trace in traces if not trace["correct"]]
    critical = [row for row in positions if row["is_crit_cdpo"] or row["is_crit_drop"] or row["is_crit_good"]]
    noncritical = [row for row in positions if row["is_noncritical"]]
    entropy = lambda rows: [float(row["entropy"]) for row in rows]
    crit_entropies, null_entropies = entropy(critical), entropy(noncritical)
    bottom_share = sum(row["entropy_decile"] <= 3 for row in critical) / len(critical) if critical else float("nan")
    lags = [trace["crit_drop"] - trace["crit_cdpo"] for trace in wrong if trace.get("crit_drop") is not None and trace.get("crit_cdpo") is not None]
    def share(predicate: Any, rows: list[Any]) -> float:
        return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else float("nan")
    lines = ["# Critical-token dataset summary", "", "## Labels", ""]
    lines.extend(f"- `{name}`: {value}" for name, value in counts.items())
    lines += [
        "", "## Trace-level checks", "",
        f"- Wrong traces: {len(wrong)}; right traces: {len(traces) - len(wrong)}",
        f"- Wrong traces with `crit_cdpo`: {share(lambda t: t.get('crit_cdpo') is not None, wrong):.1%}",
        f"- Wrong traces with `crit_drop`: {share(lambda t: t.get('crit_drop') is not None, wrong):.1%}",
        f"- Both at the same position: {share(lambda t: t.get('crit_cdpo') is not None and t.get('crit_cdpo') == t.get('crit_drop'), wrong):.1%}",
        f"- Non-monotone wrong traces: {share(lambda t: t.get('non_monotone'), wrong):.1%}",
        f"- Traces whose stop string could not be aligned to tokens (stop tokens kept): {sum(bool(t.get('stop_unaligned')) for t in traces)}",
        f"- Mean lag (`crit_drop - crit_cdpo`): {statistics.fmean(lags) if lags else float('nan'):.2f}",
        "", "## Position-level checks", "",
        f"- Critical entropy mean / median: {statistics.fmean(crit_entropies) if crit_entropies else float('nan'):.4f} / {statistics.median(crit_entropies) if crit_entropies else float('nan'):.4f}",
        f"- Non-critical entropy mean / median: {statistics.fmean(null_entropies) if null_entropies else float('nan'):.4f} / {statistics.median(null_entropies) if null_entropies else float('nan'):.4f}",
        f"- Critical tokens in bottom two entropy quintiles: {bottom_share:.1%}",
        f"- Critical token types: {dict(Counter(row['token_type'] for row in critical))}",
        f"- Fixable at `crit_cdpo`: {share(lambda r: r['is_fixable'], [r for r in positions if r['is_crit_cdpo']]):.1%}",
        f"- Fixable at `crit_drop`: {share(lambda r: r['is_fixable'], [r for r in positions if r['is_crit_drop']]):.1%}",
        "", "## Compute by stage", "",
    ]
    for stage in ("generate", "select", "values", "swap", "label"):
        stage_rows = [row for row in costs if row.get("stage") == stage]
        lines.append(
            f"- `{stage}`: {sum(int(row.get('generated_tokens', 0)) for row in stage_rows):,} generated tokens; "
            f"{sum(float(row.get('gpu_seconds', 0)) for row in stage_rows) / 3600:.3f} GPU-hours"
        )
    lines += [
        f"- Total: {sum(int(row.get('generated_tokens', 0)) for row in costs):,} generated tokens; "
        f"{sum(float(row.get('gpu_seconds', 0)) for row in costs) / 3600:.3f} GPU-hours"
    ]
    if select_diagnostics:
        lines += [
            "", "## HF / vLLM agreement", "",
            f"- Compared positions: {select_diagnostics.get('n_compared', 0)}",
            f"- Argmax agreement: {select_diagnostics.get('argmax_agreement')}",
            f"- Exact top-5 set agreement (diagnostic): {select_diagnostics.get('top5_exact_set_agreement')}",
        ]
    if config["mode"] == "exhaustive":
        sampled_at_cdpo = [row for row in positions if row["is_crit_cdpo"] and row.get("without_sampled_n_total")]
        pass64 = []
        for row in sampled_at_cdpo:
            n, c = int(row["without_sampled_n_total"]), int(row["without_sampled_n_correct"])
            k = min(64, n)
            pass64.append(0.0 if c == 0 else 1 - (math.comb(n - c, k) if k <= n - c else 0) / math.comb(n, k))
        lines += [
            "", "## cDPO replication", "",
            "- Paper target: critical token in about 99–100% of wrong traces; sampled-replacement pass@64 above 90%.",
            f"- This run: critical-token coverage {share(lambda t: t.get('crit_cdpo') is not None, wrong):.1%}; sampled-replacement pass@64 {statistics.fmean(pass64) if pass64 else float('nan'):.1%}.",
        ]
    replayed = [trace for trace in wrong if "crit_cdpo_search_replay" in trace]
    lines += ["", "## Search vs exhaustive", ""]
    if replayed:
        differences = [
            trace["crit_cdpo_search_replay"] - trace["crit_cdpo"]
            for trace in replayed
            if trace.get("crit_cdpo_search_replay") is not None and trace.get("crit_cdpo") is not None
        ]
        agreement = share(lambda trace: trace.get("crit_cdpo_search_replay") == trace.get("crit_cdpo"), replayed)
        lines += [
            f"- Exact `crit_cdpo` agreement: {agreement:.1%} over {len(replayed)} wrong traces.",
            f"- Position-difference distribution (`search - exhaustive`): {dict(sorted(Counter(differences).items()))}",
            f"- Mean evaluated positions per trace: search {statistics.fmean(trace['search_replay_positions'] for trace in replayed):.1f}; exhaustive {statistics.fmean(trace['exhaustive_positions'] for trace in replayed):.1f}.",
            f"- Mean rollout completions per trace: search {statistics.fmean(trace['search_replay_positions'] for trace in replayed) * int(config['n_with_rollouts']):.1f}; exhaustive {statistics.fmean(trace['exhaustive_positions'] for trace in replayed) * int(config['n_with_rollouts']):.1f}.",
        ]
    else:
        lines.append("- Not available in this run. Exhaustive runs automatically replay the search policy on the same observed values.")
    lines.append("")
    (dataset / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    revision = None
    try:
        from transformers import AutoConfig
        revision = getattr(AutoConfig.from_pretrained(config["model"]), "_commit_hash", None)
    except Exception:
        pass
    tokens_by_stage: dict[str, int] = Counter()
    gpu_seconds_by_stage: dict[str, float] = Counter()
    for row in costs:
        tokens_by_stage[row.get("stage", "unknown")] += int(row.get("generated_tokens", 0))
        gpu_seconds_by_stage[row.get("stage", "unknown")] += float(row.get("gpu_seconds", 0))
    write_json(dataset / "meta.json", {
        "config": config, "model_revision": revision,
        "versions": {name: _version(name) for name in ("vllm", "transformers", "torch", "pyarrow")},
        "seeds": {"global": config["seed"]},
        "generated_tokens_by_stage": dict(tokens_by_stage),
        "gpu_hours_by_stage": {stage: seconds / 3600 for stage, seconds in gpu_seconds_by_stage.items()},
        "generated_tokens_total": sum(int(row.get("generated_tokens", 0)) for row in costs),
    })


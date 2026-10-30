from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import append_jsonl, read_jsonl, read_parquet, write_parquet
from .randomness import request_seed
from .sampling import create_engine, exact_prefix, replacement_prefix, run_rollouts
from .types import Count, RolloutRequest


def top_alternatives(top_token_ids: list[int], original_token_id: int, n: int = 2) -> list[int]:
    """Most likely tokens after excluding the committed token."""
    return [int(token_id) for token_id in top_token_ids if int(token_id) != int(original_token_id)][:n]


def stratified_positions(
    stats: list[dict[str, Any]], exclude: set[int], n_strat: int, seed: int, crit_cdpo: int | None, per_cell: int = 2,
) -> dict[int, tuple[float, str]]:
    """Sample positions over (entropy quintile x token-type group), then fill near ``crit_cdpo``.

    Returns ``{pos: (inclusion probability, stratum)}``. Each cell gets up to
    ``per_cell`` positions; when the cells exceed ``n_strat``, quotas shrink from
    the lowest entropy quintile up, so the top quintile keeps its share.
    Remaining slots go to positions within 64 tokens of ``crit_cdpo`` (anywhere
    when there is none).
    """
    rng = random.Random(seed)
    available = sorted((row for row in stats if int(row["pos"]) not in exclude), key=lambda row: int(row["pos"]))
    cells: dict[tuple[int, str], list[int]] = defaultdict(list)
    for row in available:
        cells[(int(row["entropy_quintile"]), str(row["type_group"]))].append(int(row["pos"]))
    quotas = {cell: min(per_cell, len(members)) for cell, members in cells.items()}
    while sum(quotas.values()) > n_strat:
        cell = min((cell for cell, quota in quotas.items() if quota), key=lambda cell: (cell[0], -quotas[cell], cell[1]))
        quotas[cell] -= 1
    result: dict[int, tuple[float, str]] = {}
    cell_probability: dict[int, float] = {}
    for cell in sorted(cells):
        members, quota = cells[cell], quotas[cell]
        for position in members:
            cell_probability[position] = quota / len(members)
        for position in rng.sample(members, quota):
            result[position] = (quota / len(members), f"q{cell[0]}:{cell[1]}")
    remaining = n_strat - len(result)
    unchosen = [int(row["pos"]) for row in available if int(row["pos"]) not in result]
    if crit_cdpo is None:
        pool, stratum = unchosen, "random"
    else:
        pool, stratum = [position for position in unchosen if abs(position - crit_cdpo) <= 64], "nearby"
    take = min(max(0, remaining), len(pool))
    for position in rng.sample(pool, take):
        # Union of the cell draw and the fill draw (treated as independent).
        probability = cell_probability[position] + (1 - cell_probability[position]) * take / len(pool)
        result[position] = (probability, stratum)
    return result


def run_swap(config: dict[str, Any]) -> None:
    out = Path(config["out_dir"])
    traces = list(read_jsonl(out / "traces.jsonl"))
    stats = read_parquet(out / "token_stats.parquet")
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stats:
        by_trace[row["trace_id"]].append(row)
    values = {(row["trace_id"], int(row["pos"])): Count(int(row["n_correct"]), int(row["n_total"])) for row in read_parquet(out / "values.parquet")}
    old_rows = read_parquet(out / "swap.parquet")
    completed = {(row["trace_id"], int(row["pos"])) for row in old_rows}
    plans = []
    value_requests: dict[tuple[str, int], RolloutRequest] = {}
    requests: list[RolloutRequest] = []
    values_only: list[tuple[dict[str, Any], int, float, str]] = []

    def request_values(trace: dict[str, Any], position: int) -> None:
        for value_position in (position, position - 1):
            value_key = (trace["trace_id"], value_position)
            if value_key not in values and value_key not in value_requests:
                value_requests[value_key] = RolloutRequest(
                    trace_id=trace["trace_id"], position=value_position, variant="with",
                    prompt_token_ids=exact_prefix(trace["prompt_ids"], trace["response_ids"], value_position),
                    gold_answers=trace["gold"], dataset=trace["dataset"], n=int(config["n_with_rollouts"]),
                    seed=request_seed(config["seed"], trace["trace_id"], value_position, "with"),
                    response_prefix_ids=list(trace["response_ids"][: value_position + 1]),
                )

    for trace in traces:
        special = {int(trace[key]) for key in ("crit_cdpo", "crit_drop", "crit_good_candidate") if trace.get(key) is not None}
        trace_stats = by_trace[trace["trace_id"]]
        tested = stratified_positions(
            trace_stats, special, int(config["n_strat"]),
            request_seed(config["seed"], trace["trace_id"], -1, "strata"), trace.get("crit_cdpo"),
        )
        tested.update({position: (1.0, "special") for position in special})
        extra = stratified_positions(
            trace_stats, set(tested), int(config["n_values_only"]),
            request_seed(config["seed"], trace["trace_id"], -1, "values_only"), trace.get("crit_cdpo"),
        )
        for position, (inclusion_probability, stratum) in extra.items():
            if (trace["trace_id"], position) not in completed:
                values_only.append((trace, position, inclusion_probability, stratum))
                request_values(trace, position)
        stat_by_pos = {int(row["pos"]): row for row in trace_stats}
        for position, (inclusion_probability, stratum) in tested.items():
            if (trace["trace_id"], position) in completed:
                continue
            stat = stat_by_pos[position]
            sampled_ids = [int(value) for value in stat["replacement_token_ids"]]
            # Rank the vocabulary after excluding the committed token. This is
            # ranks 2/3 when y_i is rank 1, and includes rank 1 otherwise.
            alternatives = top_alternatives(stat["top5_token_ids"], stat["token_id"])
            if len(alternatives) < 2:
                raise ValueError(f"Fewer than two alternatives at {trace['trace_id']}:{position}")
            plans.append((trace, position, inclusion_probability, stratum, sampled_ids, alternatives))
            common = dict(trace_id=trace["trace_id"], position=position, gold_answers=trace["gold"], dataset=trace["dataset"])
            request_values(trace, position)
            for index, token_id in enumerate(sampled_ids):
                variant = f"sampled:{index}"
                requests.append(RolloutRequest(
                    **common, variant=variant,
                    prompt_token_ids=replacement_prefix(trace["prompt_ids"], trace["response_ids"], position, token_id),
                    n=1, seed=request_seed(config["seed"], trace["trace_id"], position, variant),
                    response_prefix_ids=[*trace["response_ids"][:position], token_id],
                ))
            for alt_index, token_id in enumerate(alternatives, start=2):
                variant = f"alt{alt_index}"
                requests.append(RolloutRequest(
                    **common, variant=variant,
                    prompt_token_ids=replacement_prefix(trace["prompt_ids"], trace["response_ids"], position, token_id),
                    n=int(config["n_alt_rollouts"]), seed=request_seed(config["seed"], trace["trace_id"], position, variant),
                    response_prefix_ids=[*trace["response_ids"][:position], token_id],
                ))
    if not plans and not values_only:
        return
    # Exhaustive runs already hold every V_with, so values-only positions may need no rollouts.
    engine = create_engine(config) if value_requests or requests else None
    results, generated_tokens = run_rollouts(
        engine, [*value_requests.values(), *requests], config, cache_path=out / "rollouts_swap.jsonl",
    )
    for trace_id, position in value_requests:
        values[(trace_id, position)] = results[(trace_id, position, "with")]
    new_rows = []
    for trace, position, inclusion_probability, stratum in values_only:
        with_score, before_score = values[(trace["trace_id"], position)], values[(trace["trace_id"], position - 1)]
        new_rows.append({
            "trace_id": trace["trace_id"], "pos": position, "incl_prob": inclusion_probability,
            "stratum": stratum, "swap_done": False,
            "with_n_correct": with_score.n_correct, "with_n_total": with_score.n_total,
            "before_n_correct": before_score.n_correct, "before_n_total": before_score.n_total,
            "rollout_temperature": float(config["rollout_temperature"]),
        })
    for trace, position, inclusion_probability, stratum, sampled_ids, alternatives in plans:
        trace_id = trace["trace_id"]
        with_score = values[(trace_id, position)]
        before_score = values[(trace_id, position - 1)]
        sampled = [results[(trace_id, position, f"sampled:{index}")] for index in range(len(sampled_ids))]
        alt2, alt3 = results[(trace_id, position, "alt2")], results[(trace_id, position, "alt3")]
        counts = Counter(sampled_ids)
        new_rows.append({
            "trace_id": trace_id, "pos": position, "incl_prob": inclusion_probability,
            "stratum": stratum, "swap_done": True,
            "with_n_correct": with_score.n_correct, "with_n_total": with_score.n_total,
            "before_n_correct": before_score.n_correct, "before_n_total": before_score.n_total,
            "without_sampled_n_correct": sum(score.n_correct for score in sampled),
            "without_sampled_n_total": sum(score.n_total for score in sampled),
            "replacement_token_ids": list(counts.keys()), "replacement_token_counts": list(counts.values()),
            "replacement_temperature": float(config["replacement_temperature"]),
            "alt2_token_id": alternatives[0], "alt2_n_correct": alt2.n_correct, "alt2_n_total": alt2.n_total,
            "alt3_token_id": alternatives[1], "alt3_n_correct": alt3.n_correct, "alt3_n_total": alt3.n_total,
            "rollout_temperature": float(config["rollout_temperature"]),
        })
    write_parquet(out / "swap.parquet", old_rows + new_rows)
    append_jsonl(out / "cost.jsonl", [{"stage": "swap", "generated_tokens": generated_tokens}])


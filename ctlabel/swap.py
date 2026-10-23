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


def stratified_positions(stats: list[dict[str, Any]], special: set[int], n_strat: int, seed: int, crit_cdpo: int | None) -> dict[int, float]:
    """Choose entropy/ordinary/nearby strata and return union inclusion probabilities."""
    rng = random.Random(seed)
    available = [row for row in stats if int(row["pos"]) not in special]
    ordered = sorted(available, key=lambda row: (float(row["entropy"]), int(row["pos"])))
    quintiles: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        quintiles[min(4, index * 5 // max(1, len(ordered)))].append(int(row["pos"]))
    allocations = [round(n_strat * 6 / 16), round(n_strat * 6 / 16)]
    allocations.append(max(0, n_strat - sum(allocations)))
    selections: list[tuple[list[int], int]] = []
    selections.append((quintiles[4], allocations[0]))
    # Allocate as evenly as possible across the four lower quintiles.
    ordinary_count = allocations[1]
    for quintile in range(4):
        count = ordinary_count // 4 + (1 if quintile < ordinary_count % 4 else 0)
        selections.append((quintiles[quintile], count))
    if crit_cdpo is None:
        near = [int(row["pos"]) for row in available]
    else:
        near = [int(row["pos"]) for row in available if abs(int(row["pos"]) - crit_cdpo) <= 64]
    selections.append((near, allocations[2]))
    probability_not_selected: dict[int, float] = {int(row["pos"]): 1.0 for row in available}
    selected: set[int] = set()
    for population, count in selections:
        unique = list(dict.fromkeys(population))
        take = min(count, len(unique))
        if not take:
            continue
        chosen = rng.sample(unique, take)
        probability = take / len(unique)
        selected.update(chosen)
        for position in unique:
            probability_not_selected[position] *= 1 - probability
    result = {position: 1 - probability_not_selected[position] for position in selected}
    result.update({position: 1.0 for position in special})
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
    for trace in traces:
        special = {int(trace[key]) for key in ("crit_cdpo", "crit_drop", "crit_good_candidate") if trace.get(key) is not None}
        probabilities = stratified_positions(
            by_trace[trace["trace_id"]], special, int(config["n_strat"]),
            request_seed(config["seed"], trace["trace_id"], -1, "strata"), trace.get("crit_cdpo"),
        )
        stat_by_pos = {int(row["pos"]): row for row in by_trace[trace["trace_id"]]}
        for position, inclusion_probability in probabilities.items():
            if (trace["trace_id"], position) in completed:
                continue
            stat = stat_by_pos[position]
            sampled_ids = [int(value) for value in stat["replacement_token_ids"]]
            # Rank the vocabulary after excluding the committed token. This is
            # ranks 2/3 when y_i is rank 1, and includes rank 1 otherwise.
            alternatives = top_alternatives(stat["top5_token_ids"], stat["token_id"])
            if len(alternatives) < 2:
                raise ValueError(f"Fewer than two alternatives at {trace['trace_id']}:{position}")
            plans.append((trace, position, inclusion_probability, sampled_ids, alternatives))
            common = dict(trace_id=trace["trace_id"], position=position, gold_answers=trace["gold"], dataset=trace["dataset"])
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
    if not plans:
        return
    engine = create_engine(config)
    results, generated_tokens = run_rollouts(
        engine, [*value_requests.values(), *requests], config, cache_path=out / "rollouts_swap.jsonl",
    )
    for trace_id, position in value_requests:
        values[(trace_id, position)] = results[(trace_id, position, "with")]
    new_rows = []
    for trace, position, inclusion_probability, sampled_ids, alternatives in plans:
        trace_id = trace["trace_id"]
        with_score = values[(trace_id, position)]
        before_score = values[(trace_id, position - 1)]
        sampled = [results[(trace_id, position, f"sampled:{index}")] for index in range(len(sampled_ids))]
        alt2, alt3 = results[(trace_id, position, "alt2")], results[(trace_id, position, "alt3")]
        counts = Counter(sampled_ids)
        new_rows.append({
            "trace_id": trace_id, "pos": position, "inclusion_probability": inclusion_probability,
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


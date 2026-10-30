from ctlabel.io import read_parquet, write_jsonl, write_parquet
from ctlabel.swap import run_swap
from ctlabel.types import Count


def test_swap_reuses_canonical_with_values_without_duplicate_rollouts(tmp_path, monkeypatch):
    trace = {
        "trace_id": "t", "correct": False, "dataset": "GSM8K", "model": "m",
        "prompt_ids": [1], "response_ids": [10, 11, 12], "n_labellable_tokens": 3,
        "gold": ["0"], "crit_cdpo": 0, "crit_drop": 1, "crit_good_candidate": None,
    }
    write_jsonl(tmp_path / "traces.jsonl", [trace])
    stats = []
    for position, token_id in enumerate(trace["response_ids"]):
        stats.append({
            "trace_id": "t", "pos": position, "token_id": token_id, "entropy_T1": float(position),
            "entropy_quintile": position, "type_group": "digit_operator",
            "top5_token_ids": [20, 21, token_id, 22, 23], "replacement_token_ids": [30, 31],
        })
    write_parquet(tmp_path / "token_stats.parquet", stats)
    write_parquet(tmp_path / "values.parquet", [{
        "trace_id": "t", "pos": -1, "n_correct": 4, "n_total": 4,
        "rollout_temperature": 0.7, "mode": "search",
    }])
    seen_with = []

    def fake_rollouts(_engine, requests, _config, **_kwargs):
        result = {}
        for request in requests:
            if request.variant == "with":
                seen_with.append((request.trace_id, request.position))
                assert request.response_prefix_ids == trace["response_ids"][: request.position + 1]
            result[(request.trace_id, request.position, request.variant)] = Count(1, request.n)
        return result, 0

    monkeypatch.setattr("ctlabel.swap.create_engine", lambda config: object())
    monkeypatch.setattr("ctlabel.swap.run_rollouts", fake_rollouts)
    config = {
        "out_dir": str(tmp_path), "n_strat": 1, "n_values_only": 1, "seed": 3, "n_with_rollouts": 4,
        "n_alt_rollouts": 2, "rollout_temperature": 0.7, "replacement_temperature": 1.0,
    }
    run_swap(config)
    assert len(seen_with) == len(set(seen_with))
    assert {position for _, position in seen_with} == {0, 1, 2}
    rows = read_parquet(tmp_path / "swap.parquet")
    assert rows
    assert all(row["alt2_token_id"] == 20 and row["alt3_token_id"] == 21 for row in rows)

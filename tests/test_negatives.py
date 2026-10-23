from ctlabel.labels import sample_negatives


def test_negative_sampling_is_deterministic_and_respects_constraints():
    positions = []
    for trace_id, correct in (("w", False), ("r", True)):
        for pos in range(10):
            positions.append({
                "trace_id": trace_id, "pos": pos, "token_type": "digit" if pos % 2 else "other",
                "entropy_decile": pos % 3, "is_noncritical": pos > 0,
                "model": "model",
                "is_consequence": trace_id == "w" and pos > 7,
                "is_crit_cdpo": trace_id == "w" and pos == 0, "is_crit_drop": False,
                "is_fixable": False, "is_crit_good": False,
            })
    traces = {"w": {"correct": False}, "r": {"correct": True}}
    first = sample_negatives(positions, traces, 3, 9)
    second = sample_negatives(positions, traces, 3, 9)
    assert first == second
    assert all((row["neg_trace_id"], row["neg_pos"]) != ("w", 0) for row in first)
    assert all(row["neg_trace_id"] == "w" for row in first if row["group"] in {"nearby", "consequence"})


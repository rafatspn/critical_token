from ctlabel.labels import sample_negatives


def test_negative_sampling_is_deterministic_and_respects_constraints():
    positions = []
    for trace_id, correct in (("w", False), ("r", True)):
        for pos in range(10):
            positions.append({
                "trace_id": trace_id, "pos": pos, "token_type": "digit" if pos % 2 else "other",
                "entropy_decile": pos % 3, "is_noncritical": pos > 0,
                "model": "model", "correct_trace": correct,
                "is_consequence": trace_id == "w" and pos > 7,
                "is_crit_cdpo": trace_id == "w" and pos == 0, "is_crit_drop": False,
                "is_fixable": False, "is_crit_good": False,
            })
    first = sample_negatives(positions, 3, 9)
    second = sample_negatives(positions, 3, 9)
    assert first == second
    assert all((row["neg_trace_id"], row["neg_pos"]) != ("w", 0) for row in first)
    assert all(row["neg_trace_id"] == "w" for row in first if row["group"] in {"nearby", "consequence"})

    assert all(row["neg_trace_id"] == "r" for row in first if row["group"] == "correct_trace")


def test_negatives_redraw_from_saved_positions_only(tmp_path):
    from ctlabel.io import write_parquet
    from ctlabel.labels import negatives_from_positions

    rows = [{
        "trace_id": "w", "pos": pos, "token_type": "digit", "entropy_decile": 0, "model": "m", "correct_trace": False,
        "is_noncritical": pos > 0, "is_consequence": False, "is_crit_cdpo": pos == 0,
        "is_crit_drop": False, "is_fixable": False, "is_crit_good": False,
    } for pos in range(5)]
    write_parquet(tmp_path / "positions.parquet", rows)
    assert negatives_from_positions(tmp_path / "positions.parquet", k=2, seed=1) == sample_negatives(rows, 2, 1)

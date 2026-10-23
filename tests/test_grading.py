import random
import json

import pytest

pytest.importorskip("sympy")

from ctlabel.datasets import load_problems
from ctlabel.grading import grade
from ctlabel.vendor.multi_result_check import multi_result_check


@pytest.mark.parametrize("dataset", ["MATH_test", "GSM8K_test"])
def test_grader_matches_actual_upstream_multi_result_check(dataset, tmp_path):
    rows = load_problems(dataset)
    sample = random.Random(33).sample(rows, min(100, len(rows)))
    reference_input = [
        {"id": str(index), "answer": row["answer"], "generation_result": row["response"]}
        for index, row in enumerate(sample)
    ]
    checked_path = multi_result_check(
        str(tmp_path / f"{dataset}.jsonl"), reference_input,
        dataset.removesuffix("_test"), save_id_with_qid_count=False,
    )
    reference = json.loads(__import__("pathlib").Path(checked_path).read_text())
    for index, row in enumerate(sample):
        _, expected_pred, expected_label = reference[str(index)]["generation_check"][0]
        assert grade(row["response"], row["answer"], dataset) == (expected_pred, expected_label), row["id"]


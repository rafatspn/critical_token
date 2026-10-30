from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from .config import dataset_base
from .vendor.grader import math_equal
from .vendor.parser import extract_answer_by_question_source, extract_answer_str_by_answer_pattern


def grade(pred_str: str, gold_answers: Sequence[Any], dataset: str) -> tuple[Any, bool]:
    """Pure equivalent of cDPO ``multi_result_check`` for one prediction."""
    source = dataset_base(dataset)
    answer_str = extract_answer_str_by_answer_pattern(pred_str=pred_str, answer_pattern="")
    answer_str = answer_str.split("\n\nQuestion:")[0].split("\n\nProblem:")[0]
    pred = extract_answer_by_question_source(pred_str=answer_str, question_source=source)
    if isinstance(pred, str) and source != "MATH":
        pred = pred.lower()
    correct = False
    for gold in gold_answers:
        candidate = gold.lower() if isinstance(gold, str) and source != "MATH" else gold
        if _math_equal(pred, candidate):
            correct = True
    if isinstance(pred, str):
        assert pred == pred.strip()
    return pred, correct



def _math_equal(pred: Any, gold: Any) -> bool:
    """``math_equal`` memoized on (pred, gold).

    The sympy fallback forks a process per call for its timeout, which dominates
    rollout grading; 64 rollouts per prefix repeat the same few answers, so the
    cache returns the identical verdict without re-forking.
    """
    try:
        return _math_equal_cached(pred, gold)
    except TypeError:  # unhashable answer
        return math_equal(pred, gold)


@lru_cache(maxsize=1_000_000, typed=True)
def _math_equal_cached(pred: Any, gold: Any) -> bool:
    return math_equal(pred, gold)

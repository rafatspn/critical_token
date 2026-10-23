import pytest

from ctlabel.search import exhaustive_curve, search_curve
from ctlabel.types import Count


@pytest.mark.parametrize("curve", [
    [64, 64, 60, 0, 0, 0, 0, 0],
    [64, 52, 40, 24, 7, 0, 0, 0],
    [64, 20, 0, 24, 20, 0, 0, 0],
    [64, 0, 0, 10, 10, 0, 0, 0],
])
def test_search_cdpo_matches_exhaustive(curve):
    evaluate = lambda position: Count(64 if position == -1 else curve[position], 64)
    assert search_curve(len(curve), evaluate, stride=2).crit_cdpo == exhaustive_curve(len(curve), evaluate).crit_cdpo


def test_non_monotone_grid_is_flagged():
    curve = [50, 10, 8, 40, 35, 0]
    result = search_curve(len(curve), lambda p: Count(64 if p == -1 else curve[p], 64), stride=1)
    assert result.non_monotone


def test_realistic_length_step_and_gradual_curves_match_exhaustive():
    curves = [
        [64] * 143 + [0] * 113,
        [max(0, 64 - position // 3) for position in range(256)],
    ]
    for curve in curves:
        evaluate = lambda position, values=curve: Count(64 if position == -1 else values[position], 64)
        assert search_curve(len(curve), evaluate, stride=32).crit_cdpo == exhaustive_curve(len(curve), evaluate).crit_cdpo


def test_right_trace_rise_is_not_called_non_monotone():
    curve = [10, 10, 50, 60]
    result = search_curve(len(curve), lambda p: Count(10 if p == -1 else curve[p], 64), stride=1, right_trace=True)
    assert not result.non_monotone


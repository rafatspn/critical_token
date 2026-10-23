from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable, Mapping

from .types import Count

Evaluator = Callable[[int], Count]


@dataclass(frozen=True)
class SearchResult:
    crit_cdpo: int | None
    crit_drop: int | None
    crit_good_candidate: int | None
    non_monotone: bool
    evaluated: Mapping[int, Count]


def analyse_cdpo(values: Mapping[int, Count], zero_frac: float = 3 / 64) -> int | None:
    """Token-index reimplementation of cDPO's ``analyse_all_token`` loop."""
    candidate = None
    last_candidate = None
    for position in sorted(pos for pos in values if pos >= 0):
        score = values[position]
        if score.n_correct == 0 and candidate is None:
            candidate = position
        if candidate is not None and score.value > zero_frac:
            last_candidate = candidate
            candidate = None
    return candidate if candidate is not None else last_candidate


def get_pass_at_k(correct_by_item: Mapping[str, list[bool]], max_k: int) -> dict[int, float]:
    result = {}
    for k in range(1, max_k + 1):
        total = 0.0
        for labels in correct_by_item.values():
            n, c = len(labels), sum(labels)
            k_limit = min(k, n)
            total += 0.0 if c == 0 else 1 - (comb(n - c, k_limit) if k_limit <= n - c else 0) / comb(n, k_limit)
        result[k] = total / len(correct_by_item) if correct_by_item else float("nan")
    return result


def _grid(length: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    return sorted(set(range(0, length, stride)) | {length - 1})


def _binary_zero(left: int, right: int, get: Evaluator) -> int:
    """First observed zero under the locally decreasing assumption."""
    while right - left > 1:
        middle = (left + right) // 2
        if get(middle).n_correct == 0:
            right = middle
        else:
            left = middle
    return right


def _crossing(left: int, right: int, target: float, get: Evaluator, rising: bool = False) -> int:
    while right - left > 1:
        middle = (left + right) // 2
        value = get(middle).value
        if (value >= target) if rising else (value <= target):
            right = middle
        else:
            left = middle
    return right


def search_curve(length: int, evaluate: Evaluator, stride: int = 32, zero_frac: float = 3 / 64, *, right_trace: bool = False) -> SearchResult:
    """Adaptive value search; all calls are memoized and exposed in the result."""
    cache: dict[int, Count] = {}

    def get(position: int) -> Count:
        if position not in cache:
            cache[position] = evaluate(position)
        return cache[position]

    if length <= 0:
        return SearchResult(None, None, None, False, cache)
    grid = _grid(length, stride)
    for position in [-1, *grid]:
        get(position)
    non_monotone = False
    if not right_trace:
        for start, end in zip([-1, *grid[:-1]], grid):
            if get(end).value - get(start).value > 0.25:
                non_monotone = True
                if end - start > 1:
                    get((start + end) // 2)

    crit_cdpo = None
    if not right_trace:
        last_candidate = None
        restart_after = -1
        while True:
            zeros = [p for p in grid if p > restart_after and get(p).n_correct == 0]
            if not zeros:
                crit_cdpo = last_candidate
                break
            zero_point = zeros[0]
            positive = [p for p in [-1, *grid] if restart_after <= p < zero_point and get(p).value > zero_frac]
            left = max(positive) if positive else restart_after
            candidate = _binary_zero(left, zero_point, get)
            checks = sorted(set([p for p in grid if p > candidate] + list(range(candidate + 1, min(length, candidate + 5)))))
            failing = [p for p in checks if get(p).value > zero_frac]
            if not failing:
                crit_cdpo = candidate
                break
            last_candidate = candidate
            restart_after = failing[0]

    intervals = list(zip([-1, *grid[:-1]], grid))
    crit_drop = None
    crit_good = None
    if intervals:
        if right_trace:
            start, end = max(intervals, key=lambda pair: get(pair[1]).value - get(pair[0]).value)
            target = (get(start).value + get(end).value) / 2
            crossing = _crossing(start, end, target, get, rising=True)
            neighborhood = range(max(0, crossing - 3), min(length, crossing + 4))
            crit_good = max(neighborhood, key=lambda p: get(p).value - get(p - 1).value)
        else:
            start, end = max(intervals, key=lambda pair: get(pair[0]).value - get(pair[1]).value)
            target = (get(start).value + get(end).value) / 2
            crossing = _crossing(start, end, target, get)
            neighborhood = range(max(0, crossing - 3), min(length, crossing + 4))
            crit_drop = max(neighborhood, key=lambda p: get(p - 1).value - get(p).value)
    return SearchResult(crit_cdpo, crit_drop, crit_good, non_monotone, dict(cache))


def exhaustive_curve(length: int, evaluate: Evaluator, zero_frac: float = 3 / 64, *, right_trace: bool = False) -> SearchResult:
    values = {-1: evaluate(-1)}
    values.update({position: evaluate(position) for position in range(length)})
    deltas = {position: values[position - 1].value - values[position].value for position in range(length)}
    rises = {position: -delta for position, delta in deltas.items()}
    return SearchResult(
        None if right_trace else analyse_cdpo(values, zero_frac),
        None if right_trace else max(deltas, key=deltas.get, default=None),
        max(rises, key=rises.get, default=None) if right_trace else None,
        False if right_trace else any(values[b].value - values[a].value > 0.25 for a, b in zip(range(-1, length - 1), range(0, length))),
        values,
    )


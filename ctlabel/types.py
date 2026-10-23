from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RolloutRequest:
    trace_id: str
    position: int
    variant: str
    prompt_token_ids: list[int]
    gold_answers: list[Any]
    dataset: str
    n: int
    seed: int
    # Response-only committed tokens. These are decoded together with the
    # generated continuation for grading; prompt/few-shot tokens are excluded.
    response_prefix_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Count:
    n_correct: int
    n_total: int

    @property
    def value(self) -> float:
        return self.n_correct / self.n_total if self.n_total else float("nan")


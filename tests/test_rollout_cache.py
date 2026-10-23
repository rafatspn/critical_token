from dataclasses import dataclass

import pytest

from ctlabel.sampling import run_rollouts
from ctlabel.types import RolloutRequest


@dataclass
class Output:
    token_ids: list[int]
    text: str


@dataclass
class RequestOutput:
    outputs: list[Output]


class Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "The answer is 18." if ids else ""


class Engine:
    """Answers correctly; optionally dies after a number of generate calls."""

    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.submitted = []

    def get_tokenizer(self):
        return Tokenizer()

    def generate(self, prompts, _params, use_tqdm=True):
        if self.fail_after is not None and len(self.submitted) >= self.fail_after:
            raise RuntimeError("simulated wall-time kill")
        self.submitted.append([prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt.prompt_token_ids for prompt in prompts])
        return [RequestOutput([Output([5], "x")]) for _ in prompts]


def test_killed_run_resumes_from_saved_batches(tmp_path, monkeypatch):
    monkeypatch.setattr("ctlabel.sampling._params", lambda *args, **kwargs: None)
    requests = [
        RolloutRequest("t", position, "with", [1, position], ["18"], "GSM8K", 1, 100 + position, response_prefix_ids=[7])
        for position in range(6)
    ]
    config = {"request_batch_size": 2, "rollout_temperature": 0.7}
    cache = tmp_path / "rollouts.jsonl"

    with pytest.raises(RuntimeError):
        run_rollouts(Engine(fail_after=2), requests, config, cache_path=cache)

    resumed = Engine()
    results, _ = run_rollouts(resumed, requests, config, cache_path=cache)
    # The first two batches (positions 0-3) were saved; only positions 4-5 rerun.
    assert resumed.submitted == [[[1, 4], [1, 5]]]
    assert set(results) == {("t", position, "with") for position in range(6)}
    assert all(count.n_correct == 1 and count.n_total == 1 for count in results.values())

    # A different temperature must not reuse cached counts.
    other = Engine()
    run_rollouts(other, requests, {**config, "rollout_temperature": 1.0}, cache_path=cache)
    assert sum(len(batch) for batch in other.submitted) == 6

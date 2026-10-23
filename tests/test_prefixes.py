from dataclasses import dataclass

from ctlabel.sampling import exact_prefix, labellable_token_count, replacement_prefix, run_rollouts
from ctlabel.types import RolloutRequest


@dataclass
class Output:
    token_ids: list[int]
    text: str
    finish_reason: str = "stop"
    stop_reason: object = None


@dataclass
class RequestOutput:
    outputs: list[Output]


class Tokenizer:
    all_special_ids = [99]
    eos_token_id = 99

    def decode(self, ids, skip_special_tokens=True):
        pieces = {11: "The answer is 1", 12: "8.", 90: "\n\nQuestion:", 99: "<eos>"}
        return "".join("" if skip_special_tokens and token == 99 else pieces.get(token, "") for token in ids)


class RecordingEngine:
    def __init__(self):
        self.prompts = None

    def get_tokenizer(self):
        return Tokenizer()

    def generate(self, prompts, _params, use_tqdm=True):
        self.prompts = prompts
        return [RequestOutput([Output([12], "8.")]) for _ in prompts]


def test_production_rollouts_submit_exact_prefixes_and_grade_full_response(monkeypatch):
    monkeypatch.setattr("ctlabel.sampling._params", lambda *args, **kwargs: None)
    engine = RecordingEngine()
    prompt, response, position = [1, 2], [11, 13], 0
    request = RolloutRequest(
        "trace", position, "with", exact_prefix(prompt, response, position), ["18"],
        "GSM8K", 1, 7, response_prefix_ids=response[: position + 1],
    )
    result, _ = run_rollouts(engine, [request], {"request_batch_size": 8, "rollout_temperature": 0.7})
    recorded = [item["prompt_token_ids"] if isinstance(item, dict) else item.prompt_token_ids for item in engine.prompts]
    assert recorded == [prompt + response[: position + 1]]
    assert result[("trace", position, "with")].n_correct == 1


def test_replacement_prefix_is_token_exact():
    assert replacement_prefix([1, 2], [11, 13], 1, 77) == [1, 2, 11, 77]


def test_eos_and_stop_string_tokens_are_not_labellable():
    tokenizer = Tokenizer()
    stopped = Output([11, 12, 90], "The answer is 18.", stop_reason="\n\nQuestion:")
    eos = Output([11, 12, 99], "The answer is 18.", stop_reason=None)
    assert labellable_token_count(tokenizer, stopped) == 2
    assert labellable_token_count(tokenizer, eos) == 2


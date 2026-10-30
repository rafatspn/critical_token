from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .grading import grade
from .io import append_jsonl, read_jsonl
from .prompts import STOP_STRINGS
from .types import Count, RolloutRequest


def exact_prefix(prompt_ids: list[int], response_ids: list[int], position: int) -> list[int]:
    """Return prompt plus response through ``position``; -1 means prompt alone."""
    if position < -1 or position >= len(response_ids):
        raise IndexError(position)
    return list(prompt_ids) + list(response_ids[: position + 1])


def replacement_prefix(prompt_ids: list[int], response_ids: list[int], position: int, token_id: int) -> list[int]:
    if position < 0 or position >= len(response_ids):
        raise IndexError(position)
    return list(prompt_ids) + list(response_ids[:position]) + [int(token_id)]


def tokens_prompt(token_ids: list[int]) -> Any:
    """Construct the current vLLM token-native input type (kept isolated for mocks)."""
    try:
        from vllm.inputs import TokensPrompt
        return TokensPrompt(prompt_token_ids=token_ids)
    except ImportError:
        return {"prompt_token_ids": token_ids}


def create_engine(config: dict[str, Any]) -> Any:
    from vllm import LLM
    return LLM(
        model=config["model"],
        tensor_parallel_size=int(config["tensor_parallel_size"]),
        trust_remote_code=bool(config["trust_remote_code"]),
        enable_prefix_caching=True,
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        logprobs_mode=config.get("vllm_logprobs_mode", "raw_logprobs"),
    )


def _params(config: dict[str, Any], *, n: int, seed: int, temperature: float, logprobs: int | None = None) -> Any:
    from vllm import SamplingParams
    return SamplingParams(
        n=n, temperature=temperature, top_p=float(config["top_p"]),
        max_tokens=int(config["max_tokens"]), stop=STOP_STRINGS, seed=seed, logprobs=logprobs,
    )


def _cache_key(request: RolloutRequest, temperature: float) -> tuple[str, int, str, int, int, float]:
    return (request.trace_id, request.position, request.variant, request.seed, request.n, temperature)


def run_rollouts(
    engine: Any, requests: Iterable[RolloutRequest], config: dict[str, Any], cache_path: str | Path | None = None,
) -> tuple[dict[tuple[str, int, str], Count], int]:
    """Run token-native requests in batched calls and grade by explicit identity.

    With ``cache_path``, each finished batch is appended to that jsonl file and
    requests already recorded there are not rerun, so a stage killed at the
    wall-time limit resumes where it stopped. Seeds are fixed per request, so a
    resumed run gives the same counts as an uninterrupted one.
    """
    reqs = list(requests)
    temperature = float(config["rollout_temperature"])
    result: dict[tuple[str, int, str], Count] = {}
    if cache_path is not None:
        cached = {
            (row["trace_id"], int(row["pos"]), row["variant"], int(row["seed"]), int(row["n"]), float(row["rollout_temperature"])):
                Count(int(row["n_correct"]), int(row["n_total"]))
            for row in read_jsonl(cache_path)
        }
        pending = []
        for req in reqs:
            hit = cached.get(_cache_key(req, temperature))
            if hit is None:
                pending.append(req)
            else:
                result[(req.trace_id, req.position, req.variant)] = hit
        reqs = pending
    if not reqs:
        return result, 0
    generated_tokens = 0
    tokenizer = engine.get_tokenizer()
    batch_size = int(config.get("request_batch_size", 4096))
    for start in range(0, len(reqs), batch_size):
        batch = reqs[start : start + batch_size]
        prompts = [tokens_prompt(r.prompt_token_ids) for r in batch]
        params = [_params(config, n=r.n, seed=r.seed, temperature=temperature) for r in batch]
        outputs = engine.generate(prompts, params, use_tqdm=True)
        for req, request_output in zip(batch, outputs, strict=True):
            labels = []
            for output in request_output.outputs:
                generated_tokens += len(output.token_ids)
                # A rollout completion is only the suffix after the committed
                # trace prefix. Grade the complete response candidate. Decoding
                # the combined IDs also handles tokens that straddle a text
                # concatenation boundary without re-tokenizing anything.
                candidate_ids = [*req.response_prefix_ids, *map(int, output.token_ids)]
                candidate_text = tokenizer.decode(candidate_ids, skip_special_tokens=True)
                labels.append(grade(candidate_text, req.gold_answers, req.dataset)[1])
            result[(req.trace_id, req.position, req.variant)] = Count(sum(labels), len(labels))
        if cache_path is not None:
            append_jsonl(cache_path, [{
                "trace_id": req.trace_id, "pos": req.position, "variant": req.variant, "seed": req.seed,
                "n": req.n, "rollout_temperature": temperature,
                "n_correct": result[(req.trace_id, req.position, req.variant)].n_correct,
                "n_total": result[(req.trace_id, req.position, req.variant)].n_total,
            } for req in batch])
    return result, generated_tokens


def labellable_token_count(tokenizer: Any, completion: Any) -> int:
    """Return the generated-token prefix excluding EOS/stop terminators.

    vLLM omits stop strings and EOS from ``CompletionOutput.text`` but retains
    their token IDs. Prefer exact decoded-text alignment for string stops and
    fall back conservatively for token/EOS stops.
    """
    token_ids = [int(token_id) for token_id in completion.token_ids]
    stop_reason = getattr(completion, "stop_reason", None)
    finish_reason = str(getattr(completion, "finish_reason", ""))
    if isinstance(stop_reason, str):
        target = completion.text
        # A stop suffix is short; inspect only a bounded suffix of token cuts
        # instead of decoding every O(L) prefix and doing O(L^2) work.
        minimum_end = max(0, len(token_ids) - max(32, len(stop_reason) + 4))
        matches = [
            end for end in range(len(token_ids), minimum_end - 1, -1)
            if tokenizer.decode(token_ids[:end], skip_special_tokens=True) == target
        ]
        if matches:
            return max(matches)
        # Tokenizers may normalize spaces during incremental decoding. Locate
        # the stop's character boundary in a full decode and retain only tokens
        # ending before that boundary (not merely all but its final token).
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        stop_start = decoded.rfind(stop_reason)
        if stop_start >= 0:
            candidates = [
                end for end in range(len(token_ids), minimum_end - 1, -1)
                if len(tokenizer.decode(token_ids[:end], skip_special_tokens=True)) <= stop_start
            ]
            if candidates:
                return max(candidates)
    if isinstance(stop_reason, int):
        return max(0, len(token_ids) - 1)
    if finish_reason == "stop" and token_ids:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        eos_ids = getattr(tokenizer, "eos_token_id", None)
        if token_ids[-1] in special_ids or token_ids[-1] == eos_ids:
            return len(token_ids) - 1
    return len(token_ids)


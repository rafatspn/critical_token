from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .datasets import load_problems
from .grading import grade
from .io import append_jsonl, read_jsonl
from .prompts import build_prompt, prompt_spans
from .randomness import request_seed
from .select import select_traces
from .sampling import _params, create_engine, labellable_token_count, tokens_prompt
from .tracefields import model_revision, sampled_logprobs


def _serialize_logprobs(logprobs: Any) -> list[list[dict[str, Any]]] | None:
    if logprobs is None:
        return None
    rows = []
    for position in logprobs:
        choices = []
        for token_id, item in (position or {}).items():
            choices.append({
                "token_id": int(token_id), "logprob": float(item.logprob),
                "rank": int(item.rank) if item.rank is not None else None,
                "decoded_token": getattr(item, "decoded_token", None),
            })
        rows.append(sorted(choices, key=lambda x: (x["rank"] is None, x["rank"] or 10**9)))
    return rows


def _target_met(kept: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    """True once ``select`` can fill both its wrong and right quotas from ``kept``."""
    selected = select_traces(kept, config)
    n_wrong = sum(not row["correct"] for row in selected)
    return n_wrong >= int(config["n_wrong"]) and len(selected) - n_wrong >= int(config["n_right"])


def run_generate(config: dict[str, Any], *, engine: Any = None, tokenizer: Any = None) -> None:
    """Sample and grade traces while preserving the exact token boundary.

    Problems are visited in a seeded random order, ``generate_chunk_problems`` at
    a time; each chunk is written to disk before the next starts, and sampling
    stops once the kept traces can fill ``n_wrong``/``n_right`` (unless
    ``generate_until_target`` is false).
    """
    out = Path(config["out_dir"])
    path = out / "traces_all.jsonl"
    kept = [
        {"trace_id": row["trace_id"], "problem_id": str(row["problem_id"]), "correct": bool(row["correct"])}
        for row in read_jsonl(path)
    ]
    existing = {row["trace_id"] for row in kept}
    completed_problems = {str(row["problem_id"]) for row in read_jsonl(out / "generate_done.jsonl")}
    until_target = bool(config["generate_until_target"])
    if until_target and _target_met(kept, config):
        return
    problems = load_problems(config["dataset"], config.get("input_path"), config.get("n_problems"))
    random.Random(int(config["seed"])).shuffle(problems)
    n = int(config["n_samples_per_problem"])
    remaining = [
        problem for problem in problems
        if str(problem["id"]) not in completed_problems
        and not {f"{problem['id']}:{i}" for i in range(n)} <= existing
    ]
    if not remaining:
        return
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=config["trust_remote_code"])
    chunk_size = int(config["generate_chunk_problems"])
    for start in range(0, len(remaining), chunk_size):
        chunk = remaining[start:start + chunk_size]
        prompts, params, pending = [], [], []
        for problem in chunk:
            prompt_ids = tokenizer.encode(build_prompt(config["dataset"], problem["question"]), add_special_tokens=False)
            for sample_index in range(n):
                trace_id = f"{problem['id']}:{sample_index}"
                if trace_id in existing:
                    continue
                seed = request_seed(config["seed"], trace_id, -1, "generate")
                prompts.append(tokens_prompt(prompt_ids))
                params.append(_params(config, n=1, seed=seed, temperature=float(config["gen_temperature"]), logprobs=int(config["logprobs"])))
                pending.append((problem, prompt_ids, seed, sample_index))
        if engine is None:
            engine = create_engine(config)
        rows, generated = _generate_rows(engine, tokenizer, prompts, params, pending, config)
        append_jsonl(path, rows)
        append_jsonl(out / "generate_done.jsonl", [{"problem_id": str(problem["id"])} for problem in chunk])
        append_jsonl(out / "cost.jsonl", [{"stage": "generate", "generated_tokens": generated}])
        kept.extend({"trace_id": row["trace_id"], "problem_id": row["problem_id"], "correct": row["correct"]} for row in rows)
        done = start + len(chunk)
        print(f"generate: {done}/{len(remaining)} problems sampled, {len(kept)} traces kept")
        if until_target and _target_met(kept, config):
            print(f"generate: selection target reached after {done} problems; stopping")
            return
    if until_target:
        print("warning: all problems sampled but the selection target was not reached; select will take what exists")


def _generate_rows(
    engine: Any, tokenizer: Any, prompts: list[Any], params: list[Any],
    pending: list[tuple[dict[str, Any], list[int], int, int]], config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    outputs = engine.generate(prompts, params, use_tqdm=True)
    generated = 0
    rows = []
    unaligned_stops = 0
    for (problem, prompt_ids, seed, sample_index), request_output in zip(pending, outputs, strict=True):
        completion = request_output.outputs[0]
        trace_id = f"{problem['id']}:{sample_index}"
        response_ids = [int(value) for value in completion.token_ids]
        n_labellable_tokens = labellable_token_count(tokenizer, completion)
        # A string stop always spans at least one token; keeping the full length
        # means the stop could not be aligned and its tokens would be labelled.
        stop_unaligned = isinstance(completion.stop_reason, str) and n_labellable_tokens == len(response_ids)
        generated += len(response_ids)
        finish_reason = str(completion.finish_reason)
        pred, correct = grade(completion.text, problem["answer"], config["dataset"])
        if finish_reason == "length" or len(response_ids) >= int(config["max_tokens"]) or pred == "":
            continue
        top_logprobs = _serialize_logprobs(completion.logprobs)
        rows.append({
            "trace_id": trace_id, "problem_id": str(problem["id"]), "dataset": config["dataset"],
            "model": config["model"], "model_revision": model_revision(config["model"]),
            "prompt_ids": [int(x) for x in prompt_ids],
            "response_ids": response_ids, "response_text": completion.text,
            "n_labellable_tokens": n_labellable_tokens, "stop_unaligned": stop_unaligned,
            "finish_reason": finish_reason, "stop_reason": completion.stop_reason,
            "pred": pred, "gold": problem["answer"],
            "correct": bool(correct), "gen_temperature": float(config["gen_temperature"]),
            "seed": seed, "vllm_top_logprobs": top_logprobs,
            "vllm_logprob_sampled": sampled_logprobs(response_ids, top_logprobs),
            **prompt_spans(tokenizer, config["dataset"], problem["question"], prompt_ids),
        })
        unaligned_stops += stop_unaligned
    if unaligned_stops:
        print(f"warning: {unaligned_stops} kept traces have a stop string that could not be aligned to tokens; see `stop_unaligned`")
    return rows, generated

from __future__ import annotations

from pathlib import Path
from typing import Any

from .datasets import load_problems
from .grading import grade
from .io import append_jsonl, read_jsonl
from .prompts import build_prompt
from .randomness import request_seed
from .sampling import _params, create_engine, labellable_token_count, tokens_prompt


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


def run_generate(config: dict[str, Any]) -> None:
    """Sample and grade traces while preserving the exact token boundary."""
    from transformers import AutoTokenizer

    out = Path(config["out_dir"])
    path = out / "traces_all.jsonl"
    existing = {row["trace_id"] for row in read_jsonl(path)}
    completed_problems = {str(row["problem_id"]) for row in read_jsonl(out / "generate_done.jsonl")}
    problems = load_problems(config["dataset"], config.get("input_path"), config.get("n_problems"))
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=config["trust_remote_code"])
    engine = create_engine(config)
    prompts, params, pending = [], [], []
    n = int(config["n_samples_per_problem"])
    for problem in problems:
        problem_id = str(problem["id"])
        if problem_id in completed_problems:
            continue
        expected = {f"{problem_id}:{i}" for i in range(n)}
        if expected <= existing:
            continue
        prompt_ids = tokenizer.encode(build_prompt(config["dataset"], problem["question"]), add_special_tokens=False)
        for sample_index in range(n):
            trace_id = f"{problem_id}:{sample_index}"
            if trace_id in existing:
                continue
            seed = request_seed(config["seed"], trace_id, -1, "generate")
            prompts.append(tokens_prompt(prompt_ids))
            params.append(_params(config, n=1, seed=seed, temperature=float(config["gen_temperature"]), logprobs=int(config["logprobs"])))
            pending.append((problem, prompt_ids, seed, sample_index))
    if not pending:
        return
    outputs = engine.generate(prompts, params, use_tqdm=True)
    generated = 0
    rows = []
    attempted_problems = set()
    unaligned_stops = 0
    for (problem, prompt_ids, seed, sample_index), request_output in zip(pending, outputs, strict=True):
        attempted_problems.add(str(problem["id"]))
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
        rows.append({
            "trace_id": trace_id, "problem_id": str(problem["id"]), "dataset": config["dataset"],
            "model": config["model"], "prompt_ids": [int(x) for x in prompt_ids],
            "response_ids": response_ids, "response_text": completion.text,
            "n_labellable_tokens": n_labellable_tokens, "stop_unaligned": stop_unaligned,
            "finish_reason": finish_reason, "stop_reason": completion.stop_reason,
            "pred": pred, "gold": problem["answer"],
            "correct": bool(correct), "gen_temperature": float(config["gen_temperature"]),
            "seed": seed, "vllm_top_logprobs": _serialize_logprobs(completion.logprobs),
        })
        unaligned_stops += stop_unaligned
    if unaligned_stops:
        print(f"warning: {unaligned_stops} kept traces have a stop string that could not be aligned to tokens; see `stop_unaligned`")
    append_jsonl(path, rows)
    append_jsonl(out / "generate_done.jsonl", [{"problem_id": problem_id} for problem_id in sorted(attempted_problems)])
    append_jsonl(out / "cost.jsonl", [{"stage": "generate", "generated_tokens": generated}])


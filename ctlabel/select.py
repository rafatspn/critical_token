from __future__ import annotations

import math
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, read_parquet, write_json, write_jsonl, write_parquet
from .randomness import request_seed


CONNECTIVES = {"so", "thus", "therefore", "but", "however", "wait", "then", "since", "because", "if", "or", "instead", "hence"}
LATEX_OPERATORS = {"\\le", "\\ge", "\\times", "\\frac", "\\cdot", "\\neq", "\\approx", "\\div", "\\pm"}


def token_type(text: str) -> str:
    clean = text.strip().lower()
    if any(character.isdigit() for character in clean):
        return "digit"
    if clean in CONNECTIVES:
        return "connective"
    commands = set(re.findall(r"\\[A-Za-z]+", clean))
    if clean in {"+", "-", "*", "/", "=", "<", ">", "^", "(", ")"} or bool(commands & LATEX_OPERATORS):
        return "operator"
    if re.search(r"\\[A-Za-z]+", clean):
        return "latex_other"
    if clean and all(character in string.punctuation for character in clean):
        return "punct"
    return "other"


def select_traces(traces: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_problem[str(trace["problem_id"])].append(trace)
    contested = []
    for problem_id, rows in by_problem.items():
        n_correct = sum(bool(row["correct"]) for row in rows)
        if int(config["min_correct"]) <= n_correct <= int(config["max_correct"]):
            contested.append((problem_id, rows))
    rng = random.Random(int(config["seed"]))
    rng.shuffle(contested)
    selected, wrong_count, right_count = [], 0, 0
    for _, rows in contested:
        wrong = [row for row in rows if not row["correct"]]
        right = [row for row in rows if row["correct"]]
        rng.shuffle(wrong)
        rng.shuffle(right)
        take_wrong = min(len(wrong), int(config["wrong_per_problem"]), int(config["n_wrong"]) - wrong_count)
        take_right = min(len(right), int(config["right_per_problem"]), int(config["n_right"]) - right_count)
        selected.extend(wrong[:take_wrong])
        selected.extend(right[:take_right])
        wrong_count += take_wrong
        right_count += take_right
        if wrong_count >= int(config["n_wrong"]) and right_count >= int(config["n_right"]):
            break
    return selected


def _number_starts(token_texts: list[str]) -> list[bool]:
    result = []
    previous_digit = False
    for text in token_texts:
        has_digit = any(c.isdigit() for c in text)
        continues = previous_digit and bool(text) and not text[0].isspace() and text[0].isdigit()
        result.append(bool(has_digit and not continues))
        previous_digit = has_digit and bool(text) and (text[-1].isdigit() or text[-1] in ".,")
    return result


TYPE_GROUPS = {"digit": "digit_operator", "operator": "digit_operator", "connective": "connective"}
STATS_VERSION = 2


def type_group(kind: str) -> str:
    """Coarse token-type group used for stratified sampling and coverage checks."""
    return TYPE_GROUPS.get(kind, "latex_punct_other")


def step_indices(token_texts: list[str]) -> list[int]:
    """Reasoning-step index per token: newlines in the response before the token."""
    result, newlines = [], 0
    for text in token_texts:
        result.append(newlines)
        newlines += text.count("\n")
    return result


def entropy_quintiles(entropies: list[float]) -> list[int]:
    """Within-trace entropy quintile (0 = lowest) per position; ties broken by position."""
    order = sorted(range(len(entropies)), key=lambda index: (entropies[index], index))
    result = [0] * len(entropies)
    for rank, index in enumerate(order):
        result[index] = min(4, rank * 5 // max(1, len(entropies)))
    return result


def distribution_stats(logits: Any, token_id: int, gen_temperature: float) -> dict[str, Any]:
    """Exact full-vocabulary statistics of one next-token distribution (a 1-D float tensor)."""
    import torch

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum()
    varentropy = (probs * (log_probs + entropy) ** 2).sum()
    gen_log_probs = torch.log_softmax(logits / gen_temperature, dim=-1)
    top_probs, top_ids = probs.topk(5)
    return {
        "entropy_T1": float(entropy), "entropy_gen": float(-(gen_log_probs.exp() * gen_log_probs).sum()),
        "varentropy_T1": float(varentropy),
        "p_max": float(top_probs[0]), "p_y": float(probs[token_id]), "p_y_gen": float(gen_log_probs[token_id].exp()),
        "rank_y": int((logits > logits[token_id]).sum().item()) + 1,
        "margin_top2": float(top_probs[0] - top_probs[1]),
        "top5_token_ids": [int(x) for x in top_ids.tolist()],
        "top5_probabilities": [float(x) for x in top_probs.tolist()],
    }


def engine_check(
    row: dict[str, Any], vllm_top: list[dict[str, Any]] | None, vllm_logprob: float | None,
    *, compare_prob: float, tolerance: float,
) -> dict[str, Any]:
    """Compare HF statistics at one position with vLLM's stored logprobs."""
    result: dict[str, Any] = {"hf_vllm_argmax_agree": None, "hf_vllm_top5_agree": None, "engine_mismatch": None,
                              "vllm_p_sampled": None if vllm_logprob is None else math.exp(vllm_logprob)}
    if not vllm_top:
        return result
    v_ids = [int(item["token_id"]) for item in sorted(vllm_top, key=lambda item: item.get("rank") or 10**9)]
    hf_ids = row["top5_token_ids"]
    result["hf_vllm_argmax_agree"] = v_ids[0] == hf_ids[0]
    result["hf_vllm_top5_agree"] = set(v_ids[:5]) == set(hf_ids)
    prob_gap = None if vllm_logprob is None else abs(result["vllm_p_sampled"] - compare_prob)
    result["engine_mismatch"] = bool((prob_gap is not None and prob_gap > tolerance) or v_ids[:2] != hf_ids[:2])
    return result


def compute_token_stats(
    traces: list[dict[str, Any]], config: dict[str, Any], *,
    existing_rows: list[dict[str, Any]] | None = None, checkpoint_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=config["trust_remote_code"])
    # One GPU holds the 7B model in bf16; placing it directly avoids device_map,
    # which would add an `accelerate` dependency.
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, trust_remote_code=config["trust_remote_code"],
        attn_implementation=config["attn_implementation"],
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    hidden = config.get("save_hidden_states") or None
    hidden_dir = None
    if hidden:
        hidden_dir = Path(config["out_dir"]) / "hidden"
        hidden_dir.mkdir(parents=True, exist_ok=True)
        n_tokens = sum(int(trace.get("n_labellable_tokens", len(trace["response_ids"]))) for trace in traces)
        size = n_tokens * len(hidden["layers"]) * int(model.config.hidden_size) * np.dtype(hidden.get("dtype", "float16")).itemsize
        print(f"save_hidden_states: {len(hidden['layers'])} layers x {n_tokens:,} tokens -> about {size / 1e9:.1f} GB under {hidden_dir}")
    gen_temperature = float(config["gen_temperature"])
    # vLLM's raw logprobs are taken before temperature; processed ones after it.
    compare_temperature = 1.0 if config["vllm_logprobs_mode"].startswith("raw") else gen_temperature
    tolerance = float(config["engine_mismatch_tol"])
    rows = list(existing_rows or [])
    completed = defaultdict(set)
    for row in rows:
        completed[row["trace_id"]].add(int(row["pos"]))
    with torch.inference_mode():
        for trace in traces:
            labellable_length = int(trace.get("n_labellable_tokens", len(trace["response_ids"])))
            hidden_path = hidden_dir / f"{trace['trace_id'].replace(':', '_')}.npy" if hidden_dir else None
            if len(completed[trace["trace_id"]]) >= labellable_length and (hidden_path is None or hidden_path.exists()):
                continue
            prompt_ids, response_ids = trace["prompt_ids"], trace["response_ids"]
            device = next(model.parameters()).device
            output = model(input_ids=torch.tensor([prompt_ids + response_ids], device=device), output_hidden_states=bool(hidden))
            logits = output.logits[0]
            if hidden_path is not None:
                # hidden_states[0] is the embedding output; index l is after decoder layer l.
                layers = [output.hidden_states[layer][0, len(prompt_ids): len(prompt_ids) + labellable_length] for layer in hidden["layers"]]
                np.save(hidden_path, torch.stack(layers).float().cpu().numpy().astype(hidden.get("dtype", "float16")))
            token_texts = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in response_ids]
            number_starts = _number_starts(token_texts)
            steps = step_indices(token_texts)
            vllm_logprobs = trace.get("vllm_top_logprobs") or []
            vllm_sampled = trace.get("vllm_logprob_sampled") or []
            trace_rows = []
            for pos, token_id in enumerate(response_ids[:labellable_length]):
                prediction_index = len(prompt_ids) + pos - 1
                if prediction_index < 0:
                    raise ValueError("The prompt must contain at least one token")
                token_logits = logits[prediction_index].float()
                stats = distribution_stats(token_logits, token_id, gen_temperature)
                replacement_probs = torch.log_softmax(token_logits, dim=-1).exp().cpu()
                replacement_probs[token_id] = 0
                replacement_probs.pow_(1.0 / float(config["replacement_temperature"]))
                generator = torch.Generator().manual_seed(request_seed(config["seed"], trace["trace_id"], pos, "replacement"))
                replacements = torch.multinomial(
                    replacement_probs, int(config["n_sampled_alternatives"]), replacement=True, generator=generator
                ).tolist()
                kind = token_type(token_texts[pos])
                row = {
                    "trace_id": trace["trace_id"], "pos": pos, "token_id": int(token_id),
                    "token_text": token_texts[pos], "token_type": kind, "type_group": type_group(kind),
                    "is_first_subtoken": number_starts[pos], "step_index": steps[pos], **stats,
                    "replacement_token_ids": [int(x) for x in replacements],
                    "replacement_temperature": float(config["replacement_temperature"]),
                }
                compare_prob = stats["p_y"] if compare_temperature == 1.0 else stats["p_y_gen"]
                row.update(engine_check(
                    row, vllm_logprobs[pos] if pos < len(vllm_logprobs) else None,
                    vllm_sampled[pos] if pos < len(vllm_sampled) else None,
                    compare_prob=compare_prob, tolerance=tolerance,
                ))
                trace_rows.append(row)
            for row, quintile in zip(trace_rows, entropy_quintiles([row["entropy_T1"] for row in trace_rows])):
                row["entropy_quintile"] = quintile
            rows = [row for row in rows if row["trace_id"] != trace["trace_id"]] + trace_rows
            if checkpoint_path is not None and rows:
                write_parquet(checkpoint_path, sorted(rows, key=lambda item: (item["trace_id"], item["pos"])))
    def share(key: str) -> float | None:
        checks = [bool(row[key]) for row in rows if row.get(key) is not None]
        return sum(checks) / len(checks) if checks else None
    diagnostics = {
        "n_compared": sum(row.get("hf_vllm_argmax_agree") is not None for row in rows),
        "argmax_agreement": share("hf_vllm_argmax_agree"),
        "top5_exact_set_agreement": share("hf_vllm_top5_agree"),
        "engine_mismatch_share": share("engine_mismatch"),
        "engine_mismatch_compare_temperature": compare_temperature,
        "attn_implementation": config["attn_implementation"], "dtype": "bfloat16",
        "save_hidden_states": hidden,
    }
    return rows, diagnostics


def run_select(config: dict[str, Any]) -> None:
    import json
    from transformers import AutoTokenizer
    from .datasets import load_problems
    from .tracefields import ensure_trace_fields

    out = Path(config["out_dir"])
    traces_path, stats_path = out / "traces.jsonl", out / "token_stats.parquet"
    meta_path = out / "select_meta.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    current = previous.get("stats_version") == STATS_VERSION
    if previous.get("complete") and current:
        return
    if traces_path.exists():
        traces = list(read_jsonl(traces_path))
    else:
        traces = select_traces(list(read_jsonl(out / "traces_all.jsonl")), config)
        if not traces:
            raise ValueError("No contested traces selected; inspect traces_all.jsonl or selection thresholds")
        write_jsonl(traces_path, traces)
    # Older runs lack Stage 1 fields; derive them in memory (label writes them to the dataset).
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=config["trust_remote_code"])
    questions = {str(row["id"]): row["question"] for row in load_problems(config["dataset"], config.get("input_path"))}
    for trace in traces:
        ensure_trace_fields(trace, tokenizer, questions)
    # Statistics from an older schema are recomputed; the trace selection is kept.
    existing = read_parquet(stats_path) if current else []
    rows, diagnostics = compute_token_stats(
        traces, config, existing_rows=existing, checkpoint_path=stats_path,
    )
    write_parquet(stats_path, sorted(rows, key=lambda row: (row["trace_id"], row["pos"])))
    agreement = diagnostics["argmax_agreement"]
    diagnostics["complete"] = bool(agreement is None or agreement >= 0.99)
    diagnostics["stats_version"] = STATS_VERSION
    write_json(meta_path, diagnostics)
    if agreement is not None and agreement < 0.99:
        raise AssertionError(f"HF/vLLM argmax agreement is {agreement:.3%}, below 99%; statistics and diagnostics were checkpointed")

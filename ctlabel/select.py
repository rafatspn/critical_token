from __future__ import annotations

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


def compute_token_stats(
    traces: list[dict[str, Any]], config: dict[str, Any], *,
    existing_rows: list[dict[str, Any]] | None = None, checkpoint_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=config["trust_remote_code"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, trust_remote_code=config["trust_remote_code"], device_map="auto"
    ).eval()
    rows = list(existing_rows or [])
    completed = defaultdict(set)
    for row in rows:
        completed[row["trace_id"]].add(int(row["pos"]))
    with torch.inference_mode():
        for trace in traces:
            labellable_length = int(trace.get("n_labellable_tokens", len(trace["response_ids"])))
            if len(completed[trace["trace_id"]]) >= labellable_length:
                continue
            prompt_ids, response_ids = trace["prompt_ids"], trace["response_ids"]
            all_ids = prompt_ids + response_ids
            device = next(model.parameters()).device
            logits = model(input_ids=torch.tensor([all_ids], device=device)).logits[0]
            token_texts = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in response_ids]
            number_starts = _number_starts(token_texts)
            vllm_logprobs = trace.get("vllm_top_logprobs") or []
            for pos, token_id in enumerate(response_ids[:labellable_length]):
                if pos in completed[trace["trace_id"]]:
                    continue
                prediction_index = len(prompt_ids) + pos - 1
                if prediction_index < 0:
                    raise ValueError("The prompt must contain at least one token")
                token_logits = logits[prediction_index].float()
                log_probs = torch.log_softmax(token_logits, dim=-1)
                probs = log_probs.exp()
                entropy = float(-(probs * log_probs).sum().cpu())
                top_probs, top_ids = probs.topk(5)
                target_prob = float(probs[token_id].cpu())
                rank = int((token_logits > token_logits[token_id]).sum().item()) + 1
                replacement_probs = probs.detach().cpu().clone()
                replacement_probs[token_id] = 0
                replacement_probs.pow_(1.0 / float(config["replacement_temperature"]))
                generator = torch.Generator().manual_seed(request_seed(config["seed"], trace["trace_id"], pos, "replacement"))
                replacements = torch.multinomial(
                    replacement_probs, int(config["n_sampled_alternatives"]), replacement=True, generator=generator
                ).tolist()
                row = {
                    "trace_id": trace["trace_id"], "pos": pos, "token_id": int(token_id),
                    "token_text": token_texts[pos], "token_type": token_type(token_texts[pos]),
                    "is_first_subtoken": number_starts[pos], "entropy": entropy,
                    "p_max": float(top_probs[0]), "p_y": target_prob, "rank_y": rank,
                    "top5_token_ids": [int(x) for x in top_ids.cpu().tolist()],
                    "top5_probabilities": [float(x) for x in top_probs.cpu().tolist()],
                    "replacement_token_ids": [int(x) for x in replacements],
                    "replacement_temperature": float(config["replacement_temperature"]),
                    "hf_vllm_argmax_agree": None, "hf_vllm_top5_agree": None,
                }
                if pos < len(vllm_logprobs) and vllm_logprobs[pos]:
                    vtop = sorted(vllm_logprobs[pos], key=lambda item: item.get("rank") or 10**9)
                    v_ids = [int(item["token_id"]) for item in vtop]
                    row["hf_vllm_argmax_agree"] = bool(v_ids and v_ids[0] == row["top5_token_ids"][0])
                    row["hf_vllm_top5_agree"] = set(v_ids[:5]) == set(row["top5_token_ids"])
                rows.append(row)
            if checkpoint_path is not None and rows:
                write_parquet(checkpoint_path, sorted(rows, key=lambda item: (item["trace_id"], item["pos"])))
    argmax_checks = [bool(row["hf_vllm_argmax_agree"]) for row in rows if row.get("hf_vllm_argmax_agree") is not None]
    top5_checks = [bool(row["hf_vllm_top5_agree"]) for row in rows if row.get("hf_vllm_top5_agree") is not None]
    diagnostics = {
        "n_compared": len(argmax_checks),
        "argmax_agreement": sum(argmax_checks) / len(argmax_checks) if argmax_checks else None,
        "top5_exact_set_agreement": sum(top5_checks) / len(top5_checks) if top5_checks else None,
    }
    return rows, diagnostics


def run_select(config: dict[str, Any]) -> None:
    out = Path(config["out_dir"])
    traces_path, stats_path = out / "traces.jsonl", out / "token_stats.parquet"
    meta_path = out / "select_meta.json"
    if meta_path.exists():
        import json
        if json.loads(meta_path.read_text(encoding="utf-8")).get("complete"):
            return
    if traces_path.exists():
        traces = list(read_jsonl(traces_path))
    else:
        traces = select_traces(list(read_jsonl(out / "traces_all.jsonl")), config)
        if not traces:
            raise ValueError("No contested traces selected; inspect traces_all.jsonl or selection thresholds")
        write_jsonl(traces_path, traces)
    existing = read_parquet(stats_path)
    rows, diagnostics = compute_token_stats(
        traces, config, existing_rows=existing, checkpoint_path=stats_path,
    )
    write_parquet(stats_path, sorted(rows, key=lambda row: (row["trace_id"], row["pos"])))
    agreement = diagnostics["argmax_agreement"]
    diagnostics["complete"] = bool(agreement is None or agreement >= 0.99)
    write_json(meta_path, diagnostics)
    if agreement is not None and agreement < 0.99:
        raise AssertionError(f"HF/vLLM argmax agreement is {agreement:.3%}, below 99%; statistics and diagnostics were checkpointed")


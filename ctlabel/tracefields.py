from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from .prompts import prompt_spans


def sampled_logprobs(response_ids: list[int], top_logprobs: list[list[dict[str, Any]]] | None) -> list[float | None]:
    """vLLM's log-probability of each sampled token (vLLM always returns it next to the top-k)."""
    result: list[float | None] = []
    for position, token_id in enumerate(response_ids):
        choices = top_logprobs[position] if top_logprobs and position < len(top_logprobs) else None
        match = next((item["logprob"] for item in choices or [] if int(item["token_id"]) == int(token_id)), None)
        result.append(None if match is None else float(match))
    return result


@lru_cache(maxsize=None)
def model_revision(model: str) -> str | None:
    """Commit hash of the HF model snapshot in use (local cache first)."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model, local_files_only=True, allow_patterns=["config.json"])
    except Exception:
        try:
            from huggingface_hub import HfApi
            return HfApi().model_info(model).sha
        except Exception:
            return None
    return path.rstrip("/").rsplit("/", 1)[-1]


def ensure_trace_fields(trace: dict[str, Any], tokenizer: Any, questions: Mapping[str, str]) -> dict[str, Any]:
    """Backfill Stage 1 fields added after a run was generated; values are derived, not resampled."""
    if "vllm_logprob_sampled" not in trace:
        trace["vllm_logprob_sampled"] = sampled_logprobs(trace["response_ids"], trace.get("vllm_top_logprobs"))
    if "fewshot_span" not in trace:
        trace.update(prompt_spans(tokenizer, trace["dataset"], questions[str(trace["problem_id"])], trace["prompt_ids"]))
    if "model_revision" not in trace:
        trace["model_revision"] = model_revision(trace["model"])
    return trace

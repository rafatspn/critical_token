from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "dataset": "GSM8K_test", "model": "Qwen/Qwen2.5-7B", "out_dir": "outputs/run",
    "seed": 33, "n_samples_per_problem": 8, "n_problems": None,
    "gen_temperature": 0.7, "replacement_temperature": 1.0, "rollout_temperature": 0.7,
    "top_p": 0.95, "max_tokens": 1024, "logprobs": 20,
    "min_correct": 1, "max_correct": 7, "wrong_per_problem": 2, "right_per_problem": 1,
    "n_wrong": 200, "n_right": 100, "mode": "search", "grid_stride": 32,
    "n_with_rollouts": 64, "n_sampled_alternatives": 64, "n_alt_rollouts": 16,
    "n_strat": 16, "tau_crit": 0.3, "tau_null": 0.1, "zero_frac": 3 / 64,
    "k_neg": 3, "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9,
    "trust_remote_code": True,
    "request_batch_size": 4096,
    "hf_num_gpus": 1,
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ValueError("The config root must be a mapping")
    config = deepcopy(DEFAULTS)
    config.update(supplied)
    config["dataset"] = str(config["dataset"])
    if config["mode"] not in {"search", "exhaustive"}:
        raise ValueError("mode must be 'search' or 'exhaustive'")
    return config


def dataset_base(name: str) -> str:
    base = name.removesuffix("_test").upper()
    if base not in {"MATH", "GSM8K"}:
        raise ValueError(f"Unsupported dataset: {name}")
    return base


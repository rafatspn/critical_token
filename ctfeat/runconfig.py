"""Step 2 run configs: paths plus every setting of extract, cost, the Φ replication and evaluate.

    python -m ctfeat.runconfig configs/signals_qwen25_7b_math500.yaml    # shell assignments for scripts/signals.sh

Keys missing from a YAML file take the values below, which are the settings the MATH run used.
"""
from __future__ import annotations

import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "dataset": "outputs/study_qwen25_7b_math500_exhaustive/dataset",
    "features": "outputs/signals/features",
    "results": "outputs/signals/results",
    "extract": {"layers": "every4", "mu_ref_prompts": 500, "seed": 0, "check_consistency": True, "strict_checks": True},
    "cost": {"tokens": 200, "repeats": 3},
    "phi_replication": {"m": 2000, "max_tokens": 512},
    "evaluate": {"n_boot": 1000, "seed": 0, "n_jobs": 32, "skip": []},
}
PATH_KEYS = ("dataset", "features", "results")


def load(path: str | Path) -> dict[str, Any]:
    supplied = yaml.safe_load(Path(path).read_text()) or {}
    config = deepcopy(DEFAULTS)
    for key, value in supplied.items():
        if key not in config:
            raise ValueError(f"unknown step 2 config key {key!r}")
        if isinstance(config[key], dict):
            unknown = set(value) - set(config[key])
            if unknown:
                raise ValueError(f"unknown {key} settings {sorted(unknown)}")
            config[key].update(value)
        else:
            config[key] = value
    return config


def arguments(config: dict[str, Any]) -> dict[str, list[str]]:
    extract, cost, phi, evaluate = (config[key] for key in ("extract", "cost", "phi_replication", "evaluate"))
    extract_args = ["--layers", str(extract["layers"]), "--mu-ref-prompts", str(extract["mu_ref_prompts"]), "--seed", str(extract["seed"])]
    if not extract["check_consistency"]:
        extract_args.append("--skip-consistency")
    if not extract["strict_checks"]:
        extract_args.append("--no-strict-checks")
    evaluate_args = ["--n-boot", str(evaluate["n_boot"]), "--seed", str(evaluate["seed"]), "--n-jobs", str(evaluate["n_jobs"])]
    if evaluate["skip"]:
        evaluate_args += ["--skip", *map(str, evaluate["skip"])]
    return {
        "EXTRACT_ARGS": extract_args,
        "COST_ARGS": ["--tokens", str(cost["tokens"]), "--repeats", str(cost["repeats"])],
        "PHI_ARGS": ["--m", str(phi["m"]), "--max-tokens", str(phi["max_tokens"])],
        "EVAL_ARGS": evaluate_args,
    }


def shell(config: dict[str, Any]) -> str:
    lines = [f"CFG_{key.upper()}={shlex.quote(str(config[key]))}" for key in PATH_KEYS]
    lines += [f"{name}=({' '.join(map(shlex.quote, args))})" for name, args in arguments(config).items()]
    return "\n".join(lines)


if __name__ == "__main__":
    print(shell(load(sys.argv[1])))

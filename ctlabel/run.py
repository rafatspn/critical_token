from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .io import append_jsonl


STAGES = ("generate", "select", "values", "swap", "label")


def estimate(config: dict[str, Any], stage: str) -> str:
    traces = int(config["n_wrong"]) + int(config["n_right"])
    length = 1024 if config["mode"] == "exhaustive" else 25
    values = traces * length * int(config["n_with_rollouts"])
    tested = traces * (int(config["n_strat"]) + 3)
    # Exhaustive runs already know every V_with; search runs need up to two per extra position.
    value_positions = 0 if config["mode"] == "exhaustive" else tested + traces * int(config["n_values_only"])
    swap = tested * (int(config["n_sampled_alternatives"]) + 2 * int(config["n_alt_rollouts"])) + 2 * value_positions * int(config["n_with_rollouts"])
    if stage == "values":
        return f"approximately {values:,} value-rollout completions"
    if stage == "swap":
        return f"approximately {swap:,} swap/value completions"
    return f"approximately {values + swap:,} value and swap completions"


def _confirm(config: dict[str, Any], stage: str, yes: bool) -> None:
    if yes or stage not in {"values", "swap", "all"}:
        return
    message = f"This launch is estimated at {estimate(config, stage)}. Continue? [y/N] "
    if not sys.stdin.isatty() or input(message).strip().lower() not in {"y", "yes"}:
        raise SystemExit("Cancelled. Pass --yes to confirm non-interactively.")


def _run_one(config: dict[str, Any], stage: str) -> None:
    report_after = False
    if stage == "generate":
        from .generate import run_generate
        function: Callable[[dict[str, Any]], None] = run_generate
    elif stage == "select":
        from .select import run_select
        function = run_select
    elif stage == "values":
        from .values import run_values
        function = run_values
    elif stage == "swap":
        from .swap import run_swap
        function = run_swap
    elif stage == "label":
        from .labels import run_label
        function = run_label
        report_after = True
    else:
        raise ValueError(stage)
    start = time.monotonic()
    function(config)
    elapsed = time.monotonic() - start
    if stage in {"generate", "values", "swap"}:
        gpu_count = int(config["tensor_parallel_size"])
    elif stage == "select":
        gpu_count = int(config.get("hf_num_gpus", 1))
    else:
        gpu_count = 0
    append_jsonl(Path(config["out_dir"]) / "cost.jsonl", [{
        "stage": stage, "wall_seconds": elapsed, "gpu_seconds": elapsed * gpu_count,
        "gpu_count": gpu_count,
    }])
    if report_after:
        from .report import write_report
        write_report(config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    parser.add_argument("--yes", action="store_true", help="Skip the rollout-cost confirmation")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    _confirm(config, args.stage, args.yes)
    if args.stage == "all":
        # Process isolation guarantees HF and vLLM never occupy GPU memory together.
        for stage in STAGES:
            subprocess.run([sys.executable, "-m", "ctlabel.run", "--config", args.config, "--stage", stage, "--yes"], check=True)
    else:
        _run_one(config, args.stage)


if __name__ == "__main__":
    main()


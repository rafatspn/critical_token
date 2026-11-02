"""Fail if the GSM8K rerun differs from the MATH run anywhere outside the allowed keys.

    python scripts/check_config_diff.py [--out outputs/dataset_gsm8k/config_diff.txt]

Step 1: the resolved config of each run (dataset/meta.json "config"; for GSM8K before its
label stage, configs/study_qwen25_7b_gsm8k.yaml resolved with ctlabel's defaults) and the
run metadata (model revision, dtype, attention, logprobs mode, library versions, seeds).
Step 2: the two step 2 configs, plus what extraction and evaluation recorded once they ran.
Allowed differences: dataset, out_dir and the step 2 paths. The prompt template, grader
branch and prompt spans are not config keys; they follow from `dataset`.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctfeat import runconfig  # noqa: E402
from ctlabel.config import load_config  # noqa: E402

MATH_STEP1 = REPO / "outputs/study_qwen25_7b_math500_exhaustive/dataset/meta.json"
GSM8K_STEP1_CONFIG = REPO / "configs/study_qwen25_7b_gsm8k.yaml"
MATH_STEP2_CONFIG = REPO / "configs/signals_qwen25_7b_math500.yaml"
GSM8K_STEP2_CONFIG = REPO / "configs/signals_qwen25_7b_gsm8k.yaml"
ALLOWED_STEP1 = {"dataset", "out_dir"}
ALLOWED_STEP2 = {"dataset", "features", "results"}
RUN_KEYS = ("model_revision", "dtype", "attn_implementation", "vllm_logprobs_mode", "versions", "seeds")
EXTRACT_KEYS = ("hidden_layers", "attn_stats", "received_delays", "model", "model_revision", "attn_implementation", "dtype", "versions")
EVALUATE_KEYS = ("seed", "n_boot", "model", "model_revision", "versions")
# mu_ref follows the same recipe on each dataset's own train split.
MU_REF_ALLOWED = {"dataset", "file"}


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def compare(title: str, math: dict[str, Any], gsm: dict[str, Any], allowed: set[str], lines: list[str]) -> int:
    a, b = flatten(math), flatten(gsm)
    forbidden = 0
    lines += ["", f"## {title}", ""]
    for key in sorted(set(a) | set(b)):
        if a.get(key, "<missing>") == b.get(key, "<missing>"):
            continue
        ok = key.split(".")[0] in allowed
        forbidden += not ok
        lines.append(f"- {'allowed  ' if ok else 'FORBIDDEN'} {key}: MATH={a.get(key, '<missing>')!r} GSM8K={b.get(key, '<missing>')!r}")
    lines.append(f"- {len(set(a) | set(b))} keys compared; {forbidden} forbidden differences")
    return forbidden


def read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    lines = ["# Config diff: MATH run vs GSM8K rerun"]
    forbidden = 0

    math_meta = read(MATH_STEP1)
    gsm_config = load_config(GSM8K_STEP1_CONFIG)
    gsm_meta = read(REPO / gsm_config["out_dir"] / "dataset" / "meta.json")
    source = "GSM8K dataset/meta.json" if gsm_meta else f"{GSM8K_STEP1_CONFIG.name} (planned; the run has not labelled yet)"
    lines += ["", f"- MATH step 1: {MATH_STEP1.relative_to(REPO)}", f"- GSM8K step 1: {source}"]
    forbidden += compare("Step 1 resolved config", math_meta["config"], gsm_meta["config"] if gsm_meta else gsm_config, ALLOWED_STEP1, lines)
    if gsm_meta:
        forbidden += compare("Step 1 run metadata", {k: math_meta.get(k) for k in RUN_KEYS}, {k: gsm_meta.get(k) for k in RUN_KEYS}, set(), lines)
    else:
        installed = {name: version(name) for name in math_meta["versions"]}
        forbidden += compare("Step 1 library versions (installed environment; the run has not labelled yet)",
                             math_meta["versions"], installed, set(), lines)

    math2, gsm2 = runconfig.load(MATH_STEP2_CONFIG), runconfig.load(GSM8K_STEP2_CONFIG)
    forbidden += compare("Step 2 config", math2, gsm2, ALLOWED_STEP2, lines)
    for name, keys, path in (("extraction", EXTRACT_KEYS, "extract_meta.json"), ("evaluation", EVALUATE_KEYS, "meta.json")):
        folder = "features" if name == "extraction" else "results"
        math_run, gsm_run = read(REPO / math2[folder] / path), read(REPO / gsm2[folder] / path)
        if not gsm_run:
            lines += ["", f"## Step 2 {name} record", "", f"- GSM8K {path} not written yet"]
            continue
        forbidden += compare(f"Step 2 {name} record ({path})", {k: math_run.get(k) for k in keys}, {k: gsm_run.get(k) for k in keys}, set(), lines)
        if name == "extraction":
            settings = gsm_run.get("settings", {})
            forbidden += compare("Step 2 extraction settings (GSM8K record vs MATH config)", math2["extract"], settings, set(), lines)
            mu = gsm_run.get("mu_ref", {})
            expected = {"split": "train", "n_prompts": math2["extract"]["mu_ref_prompts"], "seed": math2["extract"]["seed"]}
            forbidden += compare("Step 2 mu_ref recipe", expected, {k: mu.get(k) for k in expected}, MU_REF_ALLOWED, lines)
            lines.append(f"- GSM8K mu_ref: {mu}")

    lines += ["", f"**{'PASS' if not forbidden else 'FAIL'}**: {forbidden} forbidden differences", ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    sys.exit(1 if forbidden else 0)


if __name__ == "__main__":
    main()

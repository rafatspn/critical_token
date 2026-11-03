"""Fail if a rerun differs from its base run anywhere outside the allowed keys.

    python scripts/check_config_diff.py [--out outputs/dataset_gsm8k/config_diff.txt]          # MATH vs GSM8K (Qwen)
    python scripts/check_config_diff.py --base-meta outputs/dataset_gsm8k/meta.json \
        --config configs/study_llama31_8b_gsm8k.yaml --base-signals configs/signals_qwen25_7b_gsm8k.yaml \
        --signals configs/signals_llama31_8b_gsm8k.yaml --labels "Qwen GSM8K,Llama GSM8K" \
        --allow-step1 model,out_dir --allow-model                                              # Qwen vs Llama on GSM8K

Step 1: the resolved config of each run (dataset/meta.json "config"; for GSM8K before its
label stage, configs/study_qwen25_7b_gsm8k.yaml resolved with ctlabel's defaults) and the
run metadata (model revision, dtype, attention, logprobs mode, library versions, seeds).
Step 2: the two step 2 configs, plus what extraction and evaluation recorded once they ran.
Allowed differences (defaults): dataset, out_dir and the step 2 paths. The prompt template,
grader branch and prompt spans are not config keys; they follow from `dataset`. With
--allow-model, what follows from the model also may differ: its revision, and the layer list
of `layers: every4` (the model's depth).
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


LABELS = ["MATH", "GSM8K"]


def compare(title: str, math: dict[str, Any], gsm: dict[str, Any], allowed: set[str], lines: list[str]) -> int:
    a, b = flatten(math), flatten(gsm)
    base_label, rerun_label = LABELS
    forbidden = 0
    lines += ["", f"## {title}", ""]
    for key in sorted(set(a) | set(b)):
        if a.get(key, "<missing>") == b.get(key, "<missing>"):
            continue
        ok = key.split(".")[0] in allowed
        forbidden += not ok
        lines.append(f"- {'allowed  ' if ok else 'FORBIDDEN'} {key}: {base_label}={a.get(key, '<missing>')!r} {rerun_label}={b.get(key, '<missing>')!r}")
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
    parser.add_argument("--base-meta", type=Path, default=MATH_STEP1, help="base run's step 1 dataset/meta.json")
    parser.add_argument("--config", type=Path, default=GSM8K_STEP1_CONFIG, help="rerun's step 1 config")
    parser.add_argument("--base-signals", type=Path, default=MATH_STEP2_CONFIG)
    parser.add_argument("--signals", type=Path, default=GSM8K_STEP2_CONFIG)
    parser.add_argument("--labels", default="MATH,GSM8K", help="base,rerun names for the report")
    parser.add_argument("--allow-step1", default=",".join(sorted(ALLOWED_STEP1)), help="step 1 config keys allowed to differ")
    parser.add_argument("--allow-model", action="store_true", help="also allow the model revision and model-depth layer lists to differ")
    args = parser.parse_args()
    LABELS[:] = args.labels.split(",")
    base_label, rerun_label = LABELS
    allowed_step1 = set(args.allow_step1.split(","))
    allowed_run = {"model_revision"} if args.allow_model else set()
    allowed_record = {"model", "model_revision", "hidden_layers"} if args.allow_model else set()
    lines = [f"# Config diff: {base_label} run vs {rerun_label} rerun"]
    forbidden = 0

    math_meta = read(args.base_meta if args.base_meta.is_absolute() else REPO / args.base_meta)
    gsm_config = load_config(args.config)
    gsm_meta = read(REPO / gsm_config["out_dir"] / "dataset" / "meta.json")
    source = f"{gsm_config['out_dir']}/dataset/meta.json" if gsm_meta else f"{Path(args.config).name} (planned; the run has not labelled yet)"
    lines += ["", f"- {base_label} step 1: {args.base_meta}", f"- {rerun_label} step 1: {source}"]
    forbidden += compare("Step 1 resolved config", math_meta["config"], gsm_meta["config"] if gsm_meta else gsm_config, allowed_step1, lines)
    if gsm_meta:
        forbidden += compare("Step 1 run metadata", {k: math_meta.get(k) for k in RUN_KEYS}, {k: gsm_meta.get(k) for k in RUN_KEYS},
                             allowed_run, lines)
    else:
        installed = {name: version(name) for name in math_meta["versions"]}
        forbidden += compare("Step 1 library versions (installed environment; the run has not labelled yet)",
                             math_meta["versions"], installed, set(), lines)

    math2, gsm2 = runconfig.load(args.base_signals), runconfig.load(args.signals)
    forbidden += compare("Step 2 config", math2, gsm2, ALLOWED_STEP2, lines)
    for name, keys, path in (("extraction", EXTRACT_KEYS, "extract_meta.json"), ("evaluation", EVALUATE_KEYS, "meta.json")):
        folder = "features" if name == "extraction" else "results"
        math_run, gsm_run = read(REPO / math2[folder] / path), read(REPO / gsm2[folder] / path)
        if not gsm_run:
            lines += ["", f"## Step 2 {name} record", "", f"- {rerun_label} {path} not written yet"]
            continue
        forbidden += compare(f"Step 2 {name} record ({path})", {k: math_run.get(k) for k in keys}, {k: gsm_run.get(k) for k in keys},
                             allowed_record, lines)
        if name == "extraction":
            settings = gsm_run.get("settings", {})
            forbidden += compare(f"Step 2 extraction settings ({rerun_label} record vs {base_label} config)", math2["extract"], settings, set(), lines)
            mu = gsm_run.get("mu_ref", {})
            expected = {"split": "train", "n_prompts": math2["extract"]["mu_ref_prompts"], "seed": math2["extract"]["seed"]}
            forbidden += compare("Step 2 mu_ref recipe", expected, {k: mu.get(k) for k in expected}, MU_REF_ALLOWED, lines)
            lines.append(f"- {rerun_label} mu_ref: {mu}")

    lines += ["", f"**{'PASS' if not forbidden else 'FAIL'}**: {forbidden} forbidden differences", ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    sys.exit(1 if forbidden else 0)


if __name__ == "__main__":
    main()

"""Checks on a finished step 2 run (full feature tables, not the 5-trace smoke set).

    python scripts/check_step2_outputs.py outputs/features_gsm8k outputs/results_gsm8k

- test 2: no table column or manifest feature named like a label;
- tests 5 and 6 as recorded by extraction: feature attention vs the pinned implementation,
  prediction-head and DLA checks, W_U basis orthonormality, no recorded check failures;
- Section 5: every output file exists.
Exits 1 on any failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

FORBIDDEN = re.compile(r"V_|drop|is_|correct|swap")
FEATURE_FILES = ("tier1.parquet", "tier2.parquet", "tier3_derived.parquet", "manifest.json", "extract_meta.json")
RESULT_FILES = ("univariate_A.csv", "univariate_B.csv", "tier_models.csv", "tier_gains.csv", "probes_by_layer.csv", "delay_curve.csv",
                "breakdowns.csv", "segments.csv", "cost.csv", "report.md", "meta.json")


def main() -> None:
    features, results = Path(sys.argv[1]), Path(sys.argv[2])
    failures: list[str] = []
    lines: list[str] = []
    missing = [str(features / f) for f in FEATURE_FILES if not (features / f).exists()]
    missing += [str(results / f) for f in RESULT_FILES if not (results / f).exists()]
    missing += [str(results / "figures" / d) for d in ("", "aligned") if not list((results / "figures" / d).glob("*.png"))]
    failures += [f"missing output {m}" for m in missing]
    traces = {p.stem for p in (features / "parts" / "tier1").glob("*.parquet")}
    for directory in ("attn", "attn_recv", "hidden"):
        have = {p.stem for p in (features / directory).glob("*.npy")}
        if have != traces:
            failures.append(f"{directory}/ has {len(have)} traces, tables have {len(traces)}")
    lines.append(f"- traces with every per-trace file: {len(traces)}")

    columns = [name for table in FEATURE_FILES[:3] if (features / table).exists() for name in pq.read_schema(features / table).names]
    manifest = json.loads((features / "manifest.json").read_text()) if (features / "manifest.json").exists() else []
    leaked = [c for c in columns if FORBIDDEN.search(c)] + [m["column"] for m in manifest if FORBIDDEN.search(m["column"])]
    failures += [f"label-like feature name {c}" for c in leaked]
    lines.append(f"- test 2: {len(columns)} table columns and {len(manifest)} manifest features scanned; {len(leaked)} label-like names")

    meta = json.loads((features / "extract_meta.json").read_text()) if (features / "extract_meta.json").exists() else {}
    consistency = meta.get("attention_consistency") or {}
    checks = meta.get("head_checks") or []
    if not consistency.get("passed"):
        failures.append(f"test 5: feature attention check did not pass: {consistency}")
    if not checks or not all(c.get("logit_within_tol") and c.get("dla_max_rel_err", 1) <= 0.01 for c in checks):
        failures.append("test 6: prediction-head / DLA checks failed or missing")
    if meta.get("wu_basis_orthonormality_error", 1) > 1e-4:
        failures.append("test 6: W_U basis not orthonormal")
    failures += [f"recorded check failure: {f}" for f in meta.get("check_failures", [])]
    lines += [
        f"- test 5: logits max |Δ| {consistency.get('max_abs_logit_diff')}, top-1 {consistency.get('top1_agreement')}, "
        f"weights·V error {consistency.get('max_rel_reconstruction_err')} (eager, unused: {consistency.get('eager_max_abs_logit_diff')})",
        f"- test 6: {len(checks)} traces; max |ĥ·w − float32 logit| {max((c['max_abs_logit_diff'] for c in checks), default=None)}; "
        f"max DLA relative error {max((c['dla_max_rel_err'] for c in checks), default=None)}; "
        f"W_U orthonormality error {meta.get('wu_basis_orthonormality_error')}",
        f"- mu_ref: {meta.get('mu_ref')}",
    ]
    phi = features / "phi_replication.json"
    lines.append(f"- test 7 (optional) Φ_W replication: {json.loads(phi.read_text()) if phi.exists() else 'not run'}")
    lines += ["", f"**{'PASS' if not failures else 'FAIL'}**", *[f"- {f}" for f in failures]]
    text = "# Step 2 output checks\n\n" + "\n".join(lines) + "\n"
    print(text)
    (results / "output_checks.md").write_text(text) if results.exists() else None
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

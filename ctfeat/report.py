"""Auto-generated results/report.md (Section 7): results only, no interpretation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULTS = [
    "Attention weights: HF eager attention rounds q·kᵀ to bf16 before the softmax; with Qwen2.5-7B's attention scores (up to ~1e4) "
    "this changes the model's outputs (see the eager line above), so it fails the Section 3 check. Extraction instead runs the pinned "
    "sdpa computation unchanged and computes float32 softmax(q·kᵀ·scale + causal mask) from the same q and k for the attention features "
    "(`ctfeat/attention.py`). The check requires identical logits to the pinned implementation and float32 weights·V to reproduce each "
    "layer's sdpa output (max error ≤ 1e-2 of the output's max magnitude).",
    "Truncation test (test 1) on the real model runs the pinned weights in float32: bf16 passes over different sequence lengths differ by up to ~1 logit "
    "from kernel rounding alone, which no 1e-3 tolerance can absorb. Tolerance 1e-3 relative plus 1e-4 absolute for every feature, per-head statistic and hidden state.",
    "Prediction-head check (test 6) uses the float32 option: ĥ·w_y against the model's own final norm and lm_head applied in float32 "
    "to its final residual. The model's logits themselves are bf16; their difference from ĥ·w_y (bf16 rounding) is recorded.",
    "Query rows: response token r's pre-emission state is absolute index P+r−1 and its post-emission state is P+r. "
    "`features/attn/<trace>.npy` and `features/hidden/<trace>.npy` store one row per query index P−1 … P+L−1, so pre(r) = row r and post(r) = row r+1.",
    "`features/attn/<trace>.npy`: (rows, layers, heads, stats) in the order of `extract_meta.json:attn_stats`; "
    "`features/attn_recv/<trace>.npy`: (response tokens, delays 4/16/64, layers, heads), the mean attention received from the next k queries.",
    "Head-averaged per-layer attention features are in `features/tier2.parquet` (the spec's 'main table' for tier 2).",
    "`att_resp` counts response keys P … q (for post this includes the token itself); `att_recent16` counts keys q−15 … q.",
    "Look-ahead windows (delays 4/16/64) use the tokens that exist when the trace ends earlier; the trace's stop tokens count as tokens.",
    "`ll_top1_is_final_top1` is named `ll_top1_matches_final`, because the spec's leakage test forbids `is_` in feature names.",
    "`ll_kl_final` is KL(p_layer ‖ p_final), from the layer's distribution to the final one. The final distribution for the logit-lens "
    "comparisons (`ll_kl_final`, `ll_top1_matches_final`, `commit_layer`) is the lens at the last layer, computed in float32 like every other "
    "layer; the model's bf16-rounded logits would flip near-tied top-1s at the last layer (≈0.2% of positions) and make `commit_layer` "
    "spuriously ∞. Logit-lens and head-alignment features at `post` use the argmax of z[a] as the chosen token and the second-best as `alt`.",
    "`commit_layer` and `chosen_commit_layer` are 1-based layers; ∞ (never) is stored as +inf and mapped to n_layers + 1 for models.",
    "Group 4 (`*_L{ℓ}` head alignment) uses the final layer's `alt` and top-5 candidates at every layer, and is computed for all layers at `pre`.",
    "Direct logit attribution uses the residual increments actually added by each block (h_mid − h^{ℓ−1}, h^ℓ − h_mid), which equal the block outputs up to the bf16 add, so the contributions telescope to the final residual. `attn_out_norm` and `mlp_out_norm` use the block outputs themselves. `dla_top_block` indexes blocks as 0 = embedding, 2ℓ−1 = attention of layer ℓ, 2ℓ = MLP of layer ℓ.",
    "Φ-Align uses ĥ = the final RMSNorm output including its weight (the lm_head input). μ_ref is the mean ĥ (per layer: final RMSNorm applied to h^ℓ) over all tokens of 500 seeded MATH train prompts (few-shot + question). Windowed Φ uses the query-row states (the states that produce each response token), population covariance. Every Φ feature has a `_dnorm` version (× d), including `phi_win_64_z` and the per-layer `phi_tok_ref_dnorm_L{ℓ}` / `phi_win_64_dnorm_L{ℓ}`.",
    "`warmup` / `phi_warmup` columns flag fewer than 8 earlier tokens / window states; they are excluded from models.",
    "Univariate screen: AUROC and Spearman with 95% CIs from the same 1,000 problem-level bootstrap resamples as the tables; p-values for Benjamini–Hochberg are Wald tests with the bootstrap SE (percentile p-values cannot fall below 1/1,000, too coarse for FDR over ~20,000 features). Spearman resamples keep the global ranks. Missing values are filled with the column median for the screen only.",
    "Tier models: median imputation, standardization and L2 logistic regression (class_weight=balanced) with C ∈ {1e-3, 1e-2, 1e-1, 1} chosen by 3-fold inner grouped CV (AUROC). Target B uses ridge regression with α ∈ {0.1 … 1000}. Targets A/B/C train on wrong-trace rows; D trains on right-trace rows; all test-fold rows are scored, so right-trace negatives (the `correct_trace` group) get out-of-fold scores.",
    "Top-H heads: whole (layer, head) units, ranked on the training fold by their best statistic (|AUROC − 0.5|, or |Spearman| with drop for target B); all of a chosen head's row statistics allowed at the delay enter the model.",
    "Probes: standardize, randomized PCA to 256 dimensions, then the same logistic regression; targets A and C on wrong traces.",
    "Event-aligned bands use 200 problem-level bootstrap resamples (tables use 1,000). Segment means are computed for every feature column in the tables (plus the top per-head features and `V_with`); segments exclude the event tokens themselves; segment tests are Wilcoxon signed-rank tests over per-problem mean differences, with effect size d_z and Benjamini–Hochberg across all rows. +inf ('never') layers count as n_layers + 1 in segment means and aligned plots, as in the models.",
    "Comparison pools stay inside the target's population (wrong traces for A/C, right traces for D), except the `correct_trace` negative group of A/C, which is right-trace positions by construction.",
    "Feature tables cover every response position, including trailing stop tokens that have no label row; their `token_type` comes from step 1's own `token_type()` function. Evaluation joins features to labelled positions only.",
    "Per-head attention features (`attn/`, `attn_recv/`) are listed in `manifest.json` (table `attn` / `attn_recv`) with their layer and head.",
    "Low-entropy subset: `ent__pre` below its median over all positions in the feature tables.",
]


def _fmt(row: pd.Series) -> str:
    return f"{row['value']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def _tier_table(tier: pd.DataFrame, target: str, delay: str) -> list[str]:
    sel = tier[(tier.target == target) & (tier.delay == delay) & (tier.pool == "all_other") & (tier.mismatch == "with_mismatch")]
    if sel.empty:
        return ["(no rows)"]
    lines = ["| model | AUROC | AUPRC | false alarms / 1,000 @ 50% recall | low-entropy AUROC |", "|---|---|---|---|---|"]
    for model, chunk in sel.groupby("model", sort=False):
        get = lambda metric, subset="all": chunk[(chunk.metric == metric) & (chunk.subset == subset)]
        cells = [(_fmt(get(m).iloc[0]) if len(get(m)) else "") for m in ("auroc", "auprc", "fa_per_1000_at_50recall")]
        low = get("auroc", "low_entropy")
        cells.append(_fmt(low.iloc[0]) if len(low) else "")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return lines


def _top(univ: pd.DataFrame, key: str, tier: int) -> list[str]:
    sel = univ[(univ.tier == tier) & univ.significant].head(20) if "significant" in univ else univ.head(0)
    if sel.empty:
        return [f"(no tier-{tier} feature passes FDR control)"]
    lines = [f"| feature | {key} [95% CI] | sign | q |", "|---|---|---|---|"]
    lines += [f"| `{r.feature}` | {getattr(r, key):.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] | {int(r.sign):+d} | {r.q:.2g} |" for r in sel.itertuples()]
    return lines


def write_report(out: Path, frame: pd.DataFrame, extract_meta: dict[str, Any]) -> None:
    read = lambda name: pd.read_csv(out / name) if (out / name).exists() and (out / name).stat().st_size > 1 else pd.DataFrame()
    univ_a, univ_b = read("univariate_A.csv"), read("univariate_B.csv")
    tier, gains, probes, curve = read("tier_models.csv"), read("tier_gains.csv"), read("probes_by_layer.csv"), read("delay_curve.csv")
    segments, cost = read("segments.csv"), read("cost.csv")
    wrong = frame[~frame.correct]
    lines = ["# Critical-token signals: results", "",
             "## Dataset", "",
             f"- Problems: {frame.problem_id.nunique()}; traces: {frame.trace_id.nunique()} "
             f"({wrong.trace_id.nunique()} wrong, {frame[frame.correct].trace_id.nunique()} right); positions: {len(frame)}",
             f"- Target A (`is_crit_drop`): {int(wrong.is_crit_drop.sum())} positives / {len(wrong)} wrong-trace positions",
             f"- Target B (`drop`): {len(wrong)} wrong-trace positions; {int((wrong['drop'] >= 0.3).sum())} with drop ≥ 0.3",
             f"- Target C (`is_crit_cdpo`): {int(wrong.is_crit_cdpo.sum())} positives",
             f"- Target D (`is_crit_good`): {int(frame[frame.correct].is_crit_good.sum())} positives / {int(frame.correct.sum())} right-trace positions",
             ]
    if len(tier):
        sizes = tier[(tier.metric == "auroc") & (tier.subset == "all") & (tier.mismatch == "with_mismatch") & (tier.model == "T1") & (tier.delay == "pre")]
        lines += ["", "| target | pool | positives | negatives |", "|---|---|---|---|"]
        lines += [f"| {r.target} | {r.pool} | {int(r.n_pos)} | {int(r.n_neg)} |" for r in sizes.itertuples()]
    consistency = extract_meta.get("attention_consistency")
    if consistency:
        lines += ["", f"- Feature attention (`{consistency.get('features_attention', '?')}`) vs {consistency['pinned']}: "
                      f"max |Δlogit| {consistency['max_abs_logit_diff']:.4f}, top-1 agreement {consistency['top1_agreement']:.4%}, "
                      f"max weights·V reconstruction error {consistency.get('max_rel_reconstruction_err', float('nan')):.2e} "
                      f"(criterion ≤ 1e-2, ≥ 99.9%, ≤ 1e-2: {'pass' if consistency['passed'] else 'FAIL'})"]
        if "eager_max_abs_logit_diff" in consistency:
            lines += [f"- Eager attention (not used) vs {consistency['pinned']}: max |Δlogit| {consistency['eager_max_abs_logit_diff']:.4f}, "
                      f"top-1 agreement {consistency['eager_top1_agreement']:.4%}"]
    checks = extract_meta.get("head_checks", [])
    if checks:
        worst = lambda key: max(check.get(key, float("nan")) for check in checks)
        lines += [f"- Prediction head ({len(checks)} traces): max |ĥ·w − float32 logit| {worst('max_abs_logit_diff'):.2e} "
                  f"(≤ 1e-2: {all(check.get('logit_within_tol', False) for check in checks)}); max |ĥ·w − bf16 model logit| "
                  f"{worst('bf16_max_abs_logit_diff'):.4f}; DLA total vs logit gap: max relative error {worst('dla_max_rel_err'):.2e}"]
    for target, key, table in (("A", "auroc", univ_a), ("B", "spearman", univ_b)):
        lines += ["", f"## Top features, target {target} ({'AUROC vs pool 1' if target == 'A' else 'Spearman with drop'}, after FDR)"]
        for t in (1, 2, 3):
            lines += ["", f"### Tier {t}", "", *_top(table, key, t)]
    for target in ("A", "C", "D"):
        for delay in ("pre", "post"):
            lines += ["", f"## Tier models, target {target}, pool 1, delay `{delay}`", "", *_tier_table(tier, target, delay)]
    if len(gains):
        lines += ["", "## AUROC gain over Tier 1 (paired bootstrap, pool 1)", "", "| target | delay | model | gain [95% CI] |", "|---|---|---|---|"]
        lines += [f"| {r.target} | {r.delay} | {r.model} | {r.auroc_gain_over_T1:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |" for r in gains.itertuples()]
    lines += ["", "## Hidden-state probes", "", "![probes](figures/probes_by_layer.png)", ""]
    if len(probes):
        lines += ["| layer | delay | target | AUROC [95% CI] | Spearman with drop [95% CI] |", "|---|---|---|---|---|"]
        lines += [f"| {r.layer} | {r.delay} | {r.target} | {r.auroc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] | "
                  f"{'' if pd.isna(getattr(r, 'spearman_drop', float('nan'))) else f'{r.spearman_drop:.3f} [{r.spearman_ci_low:.3f}, {r.spearman_ci_high:.3f}]'} |"
                  for r in probes.itertuples()]
    lines += ["", "## Delay curve", "", "![delay curve](figures/delay_curve.png)", ""]
    if len(curve):
        lines += ["| target | tier | chosen model | delay | value [95% CI] |", "|---|---|---|---|---|"]
        lines += [f"| {r.target} | {r.tier} | {r.chosen_model} | {r.delay} | {r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] |"
                  for r in curve.itertuples()]
    lines += ["", "## Event-aligned trajectories", "",
              "Plots: `figures/aligned/aligned_<feature>.png` (around `crit_drop` and `crit_cdpo`, with the right-trace control), "
              "`figures/aligned/between_label_stretch.png`, `figures/aligned/hidden_drift.png`.", ""]
    if len(segments):
        top = segments.sort_values("q").head(30)
        lines += ["| feature | from → to | mean from | mean to | d_z | problems | q |", "|---|---|---|---|---|---|---|"]
        lines += [f"| `{r.feature}` | {r.seg_from} → {r.seg_to} | {r.mean_from:.4g} | {r.mean_to:.4g} | {r.effect_size_dz:+.2f} | {r.n_problems} | {r.q:.2g} |"
                  for r in top.itertuples()]
    lines += ["", "## Cost per feature family", ""]
    if len(cost):
        lines += ["| family | attention | ms/token | extra ms/token | vLLM without custom hooks |", "|---|---|---|---|---|"]
        lines += [f"| {r.family} | {r.attn_implementation} | {r.ms_per_token:.2f} | {r.extra_ms_per_token:+.2f} | {r.vllm} |" for r in cost.itertuples()]
    else:
        lines.append("Not measured yet: run `python -m ctfeat.cost` on a GPU, then regenerate this report.")
    lines += ["", "## Defaults not fixed by the task", "", *[f"- {item}" for item in DEFAULTS], ""]
    (out / "report.md").write_text("\n".join(lines))

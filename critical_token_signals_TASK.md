# Task: find online signals that detect critical tokens (step 2)

## 0. Context and scope

Step 1 built a labelled dataset of reasoning traces: Qwen2.5-7B base, MATH500, 200 wrong + 100 right traces, with exhaustive prefix values for every token of every wrong trace. It lives in `dataset/` (see Section 2).

Step 2 asks: **which signals, computable while the model is generating, detect critical tokens?** Signals come in three tiers, by what they need at inference time:

| Tier | Signals | Needs at inference |
|---|---|---|
| 1. Output | entropy family, from the next-token distribution | nothing extra |
| 2. Attention | per-head attention summaries | access to attention weights |
| 3. Hidden states | logit lens, residual-stream geometry, linear probes | access to hidden states |

**Primary target:** the *sharpest-drop* tokens (`is_crit_drop`) and the graded score `drop(i)`. A step 1 result showed replacing the sharpest-drop token repairs a trace far more often than replacing cDPO's token, so these are the tokens worth detecting. cDPO's tokens and critical-for-good tokens are secondary targets.

**In scope:** causal feature extraction, univariate and multivariate evaluation of each tier, the delay curve, cost per feature family, and a report.

**Out of scope:** new rollouts, changing the step 1 labels or negatives, interventions, test-time scaling. Known dataset limitations are recorded in `claude/critical-token-dataset-notes.md` (the project doc); don't try to fix them here.

---

## 1. The online rule (read first: this is the main correctness requirement)

A detector is "online" if, when it scores response token r, it uses only information available at that point of generation.

**Indexing.** Let P = `len(prompt_ids)`. Response token r sits at absolute index a = P + r in the full sequence `prompt_ids + response_ids`. In a teacher-forced forward pass over the full sequence:

- **Pre-emission state** (index a − 1): the forward step that *produced the distribution* from which y_r was sampled. Hidden states `h^ℓ[a−1]`, attention rows with query a − 1, and logits `z[a−1]` (whose softmax is p_r).
- **Post-emission state** (index a): the forward step that takes y_r as input. Hidden states `h^ℓ[a]`, attention rows with query a, and logits `z[a]` (the distribution for y_{r+1}). In a real decoder this is computed right after y_r is emitted, before y_{r+1} is chosen, so it's also online.

**Delay levels.** Every feature carries a `delay` tag:

| Delay | Features may use | Meaning |
|---|---|---|
| `pre` | everything at indices ≤ a − 1, plus the chosen token y_r | scored as the token is emitted |
| `post` | also index a | scored one forward step later |
| `k` (4, 16, 64) | also indices ≤ a + k | scored k tokens later |

**Forbidden in any feature:**

- anything from rollouts (V values, drops, swap results, labels);
- whether the trace was correct, its final answer, or its length L;
- any statistic normalized over the whole trace (e.g. the step 1 within-trace entropy quintiles), or any other information from beyond the declared delay;
- anything fit on test-fold data (scalers, PCA, head selection, probes).

**Running statistics** (z-scores, moving averages) use only earlier response tokens, over a causal window (default W = 64), starting from the first response token. For the first few tokens, use whatever is available and mark `warmup = true` when fewer than 8 earlier tokens exist.

The test suite (Section 6) enforces this with a truncation test: features for position r computed on the sequence truncated after index a + delay must equal those computed on the full sequence.

---

## 2. Inputs

From `dataset/` (step 1 output):

- `traces.jsonl`: `trace_id, problem_id, prompt_ids, response_ids, fewshot_span, question_span, cue_span, correct, ...`
- `positions.parquet`: one row per (trace_id, pos). Uses `V_before`, `V_with`, `is_crit_cdpo`, `is_crit_drop`, `is_fixable`, `is_crit_good`, `is_noncritical`, `is_consequence`, `is_ambiguous`, `entropy_T1`, `token_type`, `engine_mismatch`. The `pos` column is the response index r.
- `negatives.parquet`: `crit_trace_id, crit_pos, crit_label_kind, group, neg_trace_id, neg_pos`.
- `meta.json`: model revision, dtype, `attn_implementation`, library versions.

Before anything else, check:

- labels come from **exhaustive** values (`mode == exhaustive` for all wrong traces); stop with an error if not;
- `V_before` and `V_with` exist for every wrong-trace position. Define `drop = V_before − V_with`.

Load the model at the exact revision and dtype in `meta.json`.

---

## 3. Feature extraction

Create a package `ctfeat/` next to `ctlabel/`. CLI:

```
python -m ctfeat.extract --dataset dataset/ --out features/ [--traces N] [--layers every4]
```

Run **one teacher-forced forward pass per trace** over `prompt_ids + response_ids` with `output_hidden_states=True` and `output_attentions=True`. Attention weights require `attn_implementation="eager"`. Before extracting, check on 5 traces that eager-attention logits match the pinned implementation from `meta.json` (max absolute logit difference ≤ 1e-2 in bf16, top-1 agreement ≥ 99.9%), and record the result. Compute every feature on the GPU inside the pass and keep only summaries. Never write full attention maps to disk.

Extract features for **every response position of every trace** (wrong and right).

### Tier 1: output (delay `pre`, from `z[a−1]` and y_r)

| Feature | Definition |
|---|---|
| `ent` | entropy of softmax(z[a−1]) at temperature 1 |
| `varent` | varentropy at temperature 1 |
| `pmax` | top-1 probability |
| `p_chosen`, `logp_chosen`, `rank_chosen` | probability, log-probability and rank of y_r |
| `margin12` | top-1 minus top-2 probability |
| `surprisal_gap` | −log p(y_r) − ent (how unexpected the chosen token was relative to the uncertainty) |
| `ent_z`, `pmax_z` | z-score of `ent` / `pmax` against the causal window of earlier response tokens |
| `ent_ema_delta` | `ent` minus its exponential moving average (half-life 16 tokens) over earlier tokens |
| `ent_prev_delta` | `ent` minus the previous token's `ent` |
| `token_type` | from `positions.parquet` (one-hot) |
| `r`, `step_index` | absolute response position and reasoning-step index (no division by L) |

**Delay `post`:** `ent_next`, `pmax_next` (from z[a]).
**Delays 4, 16, 64:** mean and max of `ent` over tokens r+1 … r+k; mean of `logp_chosen` over r+1 … r+k.

### Tier 2: attention (eager attention; Qwen2.5-7B has 28 layers × 28 query heads)

For each layer ℓ and head h, with query q = a − 1 (delay `pre`) and q = a (delay `post`), compute over keys ≤ q:

| Feature | Definition |
|---|---|
| `att_ent` | entropy of the attention row |
| `att_max` | largest attention weight |
| `att_argmax_dist` | q minus the index of the most-attended key |
| `att_fewshot`, `att_question`, `att_cue`, `att_resp`, `att_recent16`, `att_sink` | attention mass on the few-shot span, question span, cue span, earlier response tokens, the last 16 tokens, and index 0 |
| `lookback_ratio` | `att_question / (att_question + att_resp + 1e-8)` (Lookback Lens style) |

Store these as a float16 array of shape (positions, layers, heads, n_stats), one file per trace (`features/attn/<trace_id>.npy`), plus head-averaged versions per layer in the main table.

**Delayed attention (delays 4, 16, 64):** the attention *received* by key a from queries a+1 … a+k: the mean per (layer, head), and the max over heads per layer. This is the token-level version of Thought Anchors' "receiver heads".

### Tier 3: hidden states

Save raw hidden states for the probes:

- `h_pre` = `h^ℓ[a−1]` and `h_post` = `h^ℓ[a]` at layers {4, 8, 12, 16, 20, 24, 28}, as float16 in `features/hidden/<trace_id>.npy`. Print the expected disk size first (about 8 GB for this dataset).

Derived features, computed for all 28 layers at both `pre` and `post`:

| Feature | Definition |
|---|---|
| `h_norm` | ‖h^ℓ‖ |
| `upd_norm` | ‖h^ℓ − h^{ℓ−1}‖ |
| `attn_out_norm`, `mlp_out_norm` | norms of each block's output at layer ℓ (forward hooks) |
| `upd_cos` | cosine between the layer update and the incoming residual |
| `tmp_cos` | cosine between h^ℓ at this position and at the previous position (delay `pre` uses a−1 and a−2) |
| **Logit lens** | apply the model's final norm and `lm_head` to h^ℓ[a−1]: `ll_ent`, `ll_p_chosen`, `ll_rank_chosen`, `ll_kl_final` (KL from the layer's distribution to the final one), `ll_top1_is_final_top1` |
| `commit_layer` | first layer from which the logit-lens top-1 equals the final top-1 for every later layer |
| `n_top1_flips` | number of layer-to-layer changes in the logit-lens top-1 |
| `chosen_commit_layer` | first layer from which y_r stays in the logit-lens top-1 (∞ if never) |

**Prediction-head alignment** (delay `pre`, from the state at index a − 1). Notation:

- `W_U` is `lm_head.weight` (vocab × d); Qwen2.5-7B's input and output embeddings are untied, so use `lm_head`, not the input embedding.
- `w_y` is the row of `W_U` for token y.
- `ĥ` is the final hidden state after the model's final RMSNorm, including its weight, so that `logit(y) = ĥ · w_y`.
- `alt` is the highest-probability token other than y_r.

Group 1, scale vs direction (final layer):

| Feature | Definition |
|---|---|
| `head_scale` | ‖ĥ‖ |
| `head_cos_chosen` | cos(ĥ, w_{y_r}) |
| `head_cos_alt` | cos(ĥ, w_alt) |
| `head_cos_gap` | `head_cos_chosen − head_cos_alt` |
| `head_wnorm_chosen`, `head_wnorm_alt` | ‖w_{y_r}‖ and ‖w_alt‖ (so the logit gap can be decomposed exactly into scale, weight norms and cosines) |

Group 2, direct logit attribution toward the chosen token vs the alternative:

- Let `d = w_{y_r} − w_alt`. Linearize the final norm by freezing its scale: s = the RMS of the final-layer residual at index a − 1, and g = the norm's weight vector. Each block output o (embedding, every layer's attention output, every layer's MLP output) then contributes `c(o) = ((o / s) ⊙ g) · d` to the logit gap.

| Feature | Definition |
|---|---|
| `dla_attn_L{ℓ}`, `dla_mlp_L{ℓ}` | c(o) for layer ℓ's attention and MLP outputs (all 28 layers) |
| `dla_embed` | c(o) for the token embedding |
| `dla_total` | sum of all contributions; must equal the logit gap ĥ · d up to the RMSNorm epsilon (see tests) |
| `dla_conflict` | sum of the *negative* contributions (blocks pushing toward `alt`), as a positive number |
| `dla_conflict_frac` | `dla_conflict / (sum of absolute contributions)` |
| `dla_decision_layer` | the last layer at which the running sum over layers (embedding, then attention and MLP of layer 1, 2, …) changes sign; 0 if it never does |
| `dla_top_block` | index of the block with the largest positive contribution |

Group 3, how much of the representation the head reads out:

- Compute once: the right singular vectors of `W_U` (vocab × d), via an eigendecomposition of `W_U^T W_U` (d × d) in float32. Order the singular directions by singular value.

| Feature | Definition |
|---|---|
| `readout_frac_64`, `readout_frac_512` | share of ‖ĥ‖² in the top-64 and top-512 singular directions of `W_U` |
| `cand_span_resid` | 1 − ‖proj(ĥ onto span{w of the top-5 tokens})‖² / ‖ĥ‖² |

Group 4, alignment across depth: compute `head_cos_chosen`, `head_cos_gap`, `readout_frac_64` and `cand_span_resid` per layer ℓ, using `ĥ^ℓ` = the final RMSNorm applied to h^ℓ[a−1] (as in the logit lens). Name them with a `_L{ℓ}` suffix.

Groups 1–3 are also computed at delay `post` (from index a, with y_r replaced by the argmax of z[a] and `alt` by the second-best token), so the delay curve can use them.

Group 5, **Φ-Align features.** These adapt the Φ-Align metric ("Φ-Align: Assessing Language Model Representations Through Prediction-Head Alignment", anonymous, under review at ICLR 2027) from a model-level score to per-token and windowed online features. They are **in addition to** Groups 1–4, which stay unchanged (including their uncentered head).

- **Centered head.** `W_c = W_U − 1·w̄ᵀ`, where `w̄` is the mean row of `W_U`. This removes the component that shifts all logits equally and so doesn't affect the softmax. Compute once, in float32: `W_cᵀW_c = W_UᵀW_U − |V|·w̄w̄ᵀ` (d × d), then its eigendecomposition gives the head's sensitivity directions `v_j` and strengths `s_j²`. Then `‖W_c‖²_F = Σ_j s_j²`, and for any vector x, `‖W_c x‖² = Σ_j s_j² ⟨x, v_j⟩²`. Precompute `u = Vᵀx` per token so every Φ quantity is a weighted sum of squares.
- **Representations.** Use `ĥ`, the final hidden state after the final RMSNorm (the input to `lm_head`). Record this choice in the README; the paper says "final-layer hidden state" without specifying the norm.
- **Reference means (label-free, so no fold leakage):**
  - `μ_ref`: the mean of `ĥ` over all tokens of 500 MATH **train-split** prompts (few-shot plus question), computed once;
  - `μ_prompt`: per trace, the mean of `ĥ` over that trace's own prompt tokens. This is available before generation starts, so it's online.

| Feature | Definition |
|---|---|
| `phi_tok_ref` | `‖W_c x‖² / (‖x‖² · ‖W_c‖²_F)` with `x = ĥ − μ_ref`: Φ-Align applied to a single centered token |
| `phi_tok_prompt` | the same, with `x = ĥ − μ_prompt` |
| `phi_tok_raw` | the same, with `x = ĥ` (centered head, uncentered representation) |
| `phi_win_W` for W ∈ {32, 64} | Φ-Align (the paper's Eq. 4) on the covariance of `ĥ` over the last W response positions, centered by the window mean: `tr(W_c Σ W_cᵀ) / (tr(Σ) · ‖W_c‖²_F)`. Use only states at indices ≤ the delay limit. Mark `warmup = true` when fewer than 8 states are in the window. Compute with running sums of u and u², not by rebuilding Σ |
| `phi_win_64_delta16` | `phi_win_64` minus its value 16 tokens earlier |
| `phi_win_64_z` | z-score of `phi_win_64` against its own causal window of earlier values |
| `*_dnorm` | every Φ feature multiplied by d (the paper's width-normalized dΦ_W; isotropic baseline = 1) |

Compute these at delay `pre` (states up to index a − 1) and at delay `post` (states up to index a). Compute `phi_tok_ref` and `phi_win_64` **per layer** too, at {4, 8, …, 28}, applying the final RMSNorm to h^ℓ first, with suffix `_L{ℓ}`. Use per-layer reference means for these.

**Output tables.** `features/tier1.parquet` and `features/tier3_derived.parquet`, one row per (trace_id, pos), with a `delay` suffix on each column name (e.g. `ent__pre`, `ll_kl_final_L16__pre`). Also write `features/manifest.json` listing every feature with its tier, delay, layer and head.

---

## 4. Evaluation

Create `ctfeat/evaluate.py`. Every statistic is **grouped by `problem_id`**: cross-validation folds and bootstrap resamples are drawn over problems, never over traces or positions.

### 4.1 Targets and comparison pools

| Target | Positives | Notes |
|---|---|---|
| **A (primary)** | `is_crit_drop` | |
| **B (primary, graded)** | `drop` at every wrong-trace position | correlation and ranking, no threshold |
| C | `is_crit_cdpo` | comparison with the cDPO paper |
| D | `is_crit_good` | right traces; analyse separately, never pooled with A |

Comparison pools (negatives) for the binary targets:

1. **all other wrong-trace positions** (excluding the target positives);
2. `is_noncritical` positions;
3. each group in `negatives.parquet`, deduplicated to unique (trace, pos) per group.

### 4.2 Metrics

For each detector score, report with problem-level bootstrap 95% CIs (1,000 resamples):

- AUROC and AUPRC per pool;
- **false alarms per 1,000 tokens at 50% recall** (the online-relevant number: how often a detector would fire on non-critical tokens);
- AUROC on the **low-entropy subset** (positives and negatives with `ent__pre` below its dataset median);
- for target B: Spearman correlation with `drop`, and precision@1% (the share of the top 1% of scored positions with `drop ≥ 0.3`).

Where it applies, report each metric with and without `engine_mismatch` positions.

### 4.3 Analyses

1. **Univariate screen.** For every scalar feature (every layer and head for the per-layer and per-head ones), compute AUROC for target A against pool 1, and Spearman with `drop` for target B. Flip each feature's sign so higher means "more critical" before reporting. Control the false discovery rate across features (Benjamini–Hochberg, q = 0.05). Output a ranked table, plus heatmaps:
   - layer × statistic for tier 3 derived features;
   - layer × {attention, MLP} for the direct-logit-attribution contributions (`dla_attn_L{ℓ}`, `dla_mlp_L{ℓ}`), shown as the mean contribution at positives minus the mean at negatives;
   - layer × head for the main attention statistics.
2. **Tier models.** Logistic regression (L2, C chosen by inner CV) with 5-fold grouped CV:
   - Tier 1 only;
   - Tier 1 + Tier 2 (head-averaged per layer);
   - Tier 1 + Tier 2 (top-H heads, H ∈ {8, 32}, **selected inside each training fold** by univariate AUROC);
   - Tier 1 + Tier 3 derived;
   - all three.

   Run each at delay `pre`, then `post`. Report the AUROC gain of each model over Tier 1 alone, with paired bootstrap CIs.
3. **Hidden-state probes.** Per layer in {4, …, 28}, for `h_pre` and `h_post`: standardize, PCA to 256 dimensions, then L2 logistic regression. Everything is fit inside the training fold. Report AUROC by layer for targets A and C, and probe-score Spearman with `drop`.
4. **Delay curve.** For delays `pre`, `post`, 4, 16, 64: the best tier model's AUROC (targets A and B) using all features allowed at that delay. One line per tier.
5. **Breakdowns.** For the best `pre` and `post` models: AUROC by token type and by entropy quintile (quintiles over the whole dataset, used only for reporting).
6. **Cost.** For each feature family, the measured extra latency per token in a decoder loop (HF `generate` with hooks, batch size 1, 200 tokens), and whether it's available in vLLM without custom hooks.
7. **Event-aligned trajectories** (descriptive; labels are used only to align traces, never as features). Features exist at every token, so this looks at how signals evolve around critical tokens and in the stretch between them:
   - **Alignment events:**
     - `crit_drop` in wrong traces;
     - `crit_cdpo` in wrong traces;
     - as a control, random positions in right traces matched on response position r (±16).
   - **Plot:** for the top 20 features from the univariate screen, plus `ent__pre`, `pmax__pre`, `logp_chosen__pre`, logit-lens `ll_kl_final` at layers 8/16/24, `commit_layer`, `phi_tok_ref`, `phi_win_64` and `phi_win_64_delta16`, plot the mean (with problem-level bootstrap bands) over the window −64 … +64 tokens around each event, with the control drawn on the same axes. Also show the value curve `V_with` aligned the same way.
   - **Between-label stretch:** for wrong traces where `crit_drop` comes before `crit_cdpo`, rescale each trace's segment between the two to 0–1 (20 bins) and plot features over it, together with 32 tokens before the drop and 32 after cDPO's token. Compare with the same-length segment from right traces starting at a matched r.
   - **Segment means:** for each feature, compare three segments of each wrong trace with a paired test grouped by problem: before the drop, between the drop and cDPO's token, and after cDPO's token. Report the effect size per pair of segments, with FDR control.
   - **Hidden-state drift:** per layer in {8, 16, 24}, the cosine similarity of `h_post` at each position to `h_post` at the drop position, over −64 … +64. This shows whether the representation shifts abruptly at the drop or drifts.
   - Write the plots to `results/figures/aligned/`, and the segment table to `results/segments.csv`.

### 4.4 Baselines to always include

- `ent__pre` alone; `pmax__pre` alone; `−logp_chosen__pre` alone;
- `ent_z__pre` alone (the relative-spike rule);
- random scores (sanity check: AUROC ≈ 0.5).

---

## 5. Outputs

```
features/            tier1.parquet, tier3_derived.parquet, attn/, hidden/, manifest.json
results/
  univariate_A.csv, univariate_B.csv
  tier_models.csv          # per model × delay × target × pool × metric, with CIs
  probes_by_layer.csv
  delay_curve.csv
  breakdowns.csv
  segments.csv
  figures/aligned/         # event-aligned and between-label trajectory plots
  cost.csv
  figures/                 # PNG + PDF for each heatmap and curve
  report.md                # auto-generated summary, Section 7
```

---

## 6. Tests (all must pass)

1. **Truncation test (causality).** For 10 random traces and 30 random positions each, recompute every feature on the sequence truncated after index a + delay and compare with the full-sequence value. Tolerance: 1e-3 relative (1e-2 for bf16 hidden-state norms). Any mismatch fails. This catches future leakage, including through running statistics.
2. **No label leakage.** Scan the `features/` tables: no column whose name contains `V_`, `drop`, `is_`, `correct` or `swap`. Also assert that the feature extraction code never opens `positions.parquet`, except for `token_type`, and never opens `negatives.parquet`.
3. **Fold hygiene.** A unit test where a scaler, PCA, head selector or probe is fit on fold data must not see any test-fold problem_id.
4. **Indexing.** On a hand-built 5-token example, `ent__pre` at r = 0 equals the entropy of the logits at index P − 1, and `ent__post` at r = 0 equals the entropy at index P.
5. **Eager-attention consistency** (Section 3) is run and logged.
6. **Prediction-head consistency.**
   - `ĥ · w_y` reproduces the model's logits at the final layer (max absolute difference ≤ 1e-2 in bf16, or compute in float32).
   - `dla_total` equals the logit gap `ĥ · (w_{y_r} − w_alt)` within 1% relative error on 100 random positions (the frozen-scale linearization is exact apart from the RMSNorm epsilon).
   - The `W_U` singular directions are orthonormal: ‖VᵀV − I‖ ≤ 1e-4.
7. **Φ-Align checks.**
   - Isotropic baseline: for 10,000 random isotropic Gaussian vectors, the mean of `phi_tok_raw` is within 2% of 1/d.
   - Invariances: Φ features are unchanged when x is rescaled, and when a constant vector b is added to every row of `W_U` (W_U → W_U + 1·bᵀ, the softmax-invariant shift the centering removes).
   - Windowed vs direct: on 5 random windows, `phi_win_64` from running sums equals the value from an explicitly built covariance matrix (relative error ≤ 1e-4).
   - **Optional replication of the paper's model-level number** (run it if FineWeb is downloadable): compute Φ_W for Qwen2.5-7B base from the final-layer, last-token states of M = 2,000 FineWeb sequences truncated to 512 tokens. The paper reports Φ_W × 10³ = 0.109 with M = 15,000. Log the value and the gap; it doesn't block the run.
8. **Smoke test:** the full pipeline on 5 traces with layers {4, 28} finishes and writes every output file.

---

## 7. `report.md` (auto-generated)

- dataset sizes per target and pool, and the number of problems;
- the top 20 features per tier for targets A and B (AUROC or Spearman, with CIs, after FDR control);
- the tier-model table at `pre` and `post` for target A, pool 1: AUROC, AUPRC, false alarms per 1,000 tokens at 50% recall, low-entropy AUROC;
- the gain of Tier 2 and Tier 3 over Tier 1, with CIs;
- probe AUROC by layer (figure);
- the delay curve (figure);
- the same tables for target C, and for target D separately;
- event-aligned plots around `crit_drop` and `crit_cdpo` (with the right-trace control), the between-label stretch plot, and the segment-means table;
- cost per feature family;
- every default this document didn't fix.

Write only results in the report, with no interpretation; interpretation is for the researcher.

---

## 8. Practical notes

- **Compute.** About 300 forward passes over roughly 1,200-token sequences. Eager attention and the per-head summaries dominate; expect well under an hour on one GPU. Process traces one at a time and free GPU tensors after each.
- **Memory.** Eager attention at 1,200 tokens is about layers × heads × L² × 2 bytes ≈ 2.2 GB in bf16 if all layers are held at once. Compute attention summaries inside a forward hook per layer and discard the map immediately, instead of using `output_attentions=True` for all layers at once, if memory is tight.
- **Reproducibility.** Pin the seeds for CV folds and bootstraps, and write the library versions and the model revision to `results/meta.json`.
- **Code quality.** Type hints, one function per feature family, a feature registry that records tier and delay, and the manifest generated from that registry.

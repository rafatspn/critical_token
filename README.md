# ctlabel

`ctlabel` builds a token-level critical-token dataset from token-exact counterfactual rollouts. It adapts the Apache-2.0 [Critical Tokens Matter](https://github.com/chenzhiling9954/Critical-Tokens-Matter) rollout method, fixes its decode/re-encode boundary bug, and adds right traces, adaptive search, CFT-style alternatives, confidence intervals, and matched negatives.

## Install and run

Python 3.10+ and a CUDA environment supported by current vLLM are required.

```bash
pip install -e '.[test]'
python -m ctlabel.run --config configs/smoke.yaml --stage all --yes
```

Run stages independently to resume a partial job:

```bash
python -m ctlabel.run --config configs/study_qwen25_7b_math500.yaml --stage generate
python -m ctlabel.run --config configs/study_qwen25_7b_math500.yaml --stage select
python -m ctlabel.run --config configs/study_qwen25_7b_math500.yaml --stage values --yes
python -m ctlabel.run --config configs/study_qwen25_7b_math500.yaml --stage swap --yes
python -m ctlabel.run --config configs/study_qwen25_7b_math500.yaml --stage label
```

`all` launches every stage in a fresh child process. This is intentional: the Hugging Face model used for exact statistics and the vLLM engine used for generation/rollouts never share GPU memory. vLLM is created with prefix caching enabled. Without `--yes`, expensive stages print a rollout estimate and request confirmation.

Provided configurations are:

- `smoke.yaml`: five GSM8K test problems, Qwen2.5-0.5B, rollout count 4.
- `replicate_qwen25_7b_{gsm8k,math}.yaml`: 100 wrong train traces, exhaustive 64-rollout values and Qwen's cDPO temperatures (0.9/0.5).
- `study_qwen25_7b_math500.yaml`: MATH500 adaptive search, 200 wrong and 100 right traces.

All stochastic requests use a stable seed derived from `(global_seed, trace_id, position, variant)`. Position `i` indexes `response_ids`; `-1` in the intermediate values file denotes the prompt alone. EOS and stop tokens are not response labels.

## Outputs

Intermediate files in `out_dir` are resumable:

- `traces_all.jsonl`: `trace_id`, `problem_id`, `dataset`, `model`, `model_revision` (HF snapshot commit), exact `prompt_ids` and `response_ids`, `fewshot_span`/`question_span`/`cue_span` (`[start, end)` token ranges in `prompt_ids`; `prompt_spans_merged` is true when tokenization merged across a segment boundary, in which case the straddling token is assigned to the earlier segment), `vllm_logprob_sampled` (vLLM log-probability of each sampled token), `n_labellable_tokens`, `stop_unaligned` (true when a stop string could not be aligned to token boundaries, so its tokens remain labellable), `response_text`, `finish_reason`, `stop_reason`, `pred`, `gold`, `correct`, `gen_temperature`, `seed`, and `vllm_top_logprobs`. `response_ids` preserves the exact vLLM output; only positions below `n_labellable_tokens` are analyzed.
- `traces.jsonl`: all `traces_all` columns for selected contested traces, plus `crit_cdpo`, `crit_drop`, `crit_good_candidate`, `non_monotone`, and `values_complete`. Exhaustive wrong-trace runs also store `crit_cdpo_search_replay`, `search_replay_positions`, and `exhaustive_positions`.
- `token_stats.parquet`: `trace_id`, `pos`, `token_id`, `token_text`, `token_type`, `type_group` (`digit_operator`, `connective`, `latex_punct_other`), `is_first_subtoken`, `step_index` (newlines in the response before the token), `entropy_T1` (full vocabulary, temperature 1), `entropy_gen` (generation temperature, before top-p), `varentropy_T1`, `p_max`, `p_y`, `p_y_gen` (p(y_i) at the generation temperature), `rank_y`, `margin_top2` (p1 − p2), `top5_token_ids`, `top5_probabilities`, `entropy_quintile` (within the trace, 0 = lowest), `replacement_token_ids`, `replacement_temperature`, `hf_vllm_argmax_agree`, diagnostic `hf_vllm_top5_agree`, `vllm_p_sampled`, and `engine_mismatch` (vLLM's sampled-token probability differs from HF's by more than `engine_mismatch_tol`, or the top-2 order differs; kept, not dropped). With `save_hidden_states: {layers: [...], dtype: float16}`, the same forward pass also writes `hidden/<trace_id>.npy` of shape `(len(layers), n_labellable_tokens, hidden_size)`; layer `0` is the embedding output and layer `l` the output of decoder layer `l`.
- `values.parquet`: `trace_id`, `pos` (`-1` is prompt-only), `n_correct`, `n_total`, `rollout_temperature`, and `mode`.
- `swap.parquet`: `trace_id`, `pos`, `incl_prob`, `stratum` (`special`, `q<quintile>:<type_group>`, `nearby` or `random`), `swap_done` (false for values-only positions, which carry only `with_*`/`before_*`); `with_*`, `before_*`, `without_sampled_*`, `alt2_*`, and `alt3_*` IDs/counts; `replacement_token_ids`, `replacement_token_counts`, `replacement_temperature`, and `rollout_temperature`.
- `cost.jsonl`: actual generated-token and stage-runtime records.
- `rollouts_values.jsonl`, `rollouts_swap.jsonl`: per-request rollout counts appended after every vLLM batch, keyed by `(trace_id, pos, variant, seed, n, rollout_temperature)`. A stage killed at the wall-time limit skips these requests on restart.

Final files under `out_dir/dataset` are:

- `positions.parquet`: one row for every labelable selected response token. It includes every `token_stats` column; available `swap` columns; raw `with_*`/`before_*` counts; `v_with`, `v_before`, `drop`; Wilson `with_*`, `before_*`, `without_sampled_*`, `alt2_*`, and `alt3_*` confidence bounds; `drop_ci_*`, `fixable_effect`, `fixable_ci_*`; `v_without_sampled`, `v_alt2`, `v_alt3`; `correct_trace`, `model`, `zero_frac`, `entropy_decile`, `incl_prob`, `stratum`, `swap_done`; and `is_crit_cdpo`, `is_crit_drop`, `is_fixable`, `is_crit_good`, `is_noncritical`, `is_consequence`, and `is_ambiguous`. Unevaluated positions remain present and ambiguous.
- `traces.jsonl`: exact tokenized traces and trace-level search results.
- `negatives.parquet`: `crit_trace_id`, `crit_pos`, `crit_label_kind`, `group`, `neg_trace_id`, and `neg_pos`. To redraw negatives with other rules, call `ctlabel.labels.negatives_from_positions("dataset/positions.parquet", k, seed)` or `sample_negatives(rows, k, seed)`; both use only `positions.parquet` columns.
- `meta.json`: `config`, `model_revision`, `dtype`, `attn_implementation`, `vllm_logprobs_mode`, `versions`, `seeds`, `generated_tokens_by_stage`, `gpu_hours_by_stage`, and `generated_tokens_total`.
- `summary.md`: label/trace counts, entropy and type checks, `engine_mismatch` share (overall and at critical tokens), confidently non-critical counts per entropy-quintile × type-group cell (cells under 30 listed), lag, wrong-trace monotonicity, per-stage compute, replication results, and exhaustive/search replay comparison.

Positions tested in Stage 4: the critical candidates, then `n_strat` (24) positions with up to 2 per (within-trace entropy quintile × type group) cell; when cells exceed `n_strat`, quotas shrink from the lowest quintile up so the top quintile keeps about 6; leftover slots go to positions within 64 tokens of `crit_cdpo` (uniform when none). `n_values_only` (16) further positions are drawn the same way and receive only `V_with`/`V_before` (free in exhaustive mode). `incl_prob` is exact within a cell; for a position that could be drawn by both the cell and the fill step it combines the two as independent draws.

The vLLM engine pins `logprobs_mode: raw_logprobs`, so `vllm_logprob_sampled` is taken before temperature and top-p. `engine_mismatch` therefore compares it with HF's p(y_i) at temperature 1 (with `processed_logprobs` it would use the generation temperature); comparing raw vLLM logprobs against HF at the generation temperature would flag nearly every position. The HF pass pins `attn_implementation: sdpa` in bf16; use the same setting when extracting features. Statistics from an older `token_stats` schema are recomputed automatically on the next `select` (`stats_version` in `select_meta.json`); the trace selection is kept, and Stage 1 fields missing from older runs are derived from the stored tokens and logprobs, never resampled.

Defaults not dictated by the task are: output directory `outputs/run`; tensor parallelism `1`; Stage 2 GPU count for accounting `1`; vLLM GPU-memory utilization `0.9`; request batch size `4096`; remote model code enabled; and the bundled cDPO JSONL selected unless `input_path` is supplied. The shared task defaults are collected in `ctlabel.config.DEFAULTS`. Startup prompt trailing whitespace is preserved deliberately, as required by the token-boundary warning in the task. Alternative replacement samples are drawn in the HF-only stage from the full vocabulary at `replacement_temperature`; this keeps Stage 4 vLLM-only without changing the cDPO distribution.

## Tests

```bash
pytest -q
pytest -q -m gpu
```

The regular suite covers production-path token-exact prefixes and combined-response grading, 200 rows through the pinned upstream grading loop, adaptive/exhaustive search including recovery and batched stage refinements, label logic, swap deduplication, and deterministic negative constraints. The Qwen GPU smoke test runs automatically when CUDA, torch, and vLLM are available; set `CTLABEL_SKIP_GPU_SMOKE=1` only when it must be disabled explicitly.

Vendored cDPO parser, grader, inputs, and license are in `ctlabel/vendor/`. Those files originate at upstream revision `228991e689529fed3139afc9b0374128d2029bc6`; local import paths were adjusted in the grader.


## Step 2: online critical-token signals (`ctfeat`)

`ctfeat` asks which signals, computable while the model generates, detect critical tokens (task: `critical_token_signals_TASK.md`). It reads the step 1 `dataset/` and never runs rollouts. Install the extra dependencies with `pip install -e '.[features]'`.

```bash
python -m ctfeat.extract  --dataset DATASET --out features/ [--traces N] [--layers every4]
python -m ctfeat.cost     --dataset DATASET --out results/            # per-family decoding latency (GPU)
python -m ctfeat.phi_replication --dataset DATASET --out features/    # optional model-level Φ_W on FineWeb
python -m ctfeat.evaluate --dataset DATASET --features features/ --out results/
sbatch scripts/signals.sh {smoke|extract|cost|phi|evaluate|all}      # Perlmutter wrappers
```

**Online rule.** Response token r sits at absolute index a = P + r. Every feature column is named `<feature>__<delay>`, with delay `pre` (indices ≤ a − 1 plus y_r), `post` (also index a), or `k4`/`k16`/`k64` (also indices ≤ a + k). Running statistics use only earlier response tokens (window 64; `warmup` marks fewer than 8). `tests/test_ctfeat_extract.py` recomputes features on sequences truncated after a + delay and requires identical values on a tiny float64 model; `tests/test_ctfeat_causality_gpu.py` (`scripts/signals.sh causality`) does the same for 10 real traces x 30 positions with the real weights in float32 (bf16 passes over different sequence lengths differ by up to ~1 logit from kernel rounding alone, so bf16 cannot test this to 1e-3).

**Extraction** runs one teacher-forced pass per trace with the pinned sdpa attention, so logits and hidden states are the ones step 1 used. Eager attention is not used: it rounds q·kᵀ to bf16 before the softmax, and with Qwen2.5-7B's attention scores (up to ~1e4) that changes the model's logits by up to ~17. Instead `ctfeat/attention.py` computes float32 softmax(q·kᵀ·scale + causal mask) from the same q and k alongside sdpa and hands only these weights to the hooks; the extraction check requires identical logits to sdpa and float32 weights·V to reproduce each layer's sdpa output. Per-layer hooks summarize attention rows, keep residuals only for response rows, and reduce block outputs to norms, so full attention maps never leave the GPU. Labels are never read: only `token_type` comes from `positions.parquet`, and `negatives.parquet` is never opened. Outputs:

- `tier1.parquet`, `tier2.parquet` (head-averaged attention per layer and received attention), `tier3_derived.parquet`: one row per (trace_id, pos).
- `attn/<trace>.npy` (query rows × layers × heads × stats), `attn_recv/<trace>.npy` (tokens × {4,16,64} × layers × heads), `hidden/<trace>.npy` (layers × query rows × d, float16). Query row j is absolute index P − 1 + j, so pre(r) = row r and post(r) = row r + 1.
- `manifest.json` (every feature's tier, delay, family, layer, head), `extract_meta.json` (feature-attention vs pinned-attention check, with eager recorded for reference; prediction-head checks against float32 logits, W_U basis orthonormality, disk estimate, versions), `mu_ref.pt` (Φ-Align reference means).

**Evaluation** groups every statistic by `problem_id`: 5-fold grouped CV and 1,000 problem-level bootstrap resamples, paired across models and used for the univariate screen and reported confidence intervals. Scalers, imputers, PCA, head selection and the classifier are fit only on the training fold (see `tests/test_ctfeat_eval.py::test_fold_hygiene`). Outputs go to `results/`: `univariate_{A,B}.csv`, `tier_models.csv`, `tier_gains.csv`, `probes_by_layer.csv`, `delay_curve.csv`, `breakdowns.csv`, `segments.csv`, `cost.csv`, `figures/` (heatmaps, probes, delay curve), `figures/aligned/` (event-aligned trajectories, between-label stretch, hidden-state drift), `meta.json` and `report.md`. The report ends with every default the task left open, including that `ll_top1_is_final_top1` is named `ll_top1_matches_final` (the leakage test forbids `is_` in feature names) and that ĥ, the final RMSNorm output including its weight, is the representation used for Φ-Align.

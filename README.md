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

- `traces_all.jsonl`: `trace_id`, `problem_id`, `dataset`, `model`, exact `prompt_ids` and `response_ids`, `n_labellable_tokens`, `stop_unaligned` (true when a stop string could not be aligned to token boundaries, so its tokens remain labellable), `response_text`, `finish_reason`, `stop_reason`, `pred`, `gold`, `correct`, `gen_temperature`, `seed`, and `vllm_top_logprobs`. `response_ids` preserves the exact vLLM output; only positions below `n_labellable_tokens` are analyzed.
- `traces.jsonl`: all `traces_all` columns for selected contested traces, plus `crit_cdpo`, `crit_drop`, `crit_good_candidate`, `non_monotone`, and `values_complete`. Exhaustive wrong-trace runs also store `crit_cdpo_search_replay`, `search_replay_positions`, and `exhaustive_positions`.
- `token_stats.parquet`: `trace_id`, `pos`, `token_id`, `token_text`, `token_type`, `is_first_subtoken`, `entropy`, `p_max`, `p_y`, `rank_y`, `top5_token_ids`, `top5_probabilities`, `replacement_token_ids`, `replacement_temperature`, `hf_vllm_argmax_agree`, and diagnostic `hf_vllm_top5_agree`.
- `values.parquet`: `trace_id`, `pos` (`-1` is prompt-only), `n_correct`, `n_total`, `rollout_temperature`, and `mode`.
- `swap.parquet`: `trace_id`, `pos`, `inclusion_probability`; `with_*`, `before_*`, `without_sampled_*`, `alt2_*`, and `alt3_*` IDs/counts; `replacement_token_ids`, `replacement_token_counts`, `replacement_temperature`, and `rollout_temperature`.
- `cost.jsonl`: actual generated-token and stage-runtime records.
- `rollouts_values.jsonl`, `rollouts_swap.jsonl`: per-request rollout counts appended after every vLLM batch, keyed by `(trace_id, pos, variant, seed, n, rollout_temperature)`. A stage killed at the wall-time limit skips these requests on restart.

Final files under `out_dir/dataset` are:

- `positions.parquet`: one row for every labelable selected response token. It includes every `token_stats` column; available `swap` columns; raw `with_*`/`before_*` counts; `v_with`, `v_before`, `drop`; Wilson `with_*`, `before_*`, `without_sampled_*`, `alt2_*`, and `alt3_*` confidence bounds; `drop_ci_*`, `fixable_effect`, `fixable_ci_*`; `v_without_sampled`, `v_alt2`, `v_alt3`; `correct_trace`, `model`, `zero_frac`, `entropy_decile`, `inclusion_probability`; and `is_crit_cdpo`, `is_crit_drop`, `is_fixable`, `is_crit_good`, `is_noncritical`, `is_consequence`, and `is_ambiguous`. Unevaluated positions remain present and ambiguous.
- `traces.jsonl`: exact tokenized traces and trace-level search results.
- `negatives.parquet`: `crit_trace_id`, `crit_pos`, `crit_label_kind`, `group`, `neg_trace_id`, and `neg_pos`.
- `meta.json`: `config`, `model_revision`, `versions`, `seeds`, `generated_tokens_by_stage`, `gpu_hours_by_stage`, and `generated_tokens_total`.
- `summary.md`: label/trace counts, entropy and type checks, lag, wrong-trace monotonicity, per-stage compute, replication results, and exhaustive/search replay comparison.

Defaults not dictated by the task are: output directory `outputs/run`; tensor parallelism `1`; Stage 2 GPU count for accounting `1`; vLLM GPU-memory utilization `0.9`; request batch size `4096`; remote model code enabled; and the bundled cDPO JSONL selected unless `input_path` is supplied. The shared task defaults are collected in `ctlabel.config.DEFAULTS`. Startup prompt trailing whitespace is preserved deliberately, as required by the token-boundary warning in the task. Alternative replacement samples are drawn in the HF-only stage from the full vocabulary at `replacement_temperature`; this keeps Stage 4 vLLM-only without changing the cDPO distribution.

## Tests

```bash
pytest -q
pytest -q -m gpu
```

The regular suite covers production-path token-exact prefixes and combined-response grading, 200 rows through the pinned upstream grading loop, adaptive/exhaustive search including recovery and batched stage refinements, label logic, swap deduplication, and deterministic negative constraints. The Qwen GPU smoke test runs automatically when CUDA, torch, and vLLM are available; set `CTLABEL_SKIP_GPU_SMOKE=1` only when it must be disabled explicitly.

Vendored cDPO parser, grader, inputs, and license are in `ctlabel/vendor/`. Those files originate at upstream revision `228991e689529fed3139afc9b0374128d2029bc6`; local import paths were adjusted in the grader.


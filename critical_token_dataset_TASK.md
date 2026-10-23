# Task: build a token-level critical-token dataset by adapting cDPO's rollout sampling

## 0. Context and scope

This is step 1 of a research project on **detecting critical tokens from LLM hidden representations**. A *critical token* is a token in a reasoning trace that decides whether the final answer ends up right or wrong.

Step 1 is to build a **labelled dataset** of:

- wrong and right reasoning traces;
- for every trace, token-level labels saying which tokens are critical and which are not, from counterfactual rollouts;
- matched non-critical *negative* tokens, so later analysis can check that a hidden-state pattern is specific to critical tokens.

The labelling method extends the rollout procedure from **"Critical Tokens Matter" (cDPO, ICML 2025)**, repo https://github.com/chenzhiling9954/Critical-Tokens-Matter.

**In scope:** trace generation, grading, prefix-value rollouts, the swap test, labels, negative sampling, exact token-level storage, tests, and a replication check against the paper.

**Out of scope (do not implement):** hidden-state statistics, probes, cDPO/DPO training, contrastive-estimation (CE) model training. Store everything needed so a later step can rerun an exact forward pass and pull hidden states.

---

## 1. Read the cDPO repo first

Clone it and read these files before writing code:

```
src/rollout_sampling_for_critical_tokens.py   # the labelling procedure we extend
src/utils/sampling.py                         # vLLM sampling wrapper
src/utils/utils.py                            # multi_result_check (grading loop)
src/utils/grader.py, src/utils/parser.py      # answer extraction + math_equal
src/configs/prompt.py                         # GSM8K_PROMPT, MATH_PROMPT (few-shot)
src/configs/data_config.py                    # temperatures, stop strings, model paths
src/input/{MATH,MATH_test,GSM8K,GSM8K_test}.jsonl   # fields: id, question, answer (list), ...
```

### How cDPO labels critical tokens (what the code does)

For each **wrong** trace with response tokens y_0 … y_{L-1}, two kinds of rollouts are run:

1. **"With token"** (`generate_with_and_without_for_sampling`, first half). For each position i, take the prefix that *includes* y_i and run 64 rollouts (`with_sampling_n = 64`). The score is the share of rollouts with the correct final answer.
2. **"Without token"** (second half). At each position, zero out the original token's probability in the model's next-token distribution (`probs[i-1, token_id] = 0`), sample 64 replacement tokens from what remains (`torch.multinomial`, unnormalized, temperature 1), and run **1** rollout from each (`without_sampling_n = 1`). This is a swap test: accuracy if the token is replaced by an alternative.

The criterion is implemented in `analyse_all_token`:

- walk the prefixes from short to long;
- the first prefix with 0/64 correct becomes the candidate;
- if a later prefix has more than 3/64 correct, discard the candidate and keep searching.

The critical token is the candidate that survives. The paper states this as "the first token with score 0 such that all later tokens score < 5%". `get_pass_at_k` then reports pass@k of the "without token" rollouts at the critical token. The paper reports pass@64 above 90%.

### Reuse

- `utils/grader.py`, `utils/parser.py`: reuse as-is for answer extraction and `math_equal`.
- The grading logic in `multi_result_check`: refactor into a pure function `grade(pred_str, gold_answers, dataset) -> (pred, correct)` with the same behaviour, including `split("\n\nQuestion:")[0].split("\n\nProblem:")[0]` and lower-casing for non-MATH datasets. Drop its file I/O.
- Few-shot prompts from `configs/prompt.py`, stop strings and per-model temperatures from `configs/data_config.py`.
- The input jsonl files.
- `get_pass_at_k` (copy it).

### Drop

`src/train/*`, the CE/cDPO parts of `pipeline.py`, `load_model` with offloading, and `save_merged_model`.

---

## 2. Problems in the original code that must NOT be carried over

1. **Prefixes are built as text, not tokens.** The code decodes `input_ids[:i]` to a string and later re-encodes `few_shot + question` with `.strip()` (`vllm_model_sampling`, lines 78–82). Re-tokenization can merge or split tokens at the boundary, and `.strip()` drops trailing whitespace tokens. So "prefix i" is not exactly the trace's first i tokens.
   **Fix:** work only with token IDs. Store `prompt_ids` and `response_ids` exactly as vLLM generated them, and submit rollouts with `vllm.inputs.TokensPrompt(prompt_token_ids=prompt_ids + response_ids[:i+1])`.
2. **Response start found by string matching** ("Solution:" / "Answer:"). Replace it with `len(prompt_ids)`.
3. **"With" and "without" results are matched by reconstructing strings** (`analyse_all_token`, around line 294, with an `assert`). Key every result by `(trace_id, position, variant)`.
4. **Ancient pins** (`vllm==0.2.6`, `transformers==4.42.4`). Use current vLLM and transformers. Create the engine with `enable_prefix_caching=True`; rollouts share long prefixes, and this is the main speedup.
5. **Only the first 100 rejected traces, wrong traces only.** Make the counts configurable and add correct traces (Section 4, Stage 3).
6. **Every position gets 64 + 64 rollouts.** That's fine for short GSM8K traces and unaffordable for long ones. Keep this as `mode: exhaustive` for replication, and add `mode: search` (Section 4, Stage 2).
7. **Replacement tokens are sampled at temperature 1 while rollouts use the config temperature.** Keep this behaviour in replication mode, record both temperatures in the output, and make both configurable.

---

## 3. Conventions (use these everywhere)

- Position i is the 0-based index into `response_ids`. EOS and stop tokens are not labelled.
- `V_with(i)` is the rollout accuracy from `prompt_ids + response_ids[:i+1]`, so the prefix **includes** y_i.
- `V_before(i) = V_with(i-1)`. For i = 0 it is the accuracy from the prompt alone. This is the accuracy when token i is freely resampled.
- `drop(i) = V_before(i) − V_with(i)`, the effect of committing to y_i.
- `V_without(i)` is the accuracy when y_i is replaced by alternatives:
  - **sampled variant:** cDPO-style, 64 alternatives × 1 rollout each;
  - **top-k variant:** CFT-style, the 2nd and 3rd most likely tokens, each × `n_alt_rollouts` (default 16).
- Every accuracy is stored as `(n_correct, n_total)`, never only as a ratio. Compute **Wilson 95% intervals** in the label stage.
- Each vLLM request gets an explicit seed derived from `(global_seed, trace_id, position, variant)`, so reruns are reproducible.

---

## 4. Pipeline

Create a package `ctlabel/` next to `src/` (or as a new repo that vendors the reused files under `ctlabel/vendor/` with the original license). One CLI entry point:

```
python -m ctlabel.run --config configs/<name>.yaml --stage {generate,select,values,swap,label,all}
```

Every stage reads the previous stage's outputs from `out_dir`, can resume (skip work already on disk), and appends what it generated to `out_dir/cost.jsonl`.

### Stage 1: `generate` (sample traces)

- Inputs: `dataset` (`MATH`, `MATH_test`, `GSM8K` or `GSM8K_test`); `model` (an HF path; default `Qwen/Qwen2.5-7B` base, since `meta-llama/Meta-Llama-3-8B` is gated); `n_samples_per_problem` (default 8); `gen_temperature`; `top_p = 0.95`; `max_tokens = 1024`; the stop strings from `data_config.py`.
- Prompts: few-shot prompts exactly as cDPO builds them in `prepare_startup_data`, i.e. `MATH_PROMPT + "Problem:\n" + q + "\n\nSolution:\n"`. Check the exact format in the repo and match it, then tokenize once and store `prompt_ids`.
- Request `logprobs=20` from vLLM and store the top-20 for each generated token.
- Grade every sample with `grade(...)`.
- Output: `traces_all.jsonl`, one line per sample, with:
  `trace_id, problem_id, dataset, model, prompt_ids, response_ids, response_text, finish_reason, pred, gold, correct, gen_temperature, seed`.
- Drop samples cut off at `max_tokens`, or whose prediction is empty.

### Stage 2: `select` (choose traces and compute exact per-token stats)

- **Contested problems:** 1 to 7 of 8 samples correct (configurable).
- **Pick traces:** up to `wrong_per_problem` (default 2) wrong and `right_per_problem` (default 1) right, until reaching `n_wrong` (default 200) and `n_right` (default 100). Use a fixed random seed.
- **Exact per-token stats.** Run one Hugging Face forward pass over `prompt_ids + response_ids` (bf16, no sampling) and save, for each response position:
  - exact entropy (full vocabulary, at temperature 1);
  - `p_max`, `p(y_i)`, rank of y_i;
  - the top-5 token IDs and their probabilities.

  Assert that `argmax`/`top-k` agree with vLLM's top-20 logprobs on at least 99% of positions; small numerical drift is expected.
- **Token type** for each position:
  - `digit`: contains a digit;
  - `operator`: one of `+ - * / = < > ^ ( )`, or a LaTeX relation or operator such as `\le`, `\ge`, `\times`, `\frac`;
  - `latex_other`: other LaTeX commands;
  - `connective`: so, thus, therefore, but, however, wait, then, since, because, if, or, instead, hence;
  - `punct`;
  - `other`.

  Label multi-token numbers on their first sub-token, and store `is_first_subtoken`.
- Output: `traces.jsonl` (the selected traces) and `token_stats.parquet` (one row per (trace_id, pos)).

### Stage 3: `values` (prefix-value curve)

Two modes.

**`mode: exhaustive`** (replication; short traces only). Compute `V_with(i)` for every position with `n_with_rollouts = 64`.

**`mode: search`** (default; cost grows with log L, not L). Assume V is roughly decreasing in a wrong trace, but verify it.

1. **Grid.** Compute V_with on positions `0, G, 2G, …, L−1` (default G = 32) and on the prompt alone.
2. **Point of no return (cDPO criterion).** Find the last grid point with `V > 3/64` (scaled to n), call it `a`, and the next grid point with `V == 0`, call it `b`. Binary-search in `(a, b]` for the first position with V_with = 0. Then check the criterion forwards: every later grid point must score ≤ 3/64, plus the 4 positions right after the candidate. If the check fails, restart the search after the failing point, as `analyse_all_token` does. Store the result as `crit_cdpo`, or null.
3. **Sharpest drop.** Take the grid interval with the largest `V(start) − V(end)`. Binary-search for where V crosses the interval's midpoint, then evaluate ±3 positions around the crossing and pick the largest single-step `drop(i)`. Store it as `crit_drop`.
4. **Monotonicity check.** If a grid interval shows a rise of more than 0.25, flag the trace (`non_monotone = true`) and also evaluate that interval's midpoint. Report the share of flagged traces.
5. **Right traces.** Compute the grid V. Find the largest single-step *rise* the same way and store it as `crit_good_candidate`.

Keep every evaluated position in `values.parquet`:
`trace_id, pos, n_correct, n_total, rollout_temperature, mode`.

Rollouts use `rollout_temperature` (replication default: the repo's per-model temperature; study default: 0.7), `top_p = 0.95`, `max_tokens = 1024`, the same stop strings, and `n` from config. Batch every request in a stage into a few large `llm.generate` calls.

### Stage 4: `swap` (counterfactual alternatives)

Positions to test in each trace:

- `crit_cdpo`, `crit_drop` and `crit_good_candidate`, when present;
- a **stratified sample** of `n_strat` (default 16) other positions:
  - 6 from the top entropy quintile of the trace;
  - 6 spread uniformly across the other four quintiles, which includes low-entropy tokens;
  - 4 within ±64 tokens of `crit_cdpo`, or at random if there is none.

  Store each position's inclusion probability so analyses can reweight.

At each tested position compute:

- `V_without_sampled`: cDPO style, 64 replacement tokens from the temperature-1 distribution with y_i zeroed, 1 rollout each. Store the replacement token IDs and their counts.
- `V_alt2` and `V_alt3`: force the 2nd and 3rd most likely tokens (excluding y_i), `n_alt_rollouts` (default 16) each.
- `V_with(i)` and `V_before(i)`, taken from Stage 3 when already computed, otherwise computed now with `n_with_rollouts`.

Output: `swap.parquet`.

### Stage 5: `label` (labels and negative groups)

Use Wilson intervals throughout. Thresholds live in config with these defaults: `tau_crit = 0.3`, `tau_null = 0.1`, `zero_frac = 3/64`.

Per position, store these label columns:

- `is_crit_cdpo` (wrong traces): equals `crit_cdpo`.
- `is_crit_drop`: `drop(i) ≥ tau_crit` and the interval on the drop excludes 0. For the interval on a difference, use a normal approximation on the two proportions, or a bootstrap.
- `is_fixable`: `max(V_without_sampled, V_alt2, V_alt3) − V_with(i) ≥ tau_crit`, with the interval excluding 0.
- `is_crit_good` (right traces): `V_with(i)` high (≥ 0.7) and both `V_alt2` and `V_alt3` ≤ 0.3 (the CFT-style definition); or a rise ≥ `tau_crit` at `crit_good_candidate`.
- `is_noncritical`: the drop interval lies inside `[−tau_null, tau_null]`, the position comes before any critical token, and `V_with(i) > zero_frac`.
- `is_consequence` (wrong traces): after `crit_cdpo`, with `V_with ≤ zero_frac`.
- `is_ambiguous`: none of the above. Excluded from analysis but kept in the files.

Then assign **negative groups** for each critical token, all from the same model and, where noted, the same trace:

| Group | Rule |
|---|---|
| `benign_fork` | non-critical, entropy within the same decile as the critical token (deciles computed over all labelled positions) |
| `same_type` | non-critical, same `token_type` |
| `nearby` | non-critical, same trace, within ±64 tokens |
| `consequence` | `is_consequence`, same trace |
| `correct_trace` | non-critical, in a right trace, same `token_type` and entropy decile |
| `random` | non-critical, uniform |

Sample up to `k_neg` (default 3) per group per critical token, without replacement, with a fixed seed. Save as `negatives.parquet`:
`crit_trace_id, crit_pos, crit_label_kind, group, neg_trace_id, neg_pos`.

Final outputs:

- `dataset/positions.parquet`: one row per labelled position, joining token stats, values, swap results, labels and inclusion probability.
- `dataset/traces.jsonl` and `dataset/negatives.parquet`.
- `dataset/meta.json`: config, model revision hash, vLLM / transformers / torch versions, seeds, and total generated tokens per stage.
- `dataset/summary.md`, an auto-generated report (Section 6).

---

## 5. Tests (all must pass)

Put these under `tests/`:

1. **Exact prefixes.** For random (trace, i), the token IDs submitted to vLLM equal `prompt_ids + response_ids[:i+1]`. Nothing is ever decoded and re-encoded. Use a mock engine that records requests.
2. **Grader parity.** `grade()` matches the original `multi_result_check` labels on 200 random lines of `input/MATH_test.jsonl` and `input/GSM8K_test.jsonl`, using the `response` field as the prediction.
3. **Search vs exhaustive on a synthetic V.** With a mock rollout function backed by a known V curve (step-down, gradual decline, and non-monotone cases), `search` returns the same `crit_cdpo` as an exhaustive scan run through a reimplementation of `analyse_all_token`'s loop. Also test the reset case: zero, then a recovery above 3/64, then zero again.
4. **Label logic** on hand-built tables (critical, non-critical, consequence, ambiguous, fixable).
5. **Negative sampling:** no overlap with critical positions, same-trace constraints hold, deterministic for a fixed seed.
6. **Smoke test:** end-to-end on 5 GSM8K problems with `Qwen/Qwen2.5-0.5B` and small rollout counts (n = 4, G = 16). It must finish, and every output file must load.

---

## 6. Replication and sanity checks (run, then report in `summary.md`)

### A. Replication of cDPO (exhaustive mode)

- Setting: `Qwen/Qwen2.5-7B` base (in the repo's config), or `meta-llama/Meta-Llama-3-8B` if access is available; GSM8K and MATH train; 100 wrong traces each; the repo's temperatures; 64 + 64 rollouts.
- Expected, from the paper:
  - a critical token is found in about 99–100% of wrong traces;
  - pass@64 of `V_without_sampled` at the critical token is above 90%.

  Report these numbers next to the paper's. If they differ by a lot, investigate before continuing; do not tune thresholds to match.

### B. Search vs exhaustive on the same 100 GSM8K traces

Report:

- exact agreement of `crit_cdpo` (target ≥ 90%) and the distribution of position differences;
- rollouts used per trace in each mode.

### C. Dataset summary (study run)

- counts by label;
- share of wrong traces with `crit_cdpo`, with `crit_drop`, and with both at the same position;
- how often `is_fixable` holds at `crit_cdpo` and at `crit_drop`;
- the entropy distribution of critical tokens vs non-critical tokens, and the **share of critical tokens in the bottom two entropy quintiles**;
- token types of critical tokens;
- the lag `crit_drop − crit_cdpo`;
- the share of non-monotone traces;
- total generated tokens and GPU-hours per stage.

---

## 7. Configs to provide

- `configs/smoke.yaml`: Qwen2.5-0.5B, 5 problems, tiny counts.
- `configs/replicate_qwen25_7b_gsm8k.yaml` and `configs/replicate_qwen25_7b_math.yaml`: exhaustive mode, repo temperatures, 100 wrong traces from the train split.
- `configs/study_qwen25_7b_math500.yaml`: `MATH_test`, search mode, `gen_temperature = 0.7`, `rollout_temperature = 0.7`, 200 wrong + 100 right traces, defaults as above.

---

## 8. Practical notes

- **Compute.** Log generated tokens per stage. Rough study-run budget: Stage 3 at about 15–25 evaluated positions × 64 rollouts per trace, plus Stage 4 at about 19 positions × (64 + 2 × 16) rollouts per trace. Print an estimate before launching and ask for confirmation (`--yes` to skip).
- **GPU memory.** Load the HF model (Stage 2) and the vLLM engine (Stages 1, 3, 4) in separate processes or stages, never both at once.
- **Formats.** Use parquet for position-level tables (pyarrow) and jsonl for traces. Store token IDs as int32 lists.
- **Code quality.** Type hints, small pure functions for the label logic, no global state. The search algorithm must be testable against a mock rollout function.
- **README.** Explain how to run each config, the meaning of every output column, and every default you chose that this document didn't fix.

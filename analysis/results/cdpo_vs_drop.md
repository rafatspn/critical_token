# cDPO critical token vs sharpest-drop token

Dataset: `outputs/study_qwen25_7b_math500_exhaustive/dataset`

Wrong traces with both labels: 200; different tokens: 166 (83%). `crit_drop` comes before `crit_cdpo` in 166 of them (median lag -38 tokens).

Replacement = 64 sampled alternatives (original token excluded) x 1 rollout. Resample = 64 rollouts from the prefix just before the token (V_before). Differences are drop − cDPO, paired per trace; 95% bootstrap CI; exact two-sided sign test on wins vs losses.

## Drop vs cDPO

| intervention | metric | at drop | at cDPO | diff [95% CI] | drop wins / cDPO wins / ties | p (sign) |
|---|---|---|---|---|---|---|
| replace | pass@1 | 0.509 | 0.223 | +0.286 [+0.234, +0.338] | 133 / 30 / 3 | 1.1e-16 |
| replace | pass@64 | 0.988 | 0.904 | +0.084 [+0.042, +0.133] | 15 / 1 / 150 | 0.00052 |
| resample | pass@1 | 0.480 | 0.073 | +0.407 [+0.374, +0.440] | 166 / 0 / 0 | 2.1e-50 |
| resample | pass@64 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 0 / 0 / 166 | 1 |

## Replacement vs free resampling at the same token

| token | metric | replace | resample | diff [95% CI] | replace wins / resample wins / ties | p (sign) |
|---|---|---|---|---|---|---|
| cdpo | pass@1 | 0.223 | 0.073 | +0.150 [+0.116, +0.187] | 109 / 40 / 17 | 1.4e-08 |
| cdpo | pass@64 | 0.904 | 1.000 | -0.096 [-0.145, -0.054] | 0 / 16 / 150 | 3.1e-05 |
| drop | pass@1 | 0.509 | 0.480 | +0.029 [+0.003, +0.056] | 72 / 84 / 10 | 0.38 |
| drop | pass@64 | 0.988 | 1.000 | -0.012 [-0.030, +0.000] | 0 / 2 / 164 | 0.5 |

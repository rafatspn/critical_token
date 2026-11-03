#!/bin/bash
# Step 2 (online critical-token signals) on one Perlmutter A100.
#
#   sbatch scripts/signals.sh smoke       # test 8: 5 traces, layers {4,28}, every output file
#   sbatch scripts/signals.sh causality   # test 1 on the real model (float32), 10 traces x 30 positions
#   sbatch scripts/signals.sh extract     # features for every trace (sdpa + float32 attention weights, one pass per trace)
#   sbatch scripts/signals.sh cost        # per-family decoding latency
#   sbatch scripts/signals.sh phi         # optional: model-level Φ_W on FineWeb (logged only)
#   sbatch scripts/signals.sh evaluate    # Section 4 (CPU-heavy: CV, bootstrap, plots)
#   sbatch scripts/signals.sh all         # extract, cost, phi, evaluate
#   sbatch scripts/signals.sh tests       # GPU tests 1 and 8 on the configured dataset
#
# Settings and paths come from SIGNALS_CONFIG (default: the MATH run's configs/signals_qwen25_7b_math500.yaml):
#   SIGNALS_CONFIG=configs/signals_qwen25_7b_gsm8k.yaml sbatch scripts/signals.sh all
# Paths can still be overridden: DATASET, FEATURES, RESULTS.

#SBATCH --account=dasrepo_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --job-name=ctfeat
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

STAGE=${1:-smoke}
REPO=${CTLABEL_REPO:-/global/homes/a/arafspn/crit_token}
VENV=${CTLABEL_VENV:-$SCRATCH/venvs/ctlabel}
module load python/3.12-26.1.0
set -euo pipefail
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export XDG_CACHE_HOME=$SCRATCH/cache
export TRITON_CACHE_DIR=$SCRATCH/cache/triton
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
source "$VENV/bin/activate"
cd "$REPO"

SIGNALS_CONFIG=${SIGNALS_CONFIG:-configs/signals_qwen25_7b_math500.yaml}
eval "$(python -m ctfeat.runconfig "$SIGNALS_CONFIG")"
DATASET=${DATASET:-$CFG_DATASET}
FEATURES=${FEATURES:-$CFG_FEATURES}
RESULTS=${RESULTS:-$CFG_RESULTS}
export CTFEAT_DATASET=$DATASET          # the GPU tests read the dataset from here
echo "job=${SLURM_JOB_ID:-local} stage=$STAGE config=$SIGNALS_CONFIG dataset=$DATASET features=$FEATURES results=$RESULTS"

case "$STAGE" in
    smoke)    python -m pytest -q -m gpu tests/test_ctfeat_smoke.py ;;
    causality) python -m pytest -q -s -m gpu tests/test_ctfeat_causality_gpu.py ;;
    tests)    # separate processes: a failed test's traceback would otherwise keep its float32 model on the GPU
        status=0
        python -m pytest -q -s -m gpu tests/test_ctfeat_causality_gpu.py || status=1
        python -m pytest -q -s -m gpu tests/test_ctfeat_smoke.py || status=1
        exit $status ;;
    extract)  python -m ctfeat.extract --dataset "$DATASET" --out "$FEATURES" "${EXTRACT_ARGS[@]}" ;;
    cost)     python -m ctfeat.cost --dataset "$DATASET" --out "$RESULTS" "${COST_ARGS[@]}" ;;
    phi)      python -m ctfeat.phi_replication --dataset "$DATASET" --out "$FEATURES" "${PHI_ARGS[@]}" || true ;;
    evaluate) python -m ctfeat.evaluate --dataset "$DATASET" --features "$FEATURES" --out "$RESULTS" "${EVAL_ARGS[@]}" ;;
    all)
        python -m ctfeat.extract --dataset "$DATASET" --out "$FEATURES" "${EXTRACT_ARGS[@]}"
        python -m ctfeat.cost --dataset "$DATASET" --out "$RESULTS" "${COST_ARGS[@]}"
        python -m ctfeat.phi_replication --dataset "$DATASET" --out "$FEATURES" "${PHI_ARGS[@]}" || true
        python -m ctfeat.evaluate --dataset "$DATASET" --features "$FEATURES" --out "$RESULTS" "${EVAL_ARGS[@]}"
        ;;
    *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac

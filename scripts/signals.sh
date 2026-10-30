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
#
# Paths can be overridden: DATASET, FEATURES, RESULTS.

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

DATASET=${DATASET:-outputs/study_qwen25_7b_math500_exhaustive/dataset}
FEATURES=${FEATURES:-outputs/signals/features}
RESULTS=${RESULTS:-outputs/signals/results}
echo "job=${SLURM_JOB_ID:-local} stage=$STAGE dataset=$DATASET features=$FEATURES results=$RESULTS"

case "$STAGE" in
    smoke)    python -m pytest -q -m gpu tests/test_ctfeat_smoke.py ;;
    causality) python -m pytest -q -s -m gpu tests/test_ctfeat_causality_gpu.py ;;
    extract)  python -m ctfeat.extract --dataset "$DATASET" --out "$FEATURES" --layers every4 ;;
    cost)     python -m ctfeat.cost --dataset "$DATASET" --out "$RESULTS" ;;
    phi)      python -m ctfeat.phi_replication --dataset "$DATASET" --out "$FEATURES" || true ;;
    evaluate) python -m ctfeat.evaluate --dataset "$DATASET" --features "$FEATURES" --out "$RESULTS" --n-jobs "${SLURM_CPUS_PER_TASK:-8}" ;;
    all)
        python -m ctfeat.extract --dataset "$DATASET" --out "$FEATURES" --layers every4
        python -m ctfeat.cost --dataset "$DATASET" --out "$RESULTS"
        python -m ctfeat.phi_replication --dataset "$DATASET" --out "$FEATURES" || true
        python -m ctfeat.evaluate --dataset "$DATASET" --features "$FEATURES" --out "$RESULTS" --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
        ;;
    *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac

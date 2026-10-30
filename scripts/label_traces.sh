#!/bin/bash
# Run the ctlabel pipeline on one Perlmutter A100 (shared GPU QOS, charged 1/4 node).
#
# To submit the whole workflow at once, use scripts/submit_all.sh instead.
#
# Usage (from anywhere):
#   sbatch scripts/label_traces.sh                                   # smoke test, all stages
#   sbatch scripts/label_traces.sh configs/study_qwen25_7b_math500.yaml
#   sbatch scripts/label_traces.sh configs/study_qwen25_7b_math500.yaml values   # one stage
#
# Override Slurm settings on the command line, e.g. a short smoke run in the debug queue:
#   sbatch -q debug -t 00:30:00 scripts/label_traces.sh
#
# The first job creates a virtualenv at $CTLABEL_VENV (default $SCRATCH/venvs/ctlabel) and
# installs ctlabel with vLLM into it (~10 GB, ~10 min); later jobs reuse it. $SCRATCH is
# used because home is near quota; NERSC purges scratch files unused for 8 weeks, in
# which case the next job reinstalls.

#SBATCH --account=dasrepo_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=ctlabel
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

CONFIG=${1:-configs/smoke.yaml}
STAGE=${2:-all}
REPO=${CTLABEL_REPO:-/global/homes/a/arafspn/crit_token}
VENV=${CTLABEL_VENV:-$SCRATCH/venvs/ctlabel}

# Load modules before `set -u`: the module scripts reference unset variables.
module load python/3.12-26.1.0
set -euo pipefail

# Model weights (~15 GB for Qwen2.5-7B) go to $SCRATCH, not the 40 GB home quota.
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# JIT/compile caches take file locks, and home on compute nodes does not support flock
# (FlashInfer failed with "OSError: [Errno 524]"). Keep every cache on $SCRATCH (Lustre).
# vLLM follows XDG_CACHE_HOME; FlashInfer and Triton need their own variables.
export XDG_CACHE_HOME=$SCRATCH/cache
export FLASHINFER_WORKSPACE_BASE=$SCRATCH/cache
export TRITON_CACHE_DIR=$SCRATCH/cache/triton
mkdir -p "$HF_HOME" "$(dirname "$VENV")" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

# One-time environment setup; the lock stops concurrent jobs from installing at once.
(
    flock 9
    if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import vllm, ctlabel" 2>/dev/null; then
        echo "Setting up $VENV"
        python -m venv "$VENV"
        "$VENV/bin/pip" install --upgrade pip
        "$VENV/bin/pip" install -e "$REPO[test]"
    fi
) 9>"$VENV.lock"
source "$VENV/bin/activate"

cd "$REPO"
# Configs write to outputs/<run>; keep that on $SCRATCH (traces with top-20 logprobs reach GBs).
if [ ! -e outputs ]; then
    mkdir -p "$SCRATCH/ctlabel_outputs"
    ln -s "$SCRATCH/ctlabel_outputs" outputs
fi
echo "job=$SLURM_JOB_ID node=$(hostname) config=$CONFIG stage=$STAGE"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"

# Inside a job array (submit_all.sh), tasks run one at a time and each resumes from disk:
#  - near the wall-time limit Slurm sends USR1 (--signal above); this task stops the
#    pipeline and exits 0 so the next array task continues the run;
#  - when the run finishes, or fails with a real error, the array's pending tasks are
#    cancelled so nothing reruns.
cancel_pending_tasks() {
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
        scancel --state=PENDING "$SLURM_ARRAY_JOB_ID" || true
    fi
}
on_time_limit() {
    echo "Approaching the wall-time limit; stopping so the next array task can resume."
    kill -TERM -- "-$pipeline" 2>/dev/null || true
    wait "$pipeline" 2>/dev/null || true
    exit 0
}
trap on_time_limit USR1

# --yes: batch jobs have no terminal for the cost-confirmation prompt. setsid gives the
# pipeline and its per-stage subprocesses their own process group, so they stop together.
# Running it in the background lets the USR1 trap fire immediately instead of at exit.
setsid python -m ctlabel.run --config "$CONFIG" --stage "$STAGE" --yes &
pipeline=$!
status=0
wait "$pipeline" || status=$?
cancel_pending_tasks
if [ "$status" -ne 0 ]; then
    echo "Pipeline failed with exit code $status; cancelled the remaining array tasks." >&2
fi
exit "$status"

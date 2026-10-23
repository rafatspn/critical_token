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
mkdir -p "$HF_HOME" "$(dirname "$VENV")"

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

# --yes: batch jobs have no terminal for the cost-confirmation prompt.
python -m ctlabel.run --config "$CONFIG" --stage "$STAGE" --yes

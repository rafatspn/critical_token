#!/bin/bash
# Submit the whole ctlabel workflow on Perlmutter with one command (run on a login node):
#
#   scripts/submit_all.sh             # submit everything
#   scripts/submit_all.sh --dry-run   # print the sbatch commands without submitting
#
# Job graph (each job runs scripts/label_traces.sh with one config):
#   1. smoke test (debug queue, 30 min); it also builds the Python environment once.
#   2. after the smoke test succeeds, one job array per run, in parallel:
#        replicate_qwen25_7b_gsm8k, replicate_qwen25_7b_math, study_qwen25_7b_math500
#      Each array has $CTLABEL_MAX_ATTEMPTS (default 4) tasks of up to 48 h that run one at
#      a time (%1). A task that reaches the wall-time limit stops cleanly and the next task
#      resumes from disk (rollouts are saved after every batch). When the run finishes, or
#      fails with a real error, its remaining tasks are cancelled.
#
# Logs: logs/<job-name>-<job-id>.out / .err in the repo.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JOB_SCRIPT="$REPO/scripts/label_traces.sh"
MAX_ATTEMPTS=${CTLABEL_MAX_ATTEMPTS:-4}
RUNS=(replicate_qwen25_7b_gsm8k replicate_qwen25_7b_math study_qwen25_7b_math500)
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

mkdir -p "$REPO/logs"
cd "$REPO"

# submit <job-name> <sbatch args...>: prints the new job id.
submit() {
    local name=$1
    shift
    local args=(--parsable -J "$name" -o "$REPO/logs/%x-%j.out" -e "$REPO/logs/%x-%j.err" "$@")
    if [ "$DRY_RUN" = 1 ]; then
        echo "sbatch ${args[*]}" >&2
        echo "<$name>"
        return
    fi
    sbatch "${args[@]}" | cut -d';' -f1
}

smoke=$(submit ctl-smoke -q debug -t 00:30:00 "$JOB_SCRIPT" configs/smoke.yaml)
echo "smoke test: $smoke"

all_ids=("$smoke")
for run in "${RUNS[@]}"; do
    id=$(submit "ctl-$run" --array="1-$MAX_ATTEMPTS%1" \
        --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
        "$JOB_SCRIPT" "configs/$run.yaml")
    echo "$run: array $id (up to $MAX_ATTEMPTS x 48 h)"
    all_ids+=("$id")
done

if [ "$DRY_RUN" = 0 ]; then
    echo
    echo "Monitor:    squeue --me"
    echo "Cancel all: scancel ${all_ids[*]}"
    echo "Results:    outputs/<run>/dataset/summary.md (outputs/ links to \$SCRATCH/ctlabel_outputs)"
fi

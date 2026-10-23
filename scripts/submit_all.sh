#!/bin/bash
# Submit the whole ctlabel workflow on Perlmutter with one command (run on a login node):
#
#   scripts/submit_all.sh             # submit everything
#   scripts/submit_all.sh --dry-run   # print the sbatch commands without submitting
#
# Job graph (each job runs scripts/label_traces.sh with one config):
#   1. smoke test (debug queue, 30 min); it also builds the Python environment once.
#   2. after the smoke test succeeds, in parallel:
#        replicate_qwen25_7b_gsm8k, replicate_qwen25_7b_math, study_qwen25_7b_math500
#   3. each run in step 2 gets $CTLABEL_RETRIES (default 3) retry jobs. A retry starts only
#      if every earlier attempt of that run failed (e.g. hit the 48 h limit); stages resume
#      from what is on disk (rollouts are saved after every batch), so a retry continues
#      where the previous attempt stopped.
#      Retries that are not needed are cancelled automatically by Slurm.
#
# Logs: logs/<job-name>-<job-id>.out / .err in the repo.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JOB_SCRIPT="$REPO/scripts/label_traces.sh"
RETRIES=${CTLABEL_RETRIES:-3}
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
    attempts=$(submit "ctl-$run" --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
        "$JOB_SCRIPT" "configs/$run.yaml")
    ids=("$attempts")
    for ((retry = 1; retry <= RETRIES; retry++)); do
        # afternotok on *all* earlier attempts: if any attempt succeeded the dependency can
        # never be met and --kill-on-invalid-dep cancels this retry.
        attempts=$(IFS=:; echo "${ids[*]}")
        id=$(submit "ctl-$run-retry$retry" --dependency="afternotok:$attempts" --kill-on-invalid-dep=yes \
            "$JOB_SCRIPT" "configs/$run.yaml")
        ids+=("$id")
    done
    echo "$run: ${ids[*]}"
    all_ids+=("${ids[@]}")
done

if [ "$DRY_RUN" = 0 ]; then
    echo
    echo "Monitor:    squeue --me"
    echo "Cancel all: scancel ${all_ids[*]}"
    echo "Results:    outputs/<run>/dataset/summary.md (outputs/ links to \$SCRATCH/ctlabel_outputs)"
fi

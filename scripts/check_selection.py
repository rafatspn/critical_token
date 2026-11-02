"""Gate after `select` (GSM8K task, Section 3): are there enough contested traces?

    python scripts/check_selection.py configs/study_qwen25_7b_gsm8k.yaml

Writes <out_dir>/selection_report.md with the number of contested problems, the wrong and
right traces available and selected, and the accuracy distribution over the samples of
each problem. Exits 1 if `select` could not fill n_wrong / n_right; no setting is changed.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctlabel.config import load_config  # noqa: E402
from ctlabel.io import read_jsonl  # noqa: E402


def main() -> None:
    config = load_config(sys.argv[1])
    out = REPO / config["out_dir"]
    samples = list(read_jsonl(out / "traces_all.jsonl"))
    selected = list(read_jsonl(out / "traces.jsonl"))
    n = int(config["n_samples_per_problem"])
    by_problem: dict[str, list[bool]] = defaultdict(list)
    for row in samples:
        by_problem[str(row["problem_id"])].append(bool(row["correct"]))
    correct = {problem: sum(values) for problem, values in by_problem.items()}
    low, high = int(config["min_correct"]), int(config["max_correct"])
    contested = [problem for problem, c in correct.items() if low <= c <= high]
    available_wrong = sum(min(n - correct[p], int(config["wrong_per_problem"])) for p in contested)
    available_right = sum(min(correct[p], int(config["right_per_problem"])) for p in contested)
    n_wrong = sum(not row["correct"] for row in selected)
    n_right = len(selected) - n_wrong
    histogram = Counter(correct.values())
    enough = n_wrong >= int(config["n_wrong"]) and n_right >= int(config["n_right"])
    lines = [
        f"# Selection check: {config['dataset']}", "",
        f"- Problems sampled: {len(by_problem)} ({n} samples each; {len(samples)} traces kept in traces_all.jsonl)",
        f"- Overall accuracy: {sum(r['correct'] for r in samples) / max(len(samples), 1):.3f}",
        f"- Contested problems ({low}–{high} of {n} correct): {len(contested)}",
        f"- Available from contested problems (at most {config['wrong_per_problem']} wrong / {config['right_per_problem']} right each): "
        f"{available_wrong} wrong, {available_right} right",
        f"- Selected: {n_wrong} wrong (target {config['n_wrong']}), {n_right} right (target {config['n_right']})",
        f"- Targets met: {'yes' if enough else 'NO'}", "",
        f"| correct of {n} | problems |", "|---|---|",
        *[f"| {k} | {histogram.get(k, 0)} |" for k in range(n + 1)], "",
    ]
    text = "\n".join(lines)
    (out / "selection_report.md").write_text(text)
    print(text)
    sys.exit(0 if enough else 1)


if __name__ == "__main__":
    main()

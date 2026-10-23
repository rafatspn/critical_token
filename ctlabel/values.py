from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable

from .io import append_jsonl, read_jsonl, read_parquet, write_jsonl, write_parquet
from .randomness import request_seed
from .sampling import create_engine, exact_prefix, run_rollouts
from .search import SearchResult, exhaustive_curve, search_curve
from .types import Count, RolloutRequest


def _trace_length(trace: dict[str, Any]) -> int:
    return int(trace.get("n_labellable_tokens", len(trace["response_ids"])))


def _with_request(trace: dict[str, Any], position: int, config: dict[str, Any]) -> RolloutRequest:
    return RolloutRequest(
        trace_id=trace["trace_id"], position=position, variant="with",
        prompt_token_ids=exact_prefix(trace["prompt_ids"], trace["response_ids"], position),
        gold_answers=trace["gold"], dataset=trace["dataset"], n=int(config["n_with_rollouts"]),
        seed=request_seed(config["seed"], trace["trace_id"], position, "with"),
        response_prefix_ids=list(trace["response_ids"][: position + 1]),
    )


def _value_row(trace_id: str, position: int, score: Count, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": trace_id, "pos": position, "n_correct": score.n_correct,
        "n_total": score.n_total, "rollout_temperature": float(config["rollout_temperature"]),
        "mode": config["mode"],
    }


def _analyse(trace: dict[str, Any], evaluate: Callable[[int], Count], config: dict[str, Any]) -> SearchResult:
    length = _trace_length(trace)
    if config["mode"] == "exhaustive":
        return exhaustive_curve(length, evaluate, float(config["zero_frac"]), right_trace=bool(trace["correct"]))
    return search_curve(
        length, evaluate, int(config["grid_stride"]), float(config["zero_frac"]),
        right_trace=bool(trace["correct"]),
    )


def run_values(config: dict[str, Any]) -> None:
    out = Path(config["out_dir"])
    traces = list(read_jsonl(out / "traces.jsonl"))
    old_rows = read_parquet(out / "values.parquet")
    stored = {(row["trace_id"], int(row["pos"])): Count(int(row["n_correct"]), int(row["n_total"])) for row in old_rows}
    if traces and all(trace.get("values_complete") for trace in traces):
        return
    engine = create_engine(config)
    generated_tokens = 0
    all_rows = {(row["trace_id"], int(row["pos"])): row for row in old_rows}
    unfinished = [trace for trace in traces if not trace.get("values_complete")]
    # Batch the complete non-adaptive frontier: every position in exhaustive
    # mode, or prompt + grid in search mode.
    initial_requests = []
    for trace in unfinished:
        length = _trace_length(trace)
        if config["mode"] == "exhaustive":
            positions = range(-1, length)
        else:
            positions = sorted({-1, *range(0, length, int(config["grid_stride"])), length - 1}) if length else [-1]
        for position in positions:
            if (trace["trace_id"], position) in stored:
                continue
            initial_requests.append(_with_request(trace, position, config))
    initial, tokens = run_rollouts(engine, initial_requests, config, cache_path=out / "rollouts_values.jsonl")
    generated_tokens += tokens
    for (trace_id, position, _), score in initial.items():
        stored[(trace_id, position)] = score
        all_rows[(trace_id, position)] = _value_row(trace_id, position, score, config)
    if all_rows:
        write_parquet(out / "values.parquet", sorted(all_rows.values(), key=lambda row: (row["trace_id"], row["pos"])))

    results: dict[str, SearchResult] = {}
    if config["mode"] == "exhaustive":
        for trace in unfinished:
            results[trace["trace_id"]] = _analyse(
                trace, lambda position, trace_id=trace["trace_id"]: stored[(trace_id, position)], config,
            )
    else:
        # Pure searches block on missing positions while this coordinator
        # drains all traces' requests into synchronized vLLM batches.
        work_queue: queue.Queue[tuple[RolloutRequest, Future[Count]]] = queue.Queue()
        # Set when the coordinator fails, so that blocked or about-to-block
        # workers exit and the executor shutdown below cannot hang.
        cancelled = threading.Event()

        def worker(trace: dict[str, Any]) -> SearchResult:
            def evaluate(position: int) -> Count:
                key = (trace["trace_id"], position)
                if key in stored:
                    return stored[key]
                if cancelled.is_set():
                    raise RuntimeError("values search cancelled after another failure")
                response: Future[Count] = Future()
                work_queue.put((_with_request(trace, position, config), response))
                while True:
                    try:
                        return response.result(timeout=0.1)
                    except FutureTimeoutError:
                        if cancelled.is_set():
                            raise RuntimeError("values search cancelled after another failure") from None
            return _analyse(trace, evaluate, config)

        def fail_pending(error: BaseException, batch: list[tuple[RolloutRequest, Future[Count]]]) -> None:
            cancelled.set()
            pending = list(batch)
            while True:
                try:
                    pending.append(work_queue.get_nowait())
                except queue.Empty:
                    break
            for _, response in pending:
                if not response.done():
                    response.set_exception(error)

        with ThreadPoolExecutor(max_workers=max(1, len(unfinished))) as executor:
            jobs = {trace["trace_id"]: executor.submit(worker, trace) for trace in unfinished}
            batch: list[tuple[RolloutRequest, Future[Count]]] = []
            try:
                while not all(job.done() for job in jobs.values()):
                    batch = []
                    try:
                        first = work_queue.get(timeout=0.1)
                    except queue.Empty:
                        for job in jobs.values():
                            if job.done() and job.exception() is not None:
                                raise job.exception()  # type: ignore[misc]
                        continue
                    batch = [first]
                    deadline = time.monotonic() + 0.05
                    while len(batch) < int(config["request_batch_size"]):
                        try:
                            batch.append(work_queue.get(timeout=max(0.0, deadline - time.monotonic())))
                        except queue.Empty:
                            break
                    requests = [item[0] for item in batch]
                    round_results, tokens = run_rollouts(engine, requests, config)
                    generated_tokens += tokens
                    for request, response in batch:
                        score = round_results[(request.trace_id, request.position, "with")]
                        stored[(request.trace_id, request.position)] = score
                        all_rows[(request.trace_id, request.position)] = _value_row(request.trace_id, request.position, score, config)
                        response.set_result(score)
                    write_parquet(out / "values.parquet", sorted(all_rows.values(), key=lambda row: (row["trace_id"], row["pos"])))
                results = {trace_id: job.result() for trace_id, job in jobs.items()}
            except BaseException as error:
                fail_pending(error, batch)
                raise

    for trace in traces:
        result = results.get(trace["trace_id"])
        if result is None:
            continue
        for position, score in result.evaluated.items():
            all_rows[(trace["trace_id"], position)] = _value_row(trace["trace_id"], position, score, config)
        trace.update({
            "crit_cdpo": result.crit_cdpo, "crit_drop": result.crit_drop,
            "crit_good_candidate": result.crit_good_candidate,
            "non_monotone": result.non_monotone, "values_complete": True,
        })
        if config["mode"] == "exhaustive" and not trace["correct"]:
            replay = search_curve(
                _trace_length(trace), lambda position: result.evaluated[position],
                int(config["grid_stride"]), float(config["zero_frac"]), right_trace=False,
            )
            trace["crit_cdpo_search_replay"] = replay.crit_cdpo
            trace["search_replay_positions"] = len(replay.evaluated)
            trace["exhaustive_positions"] = len(result.evaluated)
    write_parquet(out / "values.parquet", sorted(all_rows.values(), key=lambda row: (row["trace_id"], row["pos"])))
    write_jsonl(out / "traces.jsonl", traces)
    append_jsonl(out / "cost.jsonl", [{"stage": "values", "generated_tokens": generated_tokens}])


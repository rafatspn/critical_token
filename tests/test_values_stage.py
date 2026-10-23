from ctlabel.io import read_jsonl, write_jsonl
from ctlabel.types import Count
from ctlabel.values import run_values


def test_search_refinements_are_batched_and_stage_prefixes_are_exact(tmp_path, monkeypatch):
    traces = []
    for trace_index in range(12):
        traces.append({
            "trace_id": f"t{trace_index}", "correct": False, "dataset": "GSM8K",
            "prompt_ids": [1, 2], "response_ids": list(range(100, 228)),
            "n_labellable_tokens": 128, "gold": ["0"],
        })
    write_jsonl(tmp_path / "traces.jsonl", traces)
    calls = []

    def fake_rollouts(_engine, requests, _config, **_kwargs):
        requests = list(requests)
        calls.append(requests)
        result = {}
        for request in requests:
            expected_response = traces[int(request.trace_id[1:])]["response_ids"][: request.position + 1]
            assert request.prompt_token_ids == [1, 2, *expected_response]
            assert request.response_prefix_ids == expected_response
            correct = 64 if request.position < 63 else 0
            result[(request.trace_id, request.position, request.variant)] = Count(correct, 64)
        return result, 0

    monkeypatch.setattr("ctlabel.values.create_engine", lambda config: object())
    monkeypatch.setattr("ctlabel.values.run_rollouts", fake_rollouts)
    config = {
        "out_dir": str(tmp_path), "mode": "search", "grid_stride": 32,
        "zero_frac": 3 / 64, "n_with_rollouts": 64, "seed": 4,
        "rollout_temperature": 0.7, "request_batch_size": 4096,
    }
    run_values(config)
    completed = list(read_jsonl(tmp_path / "traces.jsonl"))
    assert all(trace["values_complete"] for trace in completed)
    # One grid batch plus a small number of synchronized adaptive rounds,
    # rather than one generate call per trace and refinement.
    assert len(calls) < 12
    assert max(len(batch) for batch in calls[1:]) > 1



def _run_with_timeout(config, timeout=20.0):
    import threading

    outcome = {}

    def target():
        try:
            run_values(config)
        except BaseException as error:  # noqa: BLE001 - surfaced to the test
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "run_values hung instead of propagating the failure"
    return outcome.get("error")


def _failing_setup(tmp_path, monkeypatch, fake_rollouts):
    traces = [{
        "trace_id": f"t{index}", "correct": False, "dataset": "GSM8K",
        "prompt_ids": [1], "response_ids": list(range(200)), "n_labellable_tokens": 200, "gold": ["0"],
    } for index in range(6)]
    write_jsonl(tmp_path / "traces.jsonl", traces)
    monkeypatch.setattr("ctlabel.values.create_engine", lambda config: object())
    monkeypatch.setattr("ctlabel.values.run_rollouts", fake_rollouts)
    # A batch size of 2 leaves other traces' requests queued when a round fails.
    return {
        "out_dir": str(tmp_path), "mode": "search", "grid_stride": 32,
        "zero_frac": 3 / 64, "n_with_rollouts": 64, "seed": 1,
        "rollout_temperature": 0.7, "request_batch_size": 2,
    }


def test_adaptive_round_failure_raises_instead_of_hanging(tmp_path, monkeypatch):
    calls = []

    def fake_rollouts(_engine, requests, _config, **_kwargs):
        calls.append(requests)
        if len(calls) == 2:
            raise RuntimeError("simulated vLLM failure")
        return {(r.trace_id, r.position, r.variant): Count(64 if r.position < 90 else 0, 64) for r in requests}, 0

    error = _run_with_timeout(_failing_setup(tmp_path, monkeypatch, fake_rollouts))
    assert isinstance(error, RuntimeError) and "simulated vLLM failure" in str(error)


def test_worker_failure_raises_instead_of_hanging(tmp_path, monkeypatch):
    def fake_rollouts(_engine, requests, _config, **_kwargs):
        return {(r.trace_id, r.position, r.variant): Count(64 if r.position < 90 else 0, 64) for r in requests}, 0

    config = _failing_setup(tmp_path, monkeypatch, fake_rollouts)
    original = __import__("ctlabel.values", fromlist=["_analyse"])._analyse

    def analyse(trace, evaluate, config):
        if trace["trace_id"] == "t0":
            evaluate(5)
            raise ValueError("simulated search bug")
        return original(trace, evaluate, config)

    monkeypatch.setattr("ctlabel.values._analyse", analyse)
    error = _run_with_timeout(config)
    assert isinstance(error, ValueError) and "simulated search bug" in str(error)

import json

import httpx
import respx

from triage.__main__ import main

# Config deliberately different from config/config.yaml and from every
# default (0.75, sim-fast/sim-precise, 10/3/0.5, 5000/30.0): if __main__
# hardcoded the real path or a factory value, this test would notice through a
# threshold or a model that does not match the mocked answers.
CONFIG_YAML = """
categories: [technical]
confidence_threshold: 0.7
models:
  fast: cheap-model-test
  precise: dear-model-test
http:
  timeout_s: 2
  retries: 2
  backoff_base_s: 0.01
  max_wait_s: 1
draft:
  engine: template
db:
  lock_wait_ms: 250
templates:
  technical:
    en: "Hi {name}, about your issue."
"""

STUDENT = {"student_id": "STU-1", "name": "Marta", "email": "m@e.com",
           "programme": "BSc", "enrolment_status": "active",
           "start_date": "2026-02-01", "progress_pct": 40, "tutor": "Elena",
           "language": "en"}


def _classification(confidence, cost):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(
            {"category": "technical", "urgency": 1, "confidence": confidence,
             "reason": "test reason"})}}],
        "usage": {"cost_eur": cost}})


def _prepare_env(monkeypatch, tmp_path, suffix=""):
    """Writes config/requests into tmp_path and sets the TRIAGE_* variables.

    `suffix` allows a second run, in the same test, with its own database and
    its own output directory: a genuinely new run, not a resume of the first.
    """
    config_path = tmp_path / f"config{suffix}.yaml"
    config_path.write_text(CONFIG_YAML, encoding="utf-8")

    requests = [
        {"id": "REQ-A", "received_at": "2026-09-01T09:00:00", "channel": "email",
         "student_id": "STU-1", "subject": "won't load", "message": "help please"},
        {"id": "REQ-B", "received_at": "2026-09-01T09:05:00", "channel": "email",
         "student_id": "STU-1", "subject": "another issue", "message": "still broken"},
    ]
    requests_path = tmp_path / f"requests{suffix}.json"
    requests_path.write_text(json.dumps(requests), encoding="utf-8")

    output_dir = tmp_path / f"output{suffix}"

    monkeypatch.setenv("TRIAGE_TOKEN", "tok-test")
    monkeypatch.setenv("TRIAGE_STUDENT_API_URL", "http://api-x")
    monkeypatch.setenv("TRIAGE_MODEL_URL", "http://mod-x")
    monkeypatch.setenv("TRIAGE_REQUESTS_PATH", str(requests_path))
    monkeypatch.setenv("TRIAGE_DB_PATH", str(tmp_path / f"t{suffix}.db"))
    monkeypatch.setenv("TRIAGE_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("TRIAGE_CONFIG_PATH", str(config_path))
    return output_dir


@respx.mock
def test_main_reports_cost_as_a_delta_and_never_hides_the_run(
        monkeypatch, tmp_path, capsys):
    """Covers, in one pass (Docker is not started): the full end-to-end
    wiring, that the printed cost is the DIFFERENCE between two /v1/stats
    snapshots (not the after snapshot, nor divided by "completed"), and that
    a failure taking the BEFORE snapshot neither aborts nor hides the run."""
    respx.get("http://api-x/health").mock(return_value=httpx.Response(200))
    respx.get("http://mod-x/health").mock(return_value=httpx.Response(200))
    respx.get("http://api-x/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))

    # REQ-A completes first time with high confidence; REQ-B stays below the
    # threshold on both tiers and is escalated: so "completed" (1) and
    # "processed_this_run" (2) are DIFFERENT, and dividing by the first
    # instead of the second can be detected.
    model_route = respx.post("http://mod-x/v1/chat/completions")
    model_route.side_effect = [
        _classification(0.95, 0.001),   # REQ-A, cheap-model-test: completes
        _classification(0.3, 0.002),    # REQ-B, cheap-model-test: low confidence
        _classification(0.3, 0.003),    # REQ-B, dear-model-test: still low, escalates
    ]

    stats_route = respx.get("http://mod-x/v1/stats")
    stats_route.side_effect = [
        httpx.Response(200, json={
            "total_calls": 5, "total_cost_eur": 0.010000, "by_model": {}}),
        httpx.Response(200, json={
            "total_calls": 9, "total_cost_eur": 0.034000, "by_model": {}}),
    ]

    output_dir = _prepare_env(monkeypatch, tmp_path)

    code = main(sleep=lambda s: None)
    out = capsys.readouterr().out

    assert code == 0
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "review_queue.csv").exists()

    # The delta (0.034 - 0.010 = 0.024) has to appear; neither the before
    # snapshot (0.010) nor the after one (0.034) alone will do as "the cost
    # of this run".
    assert "0.024000" in out
    assert "Model calls in this run            : 4" in out
    # denominator: processed_this_run (2), not completed (1). With completed
    # the line would say "/ 1 requests" and the result would be 24.0, not 12.0.
    assert "/ 2 requests × 1,000 = €12.0" in out
    assert "/ 1 requests" not in out

    # The summary itself goes to stdout.
    assert "RUN SUMMARY" in out
    assert "escalated             : 1" in out
    assert "completed             : 1" in out

    # --- Second invocation, in the same test: the BEFORE snapshot fails. The
    # run still has to reach the end, with no cost block, never aborting
    # silently.
    stats_route.side_effect = [httpx.ConnectError("model down")]
    output_dir_2 = _prepare_env(monkeypatch, tmp_path, suffix="-2")
    capsys.readouterr()  # discard everything captured so far

    code_2 = main(sleep=lambda s: None)
    out_2 = capsys.readouterr().out

    assert code_2 == 0
    assert "RUN SUMMARY" in out_2
    assert "COST OF THIS RUN" not in out_2
    assert (output_dir_2 / "results.csv").exists()

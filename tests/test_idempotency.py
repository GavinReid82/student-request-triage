"""The second test chosen from the suite.

Idempotency is the first non-functional requirement, and it is impossible to
check by eye in a demo: running the full pass twice over the same file (with
duplicates inside it too) must not duplicate anything, give a student a
second draft, pay the model again or count its cost twice.
"""
import json

import httpx
import pytest
import respx

from triage.store import Store
from triage.students import StudentClient
from triage.classifier import Classifier
from triage.config import Config, Env
from triage.schemas import record_hash
from triage.pipeline import process_all

# No value matches a default from triage/config.py or config/config.yaml: a
# hardcode of any of them in the pipeline would have to fail here.
TEMPLATES = {c: {"ca": "Hola {name}, sobre «{subject}»."}
             for c in ("content", "technical", "admin", "withdrawal")}
CFG = Config(
    categories=("content", "technical", "admin", "withdrawal"),
    confidence_threshold=0.62,
    fast_model="cheap-model",
    precise_model="dear-model",
    http_timeout_s=3.5,
    http_retries=2,
    http_backoff_base_s=0.25,
    draft_engine="template",
    templates=TEMPLATES,
    db_lock_wait_ms=1234,
    http_max_wait_s=7.0,
)
ENV = Env("tok", "http://api", "http://mod", "r.json", "t.db", "output")
MODEL_URL = "http://mod/v1/chat/completions"

STUDENT = {"student_id": "STU-1", "name": "Núria", "email": "n@e.com",
           "programme": "MSc", "enrolment_status": "active",
           "start_date": "2026-01-15", "progress_pct": 61, "tutor": "Fernando",
           "language": "ca"}

CLASSIFICATION = httpx.Response(200, json={
    "choices": [{"message": {"content": json.dumps(
        {"category": "technical", "urgency": 2, "confidence": 0.95,
         "reason": "the campus won't load"})}}],
    "usage": {"cost_eur": 0.001}})


def request(n):
    return {"id": f"REQ-2026-{n:04d}", "received_at": "2026-09-01T09:00:00",
            "channel": "email", "student_id": "STU-1", "subject": f"subject {n}",
            "message": "text"}


@respx.mock
def test_two_runs_over_the_same_file_duplicate_nothing_and_create_no_second_draft(tmp_path):
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    model = respx.post(MODEL_URL).mock(return_value=CLASSIFICATION)

    # Ten different requests, plus two exact duplicates inside the file itself
    # (ids 1 and 7 repeated), the way an accidental resend would arrive.
    records = [request(n) for n in range(1, 11)] + [request(1), request(7)]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    db = str(tmp_path / "t.db")

    def run():
        st = Store(db, CFG.db_lock_wait_ms)
        st.initialise()
        with httpx.Client() as c:
            sc = StudentClient(c, ENV, CFG, st, sleep=lambda s: None)
            cl = Classifier(c, ENV, CFG, st, sleep=lambda s: None)
            summary = process_all(str(path), st, sc, cl, CFG)
        bodies = {r["id"]: st.draft(r["id"])["body"] for r in st.rows()}
        st.close()
        return summary, bodies

    first, bodies_1 = run()
    calls_after_first = model.call_count
    second, bodies_2 = run()

    # Nothing is duplicated: the total of distinct known requests does not change.
    assert first["total"] == second["total"] == 10
    assert first["drafts"] == second["drafts"] == 10

    # Nobody gets a second draft: the stored body is identical, byte for
    # byte, between the two runs.
    assert bodies_1 == bodies_2

    # Everything finishes well, and the second run handles nothing new.
    assert second["completed"] == 10
    assert first["processed_this_run"] == 10
    assert second["processed_this_run"] == 0

    # The second run does not pay for the model again.
    assert model.call_count == calls_after_first

    # The measured cost is not counted twice.
    assert second["cost_eur"] == first["cost_eur"]

    # Duplicates inside the file count distinct facts (ids 1 and 7), not
    # appearances; in the second run the ten already-known ids reappear and
    # are all counted.
    assert first["duplicates_ignored"] == 2
    assert second["duplicates_ignored"] == 10


def test_save_draft_does_not_replace_an_existing_one(tmp_path):
    """The store guarantee all draft idempotency relies on: if the process
    insists after a crash, a second attempt to save cannot overwrite what the
    student already received."""
    st = Store(str(tmp_path / "t.db"), CFG.db_lock_wait_ms)
    st.initialise()
    st.register("REQ-X", "hash-x")
    assert st.save_draft("REQ-X", "ca", "first draft") is True
    assert st.save_draft("REQ-X", "ca", "second draft, different") is False
    assert st.draft("REQ-X")["body"] == "first draft"
    st.close()


@respx.mock
def test_resuming_an_already_classified_request_does_not_add_the_cost_again(tmp_path):
    """Simulates a crash right after paying for the classification and before
    the draft: on resuming with the full pass, the model is not called again
    and the cost already paid is not counted a second time."""
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    model = respx.post(MODEL_URL).mock(return_value=CLASSIFICATION)

    rec = request(99)
    path = tmp_path / "r.json"
    path.write_text(json.dumps([rec]), encoding="utf-8")
    db = str(tmp_path / "t.db")

    previous = Store(db, CFG.db_lock_wait_ms)
    previous.initialise()
    previous.register(rec["id"], record_hash(rec))
    previous.mark(rec["id"], "classified", student_id="STU-1", category="technical",
                  urgency=2, confidence=0.95,
                  classification_reason="already paid for before the crash",
                  model_used="cheap-model", cost_eur=0.0042)
    previous.close()

    st = Store(db, CFG.db_lock_wait_ms)
    st.initialise()
    with httpx.Client() as c:
        sc = StudentClient(c, ENV, CFG, st, sleep=lambda s: None)
        cl = Classifier(c, ENV, CFG, st, sleep=lambda s: None)
        summary = process_all(str(path), st, sc, cl, CFG)
    r = next(x for x in st.rows() if x["id"] == rec["id"])
    st.close()

    assert model.call_count == 0
    assert r["state"] == "completed"
    assert r["cost_eur"] == pytest.approx(0.0042)
    assert summary["cost_eur"] == pytest.approx(0.0042)

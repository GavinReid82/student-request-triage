import dataclasses
import json

import httpx
import pytest
import respx
from triage.store import TYPE_DRAFT_RESUMED, TYPE_CONFLICT, Store
from triage.students import StudentClient
from triage.classifier import Classifier
from triage.config import Config, Env
from triage.schemas import Classification, record_hash
from triage.pipeline import (
    RESULT_CONFLICT, RESULT_DUPLICATE, TYPE_EXTERNAL_FAILURE,
    TYPE_UNEXPECTED_FAILURE, TYPE_NO_TEMPLATE, process_all, process_one)

# No value matches the real one in config/config.yaml or a default
# (0.75, sim-fast, sim-precise, 10/3/0.5, 5000, 30.0): a hardcode of any of
# them in the pipeline has to fail a test.
# Templates only in Catalan and with {name}: a language fixed in the code, or
# a draft that does not use the real student record, shows in the stored body.
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

RAW = {"id": "REQ-2026-0001", "received_at": "2026-09-01T09:00:00", "channel": "email",
       "student_id": "STU-1", "subject": "won't load", "message": "help"}
RAW_2 = {**RAW, "id": "REQ-2026-0002", "subject": "something else"}


def classification(conf, cost=0.001):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(
            {"category": "technical", "urgency": 2, "confidence": conf,
             "reason": "the campus won't load"})}}],
        "usage": {"cost_eur": cost}})


def build(tmp_path, client, cfg=CFG):
    st = Store(str(tmp_path / "t.db"), cfg.db_lock_wait_ms)
    st.initialise()
    return (st,
            StudentClient(client, ENV, cfg, st, sleep=lambda s: None),
            Classifier(client, ENV, cfg, st, sleep=lambda s: None))


def input_file(tmp_path, records):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


def row(st, request_id):
    return next(r for r in st.rows() if r["id"] == request_id)


def types_of(st, request_id):
    return {i["type"] for i in st.incidents() if i["request_id"] == request_id}


@respx.mock
def test_happy_path_reaches_completed_with_a_draft_in_the_student_language(tmp_path):
    """Confidence 0.70: above the configured threshold (0.62) and below the
    real one (0.75). A 0.75 written into the code would escalate it."""
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    respx.post(MODEL_URL).mock(return_value=classification(0.70, cost=0.0031))
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        assert process_one(RAW, st, sc, cl, CFG) == "completed"
    r = row(st, RAW["id"])
    assert (r["category"], r["urgency"], r["confidence"], r["classification_reason"]) == (
        "technical", 2, 0.70, "the campus won't load")
    assert r["model_used"] == "cheap-model"
    assert r["cost_eur"] == pytest.approx(0.0031)
    draft = st.draft(RAW["id"])
    assert draft["language"] == "ca"
    assert "Núria" in draft["body"]
    assert st.count_drafts() == 1


@respx.mock
def test_low_confidence_on_both_tiers_escalates_with_no_draft(tmp_path):
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    respx.post(MODEL_URL).mock(return_value=classification(0.30))
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        assert process_one(RAW, st, sc, cl, CFG) == "escalated"
    r = row(st, RAW["id"])
    assert "0.3" in r["reason"] and "0.62" in r["reason"]
    # The classification is always persisted, even when it escalates.
    assert r["category"] == "technical"
    assert st.count_drafts() == 0


@respx.mock
def test_invalid_output_on_both_tiers_escalates_and_keeps_the_cost(tmp_path):
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    garbage = lambda cost: httpx.Response(200, json={
        "choices": [{"message": {"content": "no idea what to tell you"}}],
        "usage": {"cost_eur": cost}})
    respx.post(MODEL_URL).mock(side_effect=[garbage(0.002), garbage(0.005)])
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        assert process_one(RAW, st, sc, cl, CFG) == "escalated"
    r = row(st, RAW["id"])
    assert r["reason"].startswith("invalid_model_output")
    assert r["cost_eur"] == pytest.approx(0.007)
    assert st.count_drafts() == 0


@respx.mock
def test_no_template_for_the_category_escalates_with_no_draft(tmp_path):
    """The CATEGORY is missing, not the language: an except UnsupportedLanguage
    that was too narrow would not catch this case and the request would end
    up as failed."""
    cfg = dataclasses.replace(
        CFG, templates={k: v for k, v in TEMPLATES.items() if k != "technical"})
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    respx.post(MODEL_URL).mock(return_value=classification(0.95))
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c, cfg)
        assert process_one(RAW, st, sc, cl, cfg) == "escalated"
    assert row(st, RAW["id"])["reason"].startswith("no_template")
    assert TYPE_NO_TEMPLATE in types_of(st, RAW["id"])
    assert st.count_drafts() == 0


@respx.mock
def test_request_without_student_id_is_invalid_without_calling_anyone(tmp_path):
    student_route = respx.get("http://api/students/STU-1")
    model_route = respx.post(MODEL_URL)
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        raw = {k: v for k, v in RAW.items() if k != "student_id"}
        assert process_one(raw, st, sc, cl, CFG) == "invalid"
    assert "student_id" in row(st, RAW["id"])["reason"]
    assert student_route.call_count == 0
    assert model_route.call_count == 0


@respx.mock
def test_unknown_student_is_invalid_without_calling_the_model(tmp_path):
    respx.get("http://api/students/STU-9").mock(return_value=httpx.Response(404, json={}))
    model_route = respx.post(MODEL_URL)
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        assert process_one({**RAW, "student_id": "STU-9"}, st, sc, cl, CFG) == "invalid"
    assert row(st, RAW["id"])["reason"].startswith("student_not_found")
    assert model_route.call_count == 0


def test_a_record_that_is_not_an_object_is_invalid_without_breaking_the_run(tmp_path):
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        assert process_one(["not", "an", "object"], st, sc, cl, CFG) == "invalid"
    (r,) = st.rows()
    assert r["id"].startswith("no-id:")
    assert "object" in r["reason"]


@respx.mock
def test_same_id_with_different_content_is_a_conflict_and_leaves_the_work_alone(tmp_path):
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    respx.post(MODEL_URL).mock(return_value=classification(0.95))
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        process_one(RAW, st, sc, cl, CFG)
        before = row(st, RAW["id"])
        result = process_one({**RAW, "message": "different text"}, st, sc, cl, CFG)
    assert result == RESULT_CONFLICT
    after = row(st, RAW["id"])
    assert (after["state"], after["category"], after["reason"]) == (
        before["state"], before["category"], before["reason"])
    assert TYPE_CONFLICT in types_of(st, RAW["id"])
    assert st.summary()["conflicts"] == 1
    assert st.count_drafts() == 1


@respx.mock
def test_process_all_ignores_identical_duplicates_in_the_same_file(tmp_path):
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    model_route = respx.post(MODEL_URL).mock(return_value=classification(0.95))
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        summary = process_all(input_file(tmp_path, [RAW, RAW, RAW]), st, sc, cl, CFG)
        assert process_one(RAW, st, sc, cl, CFG) == RESULT_DUPLICATE
    # Counts distinct requests that reappear, not appearances.
    assert summary["total"] == 1
    assert summary["duplicates_ignored"] == 1
    assert model_route.call_count == 1
    assert st.count_drafts() == 1


@respx.mock
def test_api_down_leaves_failed_and_the_run_continues(tmp_path):
    respx.get("http://api/students/STU-9").mock(return_value=httpx.Response(503))
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    respx.post(MODEL_URL).mock(return_value=classification(0.95))
    down = {**RAW, "student_id": "STU-9"}
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        process_all(input_file(tmp_path, [down, RAW_2]), st, sc, cl, CFG)
    r = row(st, down["id"])
    assert r["state"] == "failed"
    assert r["reason"].startswith("external_failure")
    assert TYPE_EXTERNAL_FAILURE in types_of(st, down["id"])
    assert row(st, RAW_2["id"])["state"] == "completed"


class ExplodingClassifier:
    """Fails unexpectedly on one specific request and not on the others."""

    def __init__(self, exploding_id):
        self.exploding_id = exploding_id

    def classify(self, req):
        if req.id == self.exploding_id:
            raise RuntimeError("unexpected failure")
        return (Classification(category="technical", urgency=2, confidence=0.9,
                               reason="ok"), "cheap-model", 0.001)


@respx.mock
def test_an_unexpected_failure_leaves_failed_and_the_run_continues(tmp_path):
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    with httpx.Client() as c:
        st, sc, _ = build(tmp_path, c)
        cl = ExplodingClassifier(RAW["id"])
        process_all(input_file(tmp_path, [RAW, RAW_2]), st, sc, cl, CFG)
    r = row(st, RAW["id"])
    assert r["state"] == "failed"
    assert r["reason"].startswith("unexpected_failure")
    assert TYPE_UNEXPECTED_FAILURE in types_of(st, RAW["id"])
    assert row(st, RAW_2["id"])["state"] == "completed"


@respx.mock
def test_a_failed_request_is_not_retried_in_the_same_run_and_the_next_picks_it_up(tmp_path):
    student_route = respx.get("http://api/students/STU-1").mock(
        return_value=httpx.Response(503))
    respx.post(MODEL_URL).mock(return_value=classification(0.95))
    path = input_file(tmp_path, [RAW, RAW])
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        summary = process_all(path, st, sc, cl, CFG)
        assert row(st, RAW["id"])["state"] == "failed"
        # A single sequence of attempts: the second appearance does not repeat it.
        assert student_route.call_count == CFG.http_retries
        assert summary["duplicates_ignored"] == 1

        student_route.mock(return_value=httpx.Response(200, json=STUDENT))
        _, sc2, cl2 = build(tmp_path, c)
        process_all(path, st, sc2, cl2, CFG)
    assert row(st, RAW["id"])["state"] == "completed"
    assert st.count_drafts() == 1


@respx.mock
def test_a_second_run_over_the_same_file_repeats_nothing(tmp_path):
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    model_route = respx.post(MODEL_URL).mock(return_value=classification(0.95))
    # An invalid record with an id and a record that is not an object (a
    # no-id reference): running them again cannot create rows or reprocess
    # them either.
    invalid = {k: v for k, v in {**RAW, "id": "REQ-2026-0003"}.items()
               if k != "student_id"}
    path = input_file(tmp_path, [RAW, RAW_2, invalid, 42])
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c)
        first = process_all(path, st, sc, cl, CFG)
        calls = model_route.call_count
        _, sc2, cl2 = build(tmp_path, c)
        second = process_all(path, st, sc2, cl2, CFG)
    assert model_route.call_count == calls
    assert second["total"] == first["total"] == 4
    assert second["invalid"] == 2
    assert second["duplicates_ignored"] == 4
    assert second["drafts"] == first["drafts"] == 2
    assert second["cost_eur"] == first["cost_eur"]
    # First run: all 4 records arrive for the first time, so all 4 are
    # handled. Second run over the same file: all 4 are duplicates (same id,
    # same content) and none is handled again.
    assert first["processed_this_run"] == 4
    assert second["processed_this_run"] == 0


@pytest.mark.parametrize("previous_state,confidence,expected,drafts", [
    ("classified", 0.9, "completed", 1),
    ("drafted", 0.9, "completed", 1),
    ("failed", 0.9, "completed", 1),
    # A reused confidence below the threshold escalates too.
    ("classified", 0.4, "escalated", 0),
])
@respx.mock
def test_resuming_does_not_pay_for_the_classification_again(
        tmp_path, previous_state, confidence, expected, drafts):
    """A crash halfway leaves the classification already committed and paid
    for. On resume it is reused: no second model call, no cost added twice,
    no second draft."""
    # With the draft already saved it is not rewritten: even if the template
    # has disappeared from the configuration, the request does not escalate.
    cfg = dataclasses.replace(CFG, templates={}) if previous_state == "drafted" else CFG
    respx.get("http://api/students/STU-1").mock(return_value=httpx.Response(200, json=STUDENT))
    model_route = respx.post(MODEL_URL).mock(return_value=classification(0.95))
    with httpx.Client() as c:
        st, sc, cl = build(tmp_path, c, cfg)
        st.register(RAW["id"], record_hash(RAW))
        st.mark(RAW["id"], previous_state, student_id="STU-1", category="technical",
                urgency=2, confidence=confidence, classification_reason="earlier",
                model_used="cheap-model", cost_eur=0.0042,
                reason="external_failure: 503")
        if previous_state == "drafted":
            st.save_draft(RAW["id"], "ca", "earlier draft")
        assert process_one(RAW, st, sc, cl, cfg) == expected
    assert model_route.call_count == 0
    assert row(st, RAW["id"])["cost_eur"] == pytest.approx(0.0042)
    assert st.count_drafts() == drafts
    if expected == "completed":
        # Finished well: it does not carry the earlier failure's reason.
        assert row(st, RAW["id"])["reason"] is None
    # Resuming is not ignoring: it does not count as a duplicate.
    assert st.summary()["duplicates_ignored"] == 0
    if previous_state == "drafted":
        assert st.draft(RAW["id"])["body"] == "earlier draft"
        assert TYPE_DRAFT_RESUMED in types_of(st, RAW["id"])

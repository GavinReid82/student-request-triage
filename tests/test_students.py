import time

import httpx
import pytest
import respx
from triage.students import (
    TYPE_ALTERNATIVE_FIELD, StudentNotFound, StudentClient)
from triage.store import Store
from triage.config import Config, Env
from triage.http import HttpError

# Config has 11 fields and is built by keyword, never with the default value
# of a field the test exercises: hence 1234 and 7.0, which match neither
# DEFAULT_DB_LOCK_WAIT_MS (5000) nor DEFAULT_HTTP_MAX_WAIT_S (30.0).
CFG = Config(
    categories=("content",),
    confidence_threshold=0.75,
    fast_model="f",
    precise_model="p",
    # 3.5 on purpose: not the 1 a hardcoded timeout=1 at the call site
    # could be mistaken for.
    http_timeout_s=3.5,
    # 4 on purpose: 3 is the value a call site would most likely hardcode
    # (and also the real value in config/config.yaml), so a hardcoded
    # `retries=3` has to fail the test.
    http_retries=4,
    # 0.25 on purpose: not 0.0 (which erases the observable itself: the
    # calculated wait comes out 0.0 just like a backoff hardcoded to 0) and
    # not 0.5 (the real value in config/config.yaml, which an accidental
    # hardcode could copy without the test noticing).
    http_backoff_base_s=0.25,
    draft_engine="template",
    templates={},
    db_lock_wait_ms=1234,
    http_max_wait_s=7.0,
)
ENV = Env("tok", "http://api", "http://mod", "r.json", "t.db", "output")

ODD_RECORD = {
    "student_id": "STU-03871", "name": "Sergio", "email": "s@e.com",
    "programme": "MSc", "status_enrolment": "active",
    "start_date": "2026-02-01", "progress_pct": 55, "tutor": "Marta", "language": "en",
}


def store(tmp_path):
    # Store requires lock_wait_ms, with no default.
    s = Store(str(tmp_path / "t.db"), CFG.db_lock_wait_ms)
    s.initialise()
    return s


def _normal(student_id, **extra):
    record = {**ODD_RECORD, "student_id": student_id, "enrolment_status": "active", **extra}
    del record["status_enrolment"]
    return record


@respx.mock
def test_normalises_the_alternative_field_and_records_an_incident(tmp_path):
    """Nothing else checked the incident's real `type`, so renaming
    TYPE_ALTERNATIVE_FIELD in production went unnoticed, and that value is
    exactly what a human operator filters on to find these rows. This test
    also pins the `request_id` here (the alternative_field call site losing
    its `student:` prefix).
    """
    respx.get("http://api/students/STU-03871").mock(
        return_value=httpx.Response(200, json=ODD_RECORD))
    st = store(tmp_path)
    with httpx.Client() as c:
        student = StudentClient(c, ENV, CFG, st).get("STU-03871")
    assert student.enrolment_status == "active"
    assert st.summary()["incidents"] == 1
    incidents = st.incidents()
    assert len(incidents) == 1
    assert incidents[0]["type"] == "alternative_field"
    assert incidents[0]["request_id"] == "student:STU-03871"


@respx.mock
def test_the_cache_avoids_the_second_call(tmp_path):
    route = respx.get("http://api/students/STU-1").mock(
        return_value=httpx.Response(200, json=_normal("STU-1")))
    with httpx.Client() as c:
        client = StudentClient(c, ENV, CFG, store(tmp_path))
        client.get("STU-1")
        client.get("STU-1")
    assert route.call_count == 1
    assert client.calls == 1


@respx.mock
def test_404_becomes_student_not_found(tmp_path):
    respx.get("http://api/students/STU-09999").mock(
        return_value=httpx.Response(404, json={"detail": "not found"}))
    with httpx.Client() as c:
        with pytest.raises(StudentNotFound):
            StudentClient(c, ENV, CFG, store(tmp_path)).get("STU-09999")


# --- The cache also covers the confirmed miss (404) ---

@respx.mock
def test_404_is_cached_and_the_call_is_not_repeated(tmp_path):
    route = respx.get("http://api/students/STU-09998").mock(
        return_value=httpx.Response(404, json={"detail": "not found"}))
    with httpx.Client() as c:
        client = StudentClient(c, ENV, CFG, store(tmp_path))
        with pytest.raises(StudentNotFound):
            client.get("STU-09998")
        with pytest.raises(StudentNotFound):
            client.get("STU-09998")
    assert route.call_count == 1
    assert client.calls == 0


# --- This client's incidents are namespaced by student and dedupe across
# different instances sharing the same Store ---

@respx.mock
def test_student_incident_dedupes_across_different_clients(tmp_path):
    record = {**ODD_RECORD, "student_id": "STU-5"}
    respx.get("http://api/students/STU-5").mock(
        return_value=httpx.Response(200, json=record))
    st = store(tmp_path)
    with httpx.Client() as c:
        StudentClient(c, ENV, CFG, st).get("STU-5")
        StudentClient(c, ENV, CFG, st).get("STU-5")
    incidents = st.incidents()
    assert len(incidents) == 1
    assert incidents[0]["request_id"] == "student:STU-5"
    assert st.summary()["incidents"] == 1


# --- max_wait_s travels from config to send(); it is not hardcoded to 30.0
# (the default) and the server's Retry-After (3600s) is not used raw ---

@respx.mock
def test_max_wait_comes_from_config_not_hardcoded(tmp_path):
    # `send` binds its own `sleep=time.sleep` when it is defined, so patching
    # `time.sleep` afterwards does not intercept it: the seam StudentClient
    # exposes (`sleep=`) is used to record the wait without really sleeping.
    waits = []
    route = respx.get("http://api/students/STU-2")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "3600"}),
        httpx.Response(200, json=_normal("STU-2")),
    ]
    with httpx.Client() as c:
        client = StudentClient(c, ENV, CFG, store(tmp_path), sleep=waits.append)
        client.get("STU-2")
    assert waits == [7.0]


# --- Gaps closed by the following tests ---

@respx.mock
def test_retries_are_recorded_as_incidents(tmp_path):
    """A 503 followed by success must leave a record through `record=`.

    If `record=` is not passed to `send`, the retries still happen (the
    student is eventually fetched) but vanish from the store without a trace.
    """
    route = respx.get("http://api/students/STU-6")
    route.side_effect = [httpx.Response(503), httpx.Response(200, json=_normal("STU-6"))]
    st = store(tmp_path)
    with httpx.Client() as c:
        client = StudentClient(c, ENV, CFG, st, sleep=lambda s: None)
        client.get("STU-6")
    incidents = st.incidents()
    types = [i["type"] for i in incidents]
    assert types
    assert "http_retryable" in types
    # The `record=` call site has to namespace the same way as the
    # `alternative_field` one: if it went back to passing a bare student_id,
    # this assertion is the only one that notices.
    retry = next(i for i in incidents if i["type"] == "http_retryable")
    assert retry["request_id"] == "student:STU-6"
    # `calls` counts HTTP round trips started, not `send`'s internal
    # attempts. One `get()` started one request (which cost two attempts
    # inside): the counter has to say 1, not 2.
    assert client.calls == 1


@respx.mock
def test_normal_record_leaves_no_alternative_field_incident(tmp_path):
    """A record with the correct field name must not generate an incident."""
    respx.get("http://api/students/STU-7").mock(
        return_value=httpx.Response(200, json=_normal("STU-7")))
    st = store(tmp_path)
    with httpx.Client() as c:
        StudentClient(c, ENV, CFG, st).get("STU-7")
    assert st.summary()["incidents"] == 0


@respx.mock
def test_backoff_base_comes_from_config_not_hardcoded(tmp_path, monkeypatch):
    """The exponential backoff uses cfg.http_backoff_base_s (0.25), not a
    fixed value: neither the 0.0 that would erase the observable itself, nor
    the real 0.5 from config/config.yaml.

    `_wait` (triage/http.py) with no Retry-After calculates
    `base * 2**attempt + random.uniform(0, base)`. The jitter is neutralised
    (`random.uniform` pinned to 0) so the exact value can be pinned: with a
    single retry (`attempt=0`) the wait is `0.25 * 2**0 = 0.25`. A
    backoff_base_s hardcoded to 0.0 or to the YAML's 0.5 would produce 0.0 or
    0.5 respectively, and both fail this assertion.
    """
    monkeypatch.setattr("triage.http.random.uniform", lambda a, b: 0.0)
    waits = []
    route = respx.get("http://api/students/STU-8")
    route.side_effect = [httpx.Response(500), httpx.Response(200, json=_normal("STU-8"))]
    with httpx.Client() as c:
        client = StudentClient(c, ENV, CFG, store(tmp_path), sleep=waits.append)
        client.get("STU-8")
    assert waits == [0.25]


@respx.mock
def test_sends_the_authorization_header_with_the_token(tmp_path):
    """The student API requires the environment's token in Authorization: Bearer."""
    route = respx.get("http://api/students/STU-9").mock(
        return_value=httpx.Response(200, json=_normal("STU-9")))
    with httpx.Client() as c:
        StudentClient(c, ENV, CFG, store(tmp_path)).get("STU-9")
    assert route.calls.call_count == 1
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {ENV.token}"


@respx.mock
def test_timeout_comes_from_config_not_hardcoded(tmp_path):
    """The per-attempt timeout is cfg.http_timeout_s (3.5), not a fixed literal.

    httpx applies the request's `timeout=` to all four extensions
    (connect/read/write/pool); if the call site hardcoded `timeout_s=1` (the
    value CFG used to match by accident), this assertion tells them apart.
    """
    route = respx.get("http://api/students/STU-10").mock(
        return_value=httpx.Response(200, json=_normal("STU-10")))
    with httpx.Client() as c:
        StudentClient(c, ENV, CFG, store(tmp_path)).get("STU-10")
    assert route.calls.call_count == 1
    timeout = route.calls.last.request.extensions["timeout"]
    assert timeout["connect"] == CFG.http_timeout_s == 3.5


@respx.mock
def test_retries_come_from_config_not_hardcoded(tmp_path):
    """cfg.http_retries (4) governs how many attempts `send` makes, and it is
    pinned from both sides.

    Three 503s followed by a 200 have to succeed: that proves >= 4 attempts
    are allowed, and a `retries=3` hardcoded at the call site breaks it (the
    fourth attempt, the one bringing the 200, would never be made). Four 503s
    in a row have to exhaust the attempts and leave the respx route's call
    count at exactly 4 (not 3, not 5), which proves the ceiling is 4 and not
    just ">= 4".
    """
    success_route = respx.get("http://api/students/STU-11")
    success_route.side_effect = [
        httpx.Response(503), httpx.Response(503), httpx.Response(503),
        httpx.Response(200, json=_normal("STU-11")),
    ]
    exhausted_route = respx.get("http://api/students/STU-12")
    exhausted_route.side_effect = [httpx.Response(503)] * 4
    with httpx.Client() as c:
        client = StudentClient(
            c, ENV, CFG, store(tmp_path), sleep=lambda s: None)
        student = client.get("STU-11")
        assert student.student_id == "STU-11"
        assert success_route.calls.call_count == 4

        with pytest.raises(HttpError):
            client.get("STU-12")
    assert exhausted_route.calls.call_count == 4


# --- The hit cache is keyed by student_id, not by a fixed key ---

@respx.mock
def test_two_different_students_do_not_share_a_cache_entry(tmp_path):
    """Two different student_ids on the SAME client must each return their own record.

    If the cache were keyed by a constant (or by anything other than
    student_id), the second `get` would silently return the first record.
    """
    route_a = respx.get("http://api/students/STU-20").mock(
        return_value=httpx.Response(200, json=_normal("STU-20", name="Ana")))
    route_b = respx.get("http://api/students/STU-21").mock(
        return_value=httpx.Response(200, json=_normal("STU-21", name="Beatriz")))
    with httpx.Client() as c:
        client = StudentClient(c, ENV, CFG, store(tmp_path))
        a = client.get("STU-20")
        b = client.get("STU-21")
    assert a.student_id == "STU-20"
    assert a.name == "Ana"
    assert b.student_id == "STU-21"
    assert b.name == "Beatriz"
    assert route_a.call_count == 1
    assert route_b.call_count == 1
    assert client.calls == 2


# --- The cache of confirmed MISSES is also keyed by student_id, not by a
# fixed key: the previous test's gap in its negative twin ---

@respx.mock
def test_the_negative_cache_is_keyed_by_student_id(tmp_path):
    """Ask for a missing student and then a real one, on the SAME client.

    If `_missing` were read and written with a constant key (instead of
    student_id), the real student would silently inherit the missing one's
    404: StudentNotFound would fire without the request ever being made.
    """
    missing_route = respx.get("http://api/students/STU-MISSING").mock(
        return_value=httpx.Response(404, json={"detail": "not found"}))
    real_route = respx.get("http://api/students/STU-REAL").mock(
        return_value=httpx.Response(200, json=_normal("STU-REAL")))
    with httpx.Client() as c:
        client = StudentClient(c, ENV, CFG, store(tmp_path))
        with pytest.raises(StudentNotFound):
            client.get("STU-MISSING")
        student = client.get("STU-REAL")
    assert student.student_id == "STU-REAL"
    assert missing_route.calls.call_count == 1
    assert real_route.calls.call_count == 1


# --- The constructor's default `sleep` is the real time.sleep ---

def test_default_sleep_is_time_sleep(tmp_path):
    """Without passing `sleep=`, the client must be wired to the real time.sleep.

    A real sleep is not timed (that would be a flaky test): checking the
    identity of the stored attribute is enough. If the constructor's default
    were changed to a no-op, this is the only test in the whole suite that
    notices.
    """
    client = StudentClient(
        httpx.Client(), ENV, CFG, store(tmp_path))
    assert client._sleep is time.sleep


# --- The student-not-found error has to say WHICH student ---

@respx.mock
def test_student_not_found_names_the_student_in_the_message(tmp_path):
    """"A failure must never pass silently" includes not passing without
    saying who: a StudentNotFound with a generic message forces someone to
    reconstruct by hand which request caused it. The id goes in the message,
    and also in the second raise, the one that comes from the miss cache
    without touching the network."""
    respx.get("http://api/students/STU-30").mock(
        return_value=httpx.Response(404, json={"detail": "not found"}))
    with httpx.Client() as c:
        client = StudentClient(c, ENV, CFG, store(tmp_path))
        with pytest.raises(StudentNotFound) as first:
            client.get("STU-30")
        with pytest.raises(StudentNotFound) as second:
            client.get("STU-30")
    assert "STU-30" in str(first.value)
    assert "STU-30" in str(second.value)


# --- The alternative-field incident only applies if an alternative was used ---

@respx.mock
def test_record_with_both_field_names_leaves_no_incident(tmp_path):
    """If the record carries BOTH names, Student keeps the canonical one
    ('enrolment_status' comes first in AliasChoices), so no alternative name
    was used and recording the incident would state something false. The
    condition has two halves and this test pins the second."""
    record = {**ODD_RECORD, "student_id": "STU-31",
              "status_enrolment": "withdrawn", "enrolment_status": "active"}
    respx.get("http://api/students/STU-31").mock(
        return_value=httpx.Response(200, json=record))
    st = store(tmp_path)
    with httpx.Client() as c:
        student = StudentClient(c, ENV, CFG, st).get("STU-31")
    assert student.enrolment_status == "active"
    types = [i["type"] for i in st.incidents()]
    assert TYPE_ALTERNATIVE_FIELD not in types
    assert st.summary()["incidents"] == 0

import dataclasses
import json
import time

import httpx
import pytest
import respx
from triage.store import Store
from triage.classifier import (
    TYPE_LOW_CONFIDENCE, TYPE_LOW_CONFIDENCE_PERSISTS, TYPE_COST_NOT_REPORTED,
    TYPE_INVALID_OUTPUT, Classifier, ClassificationFailed, _content,
    build_prompt)
from triage.config import Config, Env
from triage.http import HttpError
from triage.schemas import Request
from triage.model_output import InvalidOutput

# Config has 11 fields and is built by keyword. No value matches its own
# default or the real one in config/config.yaml (0.75, sim-fast,
# sim-precise, backoff 0.5, max_wait 30, db 5000), so that a hardcode of any
# of them fails a test instead of passing them all by chance.
CFG = Config(
    categories=("content", "technical", "admin", "withdrawal"),
    confidence_threshold=0.62,      # NOT the yaml's 0.75
    fast_model="cheap-model",       # NOT "sim-fast"
    precise_model="dear-model",     # NOT "sim-precise"
    http_timeout_s=3.5,
    http_retries=4,
    http_backoff_base_s=0.25,       # NOT 0.0, NOT the yaml's 0.5
    draft_engine="template",
    templates={},
    db_lock_wait_ms=1234,           # NOT the default 5000
    http_max_wait_s=7.0,            # NOT the default 30.0
)
# Twin Config with a different, SMALLER category catalogue than the real one:
# it proves the enum `validate_output` checks against comes from config and
# not from a list of real categories written into production code.
CFG_OTHER_CATEGORIES = dataclasses.replace(CFG, categories=("x", "y"))
# Twin Config with timeout/retries/backoff DIFFERENT from CFG's (9.25, 6, 0.15
# versus 3.5, 4, 0.25) and from the real ones in config/config.yaml (10, 3,
# 0.5). Without it, a hardcode of any of the three TO THE VALUE CFG ALREADY
# HAS would go unnoticed: the observed behaviour would match the correct one.
CFG_OTHER_HTTP = dataclasses.replace(
    CFG, http_timeout_s=9.25, http_retries=6, http_backoff_base_s=0.15)

ENV = Env("tok", "http://api", "http://mod", "r.json", "t.db", "output")
URL = "http://mod/v1/chat/completions"

REQ = Request(id="REQ-2026-0001", received_at="2026-09-01T09:00:00", channel="email",
              student_id="STU-1", subject="can't get in", message="help")

OK_HIGH = '{"category":"technical","urgency":2,"confidence":0.99,"reason":"ok"}'
OK_LOW = '{"category":"technical","urgency":2,"confidence":0.10,"reason":"ok"}'
URGENCY_TEXT = '{"category":"content","urgency":"high","confidence":0.9,"reason":"ok"}'


def store(tmp_path):
    # Store requires lock_wait_ms, with no default.
    s = Store(str(tmp_path / "t.db"), CFG.db_lock_wait_ms)
    s.initialise()
    return s


def response(content, cost=0.001):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}],
        "usage": {"cost_eur": cost}})


def response_without_usage(content):
    """A valid 200 but with no `usage` block: the cost is not reported."""
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def response_without_choices(cost=0.001):
    """A 200 with a corrupted envelope: `choices` is empty."""
    return httpx.Response(200, json={"choices": [], "usage": {"cost_eur": cost}})


def response_null_content(cost=0.001):
    """A 200 with a well-formed envelope but `content` is not text."""
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None}}], "usage": {"cost_eur": cost}})


def response_non_finite_cost(content, json_literal):
    """A 200 whose `usage.cost_eur` is a non-finite JSON literal (NaN or
    Infinity). httpx serialises with `json.dumps(allow_nan=False)` under
    `json=`, so these two values have to be sent as raw bytes; `json.loads`
    (what `.json()` uses on the other side) does accept them by default,
    which is exactly the case _cost has to guard against."""
    body = (
        '{"choices":[{"message":{"content":' + json.dumps(content) + '}}],'
        '"usage":{"cost_eur":' + json_literal + '}}'
    )
    return httpx.Response(
        200, content=body.encode("utf-8"),
        headers={"content-type": "application/json"})


def body(route, index):
    return json.loads(route.calls[index].request.content)


def test_the_prompt_carries_the_subject_and_the_id():
    messages = build_prompt(REQ, CFG.categories)
    text = " ".join(m["content"] for m in messages)
    assert "can't get in" in text
    assert "REQ-2026-0001" in text
    # The mock service here concatenates content without looking at the role,
    # but a mislabelled `role` would be invisible to it and fatal against a
    # real OpenAI-compatible API: a system prompt downgraded to user loses its
    # force. Content alone is not enough; the role has to be pinned on the
    # specific message where it occurs.
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@respx.mock
def test_high_confidence_does_not_retry(tmp_path):
    # 0.70 sits BETWEEN 0.62 (CFG's real threshold) and 0.75 (the yaml's):
    # with a threshold hardcoded to 0.75 this confidence would retry
    # needlessly, and `route.call_count == 1` catches it.
    route = respx.post(URL).mock(return_value=response(
        '{"category":"technical","urgency":3,"confidence":0.70,"reason":"ok"}'))
    with httpx.Client() as c:
        classif, model, cost = Classifier(c, ENV, CFG, store(tmp_path)).classify(REQ)
    assert route.call_count == 1
    assert model == "cheap-model"
    assert classif.confidence == 0.70
    assert cost == 0.001


@respx.mock
def test_confidence_exactly_at_the_threshold_does_not_retry(tmp_path):
    """`>=` and not `>`: confidence EQUAL to the threshold is accepted without retrying."""
    route = respx.post(URL).mock(return_value=response(
        '{"category":"technical","urgency":2,"confidence":0.62,"reason":"ok"}'))
    with httpx.Client() as c:
        classif, model, _ = Classifier(c, ENV, CFG, store(tmp_path)).classify(REQ)
    assert route.call_count == 1
    assert model == "cheap-model"
    assert classif.confidence == 0.62


@respx.mock
def test_low_confidence_retries_with_the_precise_model(tmp_path):
    # 0.1 + 0.2 in binary is 0.30000000000000004, not 0.3. With these two
    # costs (one per call on the escalation path) the test only passes if
    # `classify` really rounds the accumulated cost; with 0.001 + 0.001 the
    # sum is already exact in binary and a deleted `round()` would go
    # unnoticed.
    route = respx.post(URL)
    route.side_effect = [
        response('{"category":"technical","urgency":3,"confidence":0.50,"reason":"ok"}',
                 cost=0.1),
        response('{"category":"technical","urgency":3,"confidence":0.95,"reason":"ok"}',
                 cost=0.2),
    ]
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, cost = Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 2
    assert model == "dear-model"
    assert classif.confidence == 0.95
    assert cost == 0.3
    types = [i["type"] for i in st.incidents()]
    assert TYPE_LOW_CONFIDENCE in types


@respx.mock
def test_invalid_output_retries_with_the_precise_model_and_then_gives_up(tmp_path):
    # Twin: the same argument as in the escalation test, applied to the
    # give-up path. 0.1 + 0.2 is not exactly 0.3 in binary, so this test only
    # passes if ClassificationFailed.cost_eur is also rounded with the same
    # constant.
    route = respx.post(URL)
    route.side_effect = [
        response(URGENCY_TEXT, cost=0.1),
        response(URGENCY_TEXT, cost=0.2),
    ]
    st = store(tmp_path)
    with httpx.Client() as c:
        with pytest.raises(InvalidOutput) as info:
            Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 2
    incidents = st.incidents()
    assert st.summary()["incidents"] >= 1
    assert incidents
    assert all(i["type"] == TYPE_INVALID_OUTPUT for i in incidents)
    # The cost of BOTH calls is not lost when giving up.
    assert isinstance(info.value, ClassificationFailed)
    assert info.value.cost_eur == 0.3
    assert info.value.cost_eur > 0


@respx.mock
def test_low_confidence_on_both_models_returns_the_precise_one(tmp_path):
    route = respx.post(URL)
    route.side_effect = [
        response('{"category":"content","urgency":1,"confidence":0.09,"reason":"ok"}'),
        response('{"category":"content","urgency":1,"confidence":0.24,"reason":"ok"}'),
    ]
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, _ = Classifier(c, ENV, CFG, st).classify(REQ)
    assert model == "dear-model"
    assert classif.confidence == 0.24
    # There is no third tier: the confidence that stays low after the precise
    # model is returned as it is (the decision passes to a person), but it is
    # recorded explicitly instead of vanishing silently.
    types = [i["type"] for i in st.incidents()]
    assert TYPE_LOW_CONFIDENCE_PERSISTS in types


@respx.mock
def test_category_outside_the_config_is_rejected_even_if_valid_in_general(tmp_path):
    """Proves the category enum comes from `cfg.categories`, not from a list
    of real categories written into production: "technical" is a real
    category, but CFG_OTHER_CATEGORIES does not include it."""
    route = respx.post(URL).mock(return_value=response(OK_HIGH))
    with httpx.Client() as c:
        with pytest.raises(InvalidOutput) as info:
            Classifier(c, ENV, CFG_OTHER_CATEGORIES, store(tmp_path)).classify(REQ)
    assert route.call_count == 2
    assert "technical" in str(info.value)


@respx.mock
def test_missing_cost_records_an_incident_and_is_not_lost_silently(tmp_path):
    route = respx.post(URL).mock(return_value=response_without_usage(OK_HIGH))
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, cost = Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 1
    assert cost == 0.0
    incidents = st.incidents()
    assert incidents
    assert any(i["type"] == TYPE_COST_NOT_REPORTED for i in incidents)
    detail = next(i["detail"] for i in incidents if i["type"] == TYPE_COST_NOT_REPORTED)
    assert "cheap-model" in detail


@respx.mock
def test_envelope_without_choices_escalates_to_the_precise_model(tmp_path):
    route = respx.post(URL)
    route.side_effect = [response_without_choices(), response(OK_HIGH)]
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, _ = Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 2
    assert model == "dear-model"
    types = [i["type"] for i in st.incidents()]
    assert TYPE_INVALID_OUTPUT in types


@respx.mock
def test_null_content_escalates_to_the_precise_model(tmp_path):
    route = respx.post(URL)
    route.side_effect = [response_null_content(), response(OK_HIGH)]
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, _ = Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 2
    assert model == "dear-model"
    types = [i["type"] for i in st.incidents()]
    assert TYPE_INVALID_OUTPUT in types


@respx.mock
def test_corrupted_envelope_from_the_precise_model_also_gives_up_without_crashing(tmp_path):
    """The twin rule: extracting the content has to be treated the same way
    on BOTH calls. If the second attempt does not wrap `_content` in the same
    try/except, a corrupted envelope from the precise model escapes as an
    uncaught IndexError instead of as ClassificationFailed."""
    route = respx.post(URL)
    route.side_effect = [
        response(OK_LOW, cost=0.002),
        response_without_choices(cost=0.004),
    ]
    st = store(tmp_path)
    with httpx.Client() as c:
        with pytest.raises(InvalidOutput) as info:
            Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 2
    assert isinstance(info.value, ClassificationFailed)
    assert info.value.cost_eur == pytest.approx(0.006)
    assert info.value.cost_eur > 0


@respx.mock
def test_request_body_carries_the_right_model_and_json_object_on_both_calls(
        tmp_path):
    """The model service picks the model from `body["model"]` and only
    returns JSON if `response_format` asks for it; both are surfaces for
    silent degradation and have to be pinned on BOTH calls."""
    route = respx.post(URL)
    route.side_effect = [response(OK_LOW), response(OK_HIGH)]
    with httpx.Client() as c:
        Classifier(c, ENV, CFG, store(tmp_path)).classify(REQ)
    assert route.call_count == 2
    first, second = body(route, 0), body(route, 1)
    assert first["model"] == "cheap-model" == CFG.fast_model
    assert second["model"] == "dear-model" == CFG.precise_model
    formats = [first.get("response_format"), second.get("response_format")]
    assert formats
    assert all(f == {"type": "json_object"} for f in formats)


@respx.mock
def test_authorization_header_on_both_calls(tmp_path):
    route = respx.post(URL)
    route.side_effect = [response(OK_LOW), response(OK_HIGH)]
    with httpx.Client() as c:
        Classifier(c, ENV, CFG, store(tmp_path)).classify(REQ)
    assert route.call_count == 2
    headers = [call.request.headers["Authorization"] for call in route.calls]
    assert headers
    assert all(h == f"Bearer {ENV.token}" == "Bearer tok" for h in headers)


class _StoreThatFailsOnOneType:
    """A duck-typed Store whose `incident()` blows up for ONE specific type
    and delegates to a real Store for the rest. It shows that a failure
    unrelated to validation (a broken incident write, for example) cannot be
    mistaken for invalid model output."""

    def __init__(self, real, failing_type):
        self._real = real
        self._failing_type = failing_type

    def incident(self, request_id, kind, detail):
        if kind == self._failing_type:
            raise RuntimeError("simulated failure writing the incident")
        self._real.incident(request_id, kind, detail)


@respx.mock
def test_an_unrelated_failure_on_the_fast_model_is_not_swallowed_as_invalid_output(tmp_path):
    """The first tier's except must catch ONLY InvalidOutput. If it caught
    `Exception` in general, a real bug (here: writing the low-confidence
    incident fails) would be mistaken for invalid model output, swallowed
    silently, and the request would escalate to the precise model instead of
    the real failure surfacing. With the correct except, the exception comes
    out immediately and the precise model is never called."""
    route = respx.post(URL)
    route.side_effect = [response(OK_LOW), response(OK_HIGH)]
    failing_store = _StoreThatFailsOnOneType(store(tmp_path), TYPE_LOW_CONFIDENCE)
    with httpx.Client() as c:
        with pytest.raises(RuntimeError):
            Classifier(c, ENV, CFG, failing_store).classify(REQ)
    assert route.call_count == 1


@respx.mock
def test_an_unrelated_failure_on_the_precise_model_is_not_wrapped_in_classification_failed(
        tmp_path):
    """The previous test's twin for the second tier: if the precise model's
    except also caught `Exception`, an unrelated failure (here: writing the
    low-confidence-persists incident) would be wrapped in ClassificationFailed
    as if it were invalid model output, instead of propagating as it is."""
    route = respx.post(URL)
    route.side_effect = [response(OK_LOW), response(OK_LOW)]
    failing_store = _StoreThatFailsOnOneType(store(tmp_path), TYPE_LOW_CONFIDENCE_PERSISTS)
    with httpx.Client() as c:
        with pytest.raises(RuntimeError):
            Classifier(c, ENV, CFG, failing_store).classify(REQ)
    assert route.call_count == 2


def test_default_sleep_is_time_sleep(tmp_path):
    """Twin: without passing `sleep=`, the classifier must be wired to the
    real time.sleep, just like StudentClient."""
    classifier = Classifier(httpx.Client(), ENV, CFG, store(tmp_path))
    assert classifier._sleep is time.sleep


@respx.mock
def test_sleep_is_forwarded_to_send_and_captures_the_backoff_from_config(tmp_path, monkeypatch):
    """`sleep=self._sleep` has to reach `send`. A 500 followed by success
    forces a retry INSIDE the first call; with the jitter neutralised, the
    calculated wait is exactly `cfg.http_backoff_base_s * 2**0 = 0.25`, not
    the 0.0 that would erase the observable itself nor the real 0.5 from
    config/config.yaml."""
    monkeypatch.setattr("triage.http.random.uniform", lambda a, b: 0.0)
    waits = []
    route = respx.post(URL)
    route.side_effect = [httpx.Response(500), response(OK_HIGH)]
    st = store(tmp_path)
    with httpx.Client() as c:
        classifier = Classifier(c, ENV, CFG, st, sleep=waits.append)
        classifier.classify(REQ)
    assert waits == [0.25]
    # The `record=` passed to `send` also has to namespace with req.id: if it
    # used another field of the request (e.g. student_id), the HTTP retry
    # incident would be filed under the wrong id.
    retries = [i for i in st.incidents() if i["type"] == "http_retryable"]
    assert retries
    assert all(i["request_id"] == "REQ-2026-0001" for i in retries)


@respx.mock
def test_incidents_carry_the_request_id_without_a_prefix(tmp_path):
    """Unlike the student client (which namespaces with 'student:'), here the
    incident IS about a request, so `req.id` goes as it is into the
    `request_id` column."""
    route = respx.post(URL)
    route.side_effect = [response(OK_LOW), response(OK_HIGH)]
    st = store(tmp_path)
    with httpx.Client() as c:
        Classifier(c, ENV, CFG, st).classify(REQ)
    incidents = st.incidents()
    assert incidents
    assert all(i["request_id"] == "REQ-2026-0001" for i in incidents)


# ---------------------------------------------------------------------------
# The following findings came from an independent review. Each one named a
# pair of twin tests; those below close the main test AND its twin, not just
# the first.
# ---------------------------------------------------------------------------


class _ExplodingDict(dict):
    """A dict whose key access ALWAYS raises something that is NOT
    KeyError/IndexError/TypeError. It proves `_content` only catches those
    three types and lets anything else through untouched."""

    def __getitem__(self, key):
        raise RuntimeError("unexpected failure, not a KeyError/IndexError/TypeError")


def test_content_does_not_catch_exceptions_outside_the_three_named():
    """`_content` declares `except (KeyError, IndexError, TypeError)`. A
    broader `except Exception` there would turn ANY failure (a real bug on
    that line, not just a corrupted envelope) into InvalidOutput: exactly the
    silence the system must not allow.

    Twin: `_cost` uses no try/except, only `isinstance` and `math.isfinite`,
    so it has no equivalent bare except to close (confirmed by reading
    triage/classifier.py: `_cost` contains no `except` clause)."""
    with pytest.raises(RuntimeError):
        _content(_ExplodingDict())


@respx.mock
def test_boolean_true_cost_is_not_accepted_as_a_fabricated_euro(tmp_path):
    """`isinstance(True, int)` is `True` in Python. Without the explicit
    `and not isinstance(raw, bool)` guard in `_cost`, a `"cost_eur": true`
    would be accepted as €1.0 fabricated out of nothing, in the number that
    feeds the cost per 1,000 requests."""
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={
        "choices": [{"message": {"content": OK_HIGH}}],
        "usage": {"cost_eur": True},
    }))
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, cost = Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 1
    assert cost == 0.0
    assert any(i["type"] == TYPE_COST_NOT_REPORTED for i in st.incidents())


@respx.mock
def test_boolean_false_cost_also_records_an_incident_not_just_zero(tmp_path):
    """The more insidious half of this case: `False` ALREADY gives 0.0
    numerically, so without the guard the returned cost would be identical to
    the correct one, but the incident documenting that the cost was not
    reported would silently disappear."""
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={
        "choices": [{"message": {"content": OK_HIGH}}],
        "usage": {"cost_eur": False},
    }))
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, cost = Classifier(c, ENV, CFG, st).classify(REQ)
    assert cost == 0.0
    assert any(i["type"] == TYPE_COST_NOT_REPORTED for i in st.incidents())


@respx.mock
def test_nan_cost_is_counted_as_zero_with_an_incident(tmp_path):
    """Twin of the boolean cost tests: `float("nan")` parses as a float
    without error, so without `math.isfinite` it would pass the type filter."""
    route = respx.post(URL).mock(return_value=response_non_finite_cost(OK_HIGH, "NaN"))
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, cost = Classifier(c, ENV, CFG, st).classify(REQ)
    assert cost == 0.0
    assert any(i["type"] == TYPE_COST_NOT_REPORTED for i in st.incidents())


@respx.mock
def test_infinite_cost_is_counted_as_zero_with_an_incident(tmp_path):
    """Twin of the boolean cost tests: like NaN, `float("inf")` also parses without error."""
    route = respx.post(URL).mock(return_value=response_non_finite_cost(OK_HIGH, "Infinity"))
    st = store(tmp_path)
    with httpx.Client() as c:
        classif, model, cost = Classifier(c, ENV, CFG, st).classify(REQ)
    assert cost == 0.0
    assert any(i["type"] == TYPE_COST_NOT_REPORTED for i in st.incidents())


@respx.mock
def test_max_wait_comes_from_config_not_hardcoded(tmp_path):
    """`cfg.http_max_wait_s` (7.0 in CFG) has to CAP a real wait, not just
    pass unexercised. A 429 with Retry-After: 3600 forces the situation where
    the cap matters: if the code hardcoded 30.0 (its own default, see
    DEFAULT_HTTP_MAX_WAIT_S) or used Retry-After raw, the recorded wait would
    not be 7.0."""
    waits = []
    route = respx.post(URL)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "3600"}),
        response(OK_HIGH),
    ]
    with httpx.Client() as c:
        Classifier(c, ENV, CFG, store(tmp_path), sleep=waits.append).classify(REQ)
    assert waits == [7.0]


@respx.mock
def test_timeout_comes_from_config_not_hardcoded_to_any_fixed_value(tmp_path):
    """Twin: uses CFG_OTHER_HTTP.http_timeout_s (9.25), DIFFERENT from both
    CFG's 3.5 and the real 10 in config/config.yaml. Only this way is a
    hardcode to the value CFG ALREADY has exposed instead of passing by
    chance."""
    route = respx.post(URL).mock(return_value=response(OK_HIGH))
    with httpx.Client() as c:
        Classifier(c, ENV, CFG_OTHER_HTTP, store(tmp_path)).classify(REQ)
    assert route.call_count == 1
    timeout = route.calls.last.request.extensions["timeout"]
    assert timeout["connect"] == CFG_OTHER_HTTP.http_timeout_s == 9.25


@respx.mock
def test_retries_come_from_config_not_hardcoded_to_any_fixed_value(tmp_path):
    """Twin: CFG_OTHER_HTTP.http_retries=6, different from CFG's 4. Five
    503s followed by a 200 only succeed if >= 6 attempts are allowed."""
    success_route = respx.post(URL)
    success_route.side_effect = [httpx.Response(503)] * 5 + [response(OK_HIGH)]
    with httpx.Client() as c:
        classif, model, _ = Classifier(
            c, ENV, CFG_OTHER_HTTP, store(tmp_path), sleep=lambda s: None,
        ).classify(REQ)
    assert model == "cheap-model"
    assert success_route.calls.call_count == 6


@respx.mock
def test_exhausted_retries_respect_the_exact_limit_from_config(tmp_path):
    """Twin: six 503s in a row exhaust the attempts at exactly 6 (not 4,
    CFG's value, nor 5 nor 7), proving the ceiling and not just ">= 6"."""
    route = respx.post(URL)
    route.side_effect = [httpx.Response(503)] * 6
    with httpx.Client() as c:
        with pytest.raises(HttpError):
            Classifier(
                c, ENV, CFG_OTHER_HTTP, store(tmp_path), sleep=lambda s: None,
            ).classify(REQ)
    assert route.calls.call_count == 6


@respx.mock
def test_backoff_base_comes_from_config_not_hardcoded_to_any_fixed_value(
        tmp_path, monkeypatch):
    """Twin: CFG_OTHER_HTTP.http_backoff_base_s (0.15), different from CFG's
    0.25 that `test_sleep_is_forwarded_to_send_...` already used. That test
    alone cannot tell a backoff_base_s hardcoded to 0.25 (CFG's own value)
    from one that is really read."""
    monkeypatch.setattr("triage.http.random.uniform", lambda a, b: 0.0)
    waits = []
    route = respx.post(URL)
    route.side_effect = [httpx.Response(500), response(OK_HIGH)]
    with httpx.Client() as c:
        Classifier(
            c, ENV, CFG_OTHER_HTTP, store(tmp_path), sleep=waits.append,
        ).classify(REQ)
    assert waits == [0.15]


@respx.mock
def test_the_request_messages_carry_the_subject_and_the_id_in_the_body(tmp_path):
    """`build_prompt` is tested in isolation
    (`test_the_prompt_carries_the_subject_and_the_id`), but nothing proved its
    output really reaches the wire. The model service matches on the subject
    and the REQ-2026-NNNN id in the joined prompt, so a silent
    `"messages": []` would degrade every classification to the service's
    default without anything here noticing."""
    route = respx.post(URL).mock(return_value=response(OK_HIGH))
    with httpx.Client() as c:
        Classifier(c, ENV, CFG, store(tmp_path)).classify(REQ)
    assert route.call_count == 1
    messages = body(route, 0)["messages"]
    assert messages
    text = " ".join(m["content"] for m in messages)
    assert REQ.id in text
    assert REQ.subject in text


@respx.mock
def test_tier_1_invalid_output_incident_carries_the_right_request_id(tmp_path):
    """Site 1 of 6: the TIER 1 invalid_model_output incident has to carry
    `req.id`, not a constant. An incident against the wrong id is worse than
    none: the failure is recorded, but tied to something that did not fail,
    so nobody finds it."""
    route = respx.post(URL)
    route.side_effect = [response(URGENCY_TEXT), response(OK_HIGH)]
    st = store(tmp_path)
    with httpx.Client() as c:
        Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 2
    rows = [i for i in st.incidents() if i["type"] == TYPE_INVALID_OUTPUT]
    assert rows
    assert all(i["request_id"] == "REQ-2026-0001" for i in rows)


@respx.mock
def test_tier_2_invalid_output_incident_carries_the_right_request_id(tmp_path):
    """Site 2 of 6: the previous test's twin for TIER 2. The first tier
    returns low confidence (valid, not invalid) to force escalation WITHOUT
    going through tier 1's invalid-output except, so the incident observed is
    unambiguously tier 2's."""
    route = respx.post(URL)
    route.side_effect = [response(OK_LOW), response(URGENCY_TEXT)]
    st = store(tmp_path)
    with httpx.Client() as c:
        with pytest.raises(ClassificationFailed):
            Classifier(c, ENV, CFG, st).classify(REQ)
    assert route.call_count == 2
    rows = [i for i in st.incidents() if i["type"] == TYPE_INVALID_OUTPUT]
    assert rows
    assert all(i["request_id"] == "REQ-2026-0001" for i in rows)


@respx.mock
def test_low_confidence_persists_incident_carries_the_right_request_id(tmp_path):
    """Site 4 of 6: the twin for `low_confidence_persists`, which no other
    test isolates (`test_incidents_carry_the_request_id_without_a_prefix`
    only exercises `low_confidence`, site 3 of 6, already covered)."""
    route = respx.post(URL)
    route.side_effect = [
        response('{"category":"content","urgency":1,"confidence":0.09,"reason":"ok"}'),
        response('{"category":"content","urgency":1,"confidence":0.24,"reason":"ok"}'),
    ]
    st = store(tmp_path)
    with httpx.Client() as c:
        Classifier(c, ENV, CFG, st).classify(REQ)
    rows = [i for i in st.incidents() if i["type"] == TYPE_LOW_CONFIDENCE_PERSISTS]
    assert rows
    assert all(i["request_id"] == "REQ-2026-0001" for i in rows)


@respx.mock
def test_cost_not_reported_incident_carries_the_right_request_id(tmp_path):
    """Site 5 of 6: the twin for the incident `_cost` writes."""
    route = respx.post(URL).mock(return_value=response_without_usage(OK_HIGH))
    st = store(tmp_path)
    with httpx.Client() as c:
        Classifier(c, ENV, CFG, st).classify(REQ)
    rows = [i for i in st.incidents() if i["type"] == TYPE_COST_NOT_REPORTED]
    assert rows
    assert all(i["request_id"] == "REQ-2026-0001" for i in rows)


# Site 3 of 6 (tier 1 low_confidence) is already covered by
# `test_incidents_carry_the_request_id_without_a_prefix`, and site 6 of 6
# (`_call`'s `record=`) by
# `test_sleep_is_forwarded_to_send_and_captures_the_backoff_from_config`.
# They are not duplicated here.


def test_build_prompt_uses_the_category_catalogue_from_its_argument():
    """If `build_prompt` ignored its `categories` argument and formatted with
    a list of real categories written into the code, this would go unnoticed
    with CFG's real catalogue (it would match by accident). A deliberately
    different catalogue exposes it."""
    odd_categories = ("zeta", "omega")
    messages = build_prompt(REQ, odd_categories)
    assert messages
    text = " ".join(m["content"] for m in messages)
    assert "zeta" in text
    assert "omega" in text
    assert "admin" not in text

import httpx
import pytest
import respx
from triage.http import HttpError, NotFound, send

URL = "http://x/test"

def call(client, **kw):
    kw.setdefault("timeout_s", 1)
    kw.setdefault("retries", 3)
    kw.setdefault("backoff_base_s", 0)
    kw.setdefault("max_wait_s", 30)
    return send(client, "GET", URL, headers={}, sleep=lambda s: None, **kw)

@respx.mock
def test_success_first_time():
    respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    with httpx.Client() as c:
        assert call(c).json() == {"ok": True}

@respx.mock
def test_retries_after_500_and_succeeds():
    route = respx.get(URL)
    route.side_effect = [httpx.Response(500), httpx.Response(200, json={"ok": True})]
    with httpx.Client() as c:
        assert call(c).status_code == 200
    assert route.call_count == 2

@respx.mock
def test_honours_retry_after():
    waits = []
    route = respx.get(URL)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, json={"ok": True}),
    ]
    with httpx.Client() as c:
        send(c, "GET", URL, headers={}, timeout_s=1, retries=3,
             backoff_base_s=0.5, max_wait_s=30, sleep=waits.append)
    assert waits == [2.0]

@respx.mock
def test_404_is_not_retried():
    route = respx.get(URL)
    route.mock(return_value=httpx.Response(404, json={"detail": "not found"}))
    with httpx.Client() as c:
        with pytest.raises(NotFound):
            call(c)
    assert route.call_count == 1

@respx.mock
def test_401_is_not_retried():
    route = respx.get(URL)
    route.mock(return_value=httpx.Response(401))
    with httpx.Client() as c:
        with pytest.raises(HttpError):
            call(c)
    assert route.call_count == 1

@respx.mock
def test_attempts_exhausted_raises_with_the_last_status():
    route = respx.get(URL)
    route.mock(return_value=httpx.Response(503))
    with httpx.Client() as c:
        with pytest.raises(HttpError) as e:
            call(c)
    assert e.value.status == 503
    assert route.call_count == 3

@respx.mock
def test_timeout_is_retried():
    route = respx.get(URL)
    route.side_effect = [httpx.ReadTimeout("slow"), httpx.Response(200, json={})]
    with httpx.Client() as c:
        assert call(c).status_code == 200
    assert route.call_count == 2


# --- Dimensions no test pinned before ---

class SpyClient:
    """Minimal client that records the arguments it is called with."""

    def __init__(self):
        self.seen = None

    def request(self, method, url, **kw):
        self.seen = kw
        return httpx.Response(200)


def test_configured_timeout_reaches_httpx():
    """Without this, deleting `timeout=timeout_s` would break no test and every
    request would fall back to httpx's default instead of the configured one."""
    spy = SpyClient()
    call(spy, timeout_s=7.5)
    assert spy.seen["timeout"] == 7.5

@respx.mock
def test_a_post_is_retried_too():
    """Both model calls in this project are POSTs: if the method decided the
    retry, nothing that matters would be retried."""
    route = respx.post(URL)
    route.side_effect = [httpx.Response(503), httpx.Response(200, json={"ok": True})]
    with httpx.Client() as c:
        response = send(c, "POST", URL, headers={}, body={"a": 1}, timeout_s=1,
                        retries=3, backoff_base_s=0, max_wait_s=30,
                        sleep=lambda s: None)
    assert response.status_code == 200
    assert route.call_count == 2

@respx.mock
def test_backoff_grows_between_attempts():
    waits = []
    respx.get(URL).mock(return_value=httpx.Response(503))
    with httpx.Client() as c:
        with pytest.raises(HttpError):
            send(c, "GET", URL, headers={}, timeout_s=1, retries=3,
                 backoff_base_s=0.5, max_wait_s=30, sleep=waits.append)
    assert len(waits) == 2
    assert waits[1] > waits[0]

@respx.mock
def test_status_is_none_if_no_response_ever_arrived():
    route = respx.get(URL)
    route.side_effect = httpx.ReadTimeout("slow")
    with httpx.Client() as c:
        with pytest.raises(HttpError) as e:
            call(c)
    assert e.value.status is None
    assert route.call_count == 3


# --- Retry-After: whatever the server sends is bounded before sleeping ---

@pytest.mark.parametrize("header, expected", [
    ("3600", 30.0),   # an hour, capped at the configured maximum
    ("-5", 0.0),      # negative: floored at 0, never a sleep that raises
    ("inf", None),    # non-finite: unusable, falls back to backoff
    ("nan", None),    # likewise
])
@respx.mock
def test_absurd_retry_after_is_bounded(header, expected):
    waits = []
    route = respx.get(URL)
    route.side_effect = [
        httpx.Response(503, headers={"Retry-After": header}),
        httpx.Response(200, json={"ok": True}),
    ]
    with httpx.Client() as c:
        send(c, "GET", URL, headers={}, timeout_s=1, retries=3,
             backoff_base_s=0.5, max_wait_s=30, sleep=waits.append)
    if expected is None:
        # backoff on the first attempt: 0.5 * 2**0 + jitter in [0, 0.5)
        assert 0.5 <= waits[0] < 1.0
    else:
        assert waits == [expected]


# --- The log has to be useful to a person ---

@respx.mock
def test_log_says_how_long_it_waits_and_why():
    lines = []
    route = respx.get(URL)
    route.side_effect = [
        httpx.Response(503, headers={"Retry-After": "1"}),
        httpx.Response(200, json={"ok": True}),
    ]
    with httpx.Client() as c:
        send(c, "GET", URL, headers={}, timeout_s=1, retries=3,
             backoff_base_s=0.5, max_wait_s=30, sleep=lambda s: None,
             record=lambda event, msg: lines.append(msg))
    assert any("1.0s" in m and "Retry-After" in m for m in lines)

@respx.mock
def test_a_permanent_failure_also_leaves_a_trace():
    """A storm of 401s cannot pass silently."""
    lines = []
    respx.get(URL).mock(return_value=httpx.Response(401))
    with httpx.Client() as c:
        with pytest.raises(HttpError):
            send(c, "GET", URL, headers={}, timeout_s=1, retries=3,
                 backoff_base_s=0, max_wait_s=30, sleep=lambda s: None,
                 record=lambda event, msg: lines.append(msg))
    assert any("401" in m for m in lines)

@respx.mock
def test_the_cap_also_bounds_the_calculated_backoff():
    """The cap is not only for what the server sends: the backoff grows
    exponentially and cannot exceed the configured maximum either."""
    waits = []
    respx.get(URL).mock(return_value=httpx.Response(503))
    with httpx.Client() as c:
        with pytest.raises(HttpError):
            send(c, "GET", URL, headers={}, timeout_s=1, retries=3,
                 backoff_base_s=100, max_wait_s=1.0, sleep=waits.append)
    # Without the cap the first backoff would already be 100s and the second 200s.
    assert waits and all(w <= 1.0 for w in waits)

@respx.mock
def test_the_cap_comes_from_configuration_not_code():
    """A non-default value has to show: if `_wait` ignored its parameter, the
    cap would not be configuration."""
    waits = []
    route = respx.get(URL)
    route.side_effect = [
        httpx.Response(503, headers={"Retry-After": "3600"}),
        httpx.Response(200, json={"ok": True}),
    ]
    with httpx.Client() as c:
        send(c, "GET", URL, headers={}, timeout_s=1, retries=3,
             backoff_base_s=0.5, max_wait_s=2.0, sleep=waits.append)
    assert waits == [2.0]

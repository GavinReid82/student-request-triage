"""HTTP client with a single retry policy.

It is deliberately our own and not the OpenAI SDK's: handling 429, 503 and
Retry-After is exactly the behaviour that matters here, and it has to be in
plain sight and testable.
"""
import math
import random
import time

import httpx

RETRYABLE = (429, 500, 502, 503, 504)


class HttpError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NotFound(HttpError):
    """404. Retrying it changes nothing."""


def _wait(response, attempt: int, base: float, maximum: float) -> tuple[float, str]:
    """How long to sleep before the next attempt, and why.

    Retry-After wins; if it is missing or unusable, exponential backoff with
    jitter. The value is chosen by the server, so it is not used raw:
    non-finite values (`inf`, `nan`) are discarded, it is floored at 0 (a
    negative sleep raises ValueError) and capped at `maximum`. The cap also
    applies to the backoff, which grows exponentially and has no reason to be
    unbounded either.
    """
    seconds = None
    reason = "backoff"
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                value = float(header)
            except ValueError:
                value = None  # includes the HTTP-date form: see send's docstring
            if value is not None and math.isfinite(value):
                seconds = value
                reason = "Retry-After"
    if seconds is None:
        seconds = base * (2 ** attempt) + random.uniform(0, base)
    return max(0.0, min(seconds, maximum)), reason


def send(client, method, url, *, headers, body=None, timeout_s,
         retries, backoff_base_s, max_wait_s, sleep=time.sleep,
         record=None):
    """Makes the request, retrying whatever is worth retrying.

    Raises NotFound on 404 and HttpError on every other permanent failure.
    It never returns an error response: either it succeeds or it raises.

    `timeout_s` is per attempt, not in total: with `retries=3` the wall-clock
    time can be three times that number plus the waits. On top of that, a
    bare float becomes `httpx.Timeout(timeout_s)`, which sets separate
    connect, read, write and pool budgets, so the pathological ceiling of a
    single attempt is four times the configured number.

    `max_wait_s` caps every wait between attempts, whether it comes from
    Retry-After or from the backoff. Without that cap, a `Retry-After: 3600`
    on each attempt would turn one call into two hours of stalled process.

    `HttpError.status` is None when the attempts ran out without receiving
    any HTTP response at all (a transport failure on every one). Anyone
    comparing `status` with a number has to allow for that None.

    POSTs are retried just like GETs, and that is safe here because of how
    these two particular services fail: the model service's 503 and the
    student API's 429 are returned *before* the server does any work or
    records any cost, so repeating the request duplicates neither executions
    nor charges. The argument relies on `timeout_s` staying above the slow
    paths of those services (3 s for the model, 4 s for students): with
    `timeout_s=1` it was measured that a single call to `send` caused 3 real
    model executions and 3 times the cost. Lowering `timeout_s` below those
    4 s breaks the guarantee, not just the performance.

    The HTTP-date form of Retry-After (RFC 9110 §10.2.3) is deliberately not
    parsed: neither service sends it, and the case falls back to backoff.
    """
    last = None
    for attempt in range(retries):
        try:
            response = client.request(
                method, url, headers=headers, json=body, timeout=timeout_s
            )
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = HttpError(f"{method} {url}: transport failure: {e}")
            _retry(
                url, f"{type(e).__name__}: {e}", "transport", None, attempt,
                retries, backoff_base_s, max_wait_s, sleep, record,
            )
            continue

        if response.status_code == 404:
            if record:
                record("http_permanent", f"{url}: 404, not retried")
            raise NotFound(f"{method} {url}: 404", 404)

        if response.status_code in RETRYABLE:
            last = HttpError(
                f"{method} {url}: {response.status_code}", response.status_code
            )
            _retry(
                url, str(response.status_code), "http_retryable", response,
                attempt, retries, backoff_base_s, max_wait_s, sleep,
                record,
            )
            continue

        if response.status_code >= 400:
            if record:
                record(
                    "http_permanent",
                    f"{url}: {response.status_code}, not retried",
                )
            raise HttpError(
                f"{method} {url}: {response.status_code} {response.text[:200]}",
                response.status_code,
            )

        return response

    raise last


def _retry(url, cause, event, response, attempt, retries, base,
           maximum, sleep, record):
    """Waits as long as needed before the next attempt, and writes it down.

    The log entry is written after the wait is worked out, not before: anyone
    reading the logs needs to see what failed, why it is retried and how long
    it waited, and that third part only exists once it has been calculated.
    """
    attempts = f"(attempt {attempt + 1}/{retries})"
    if attempt + 1 >= retries:
        if record:
            record(event, f"{url}: {cause}, no attempts left {attempts}")
        return
    seconds, reason = _wait(response, attempt, base, maximum)
    if record:
        record(
            event,
            f"{url}: {cause}, waiting {seconds:.1f}s due to {reason} {attempts}",
        )
    sleep(seconds)

"""Student API client.

Caches by student_id, both hits and confirmed misses. This is not only about
cost: the mock API fails according to a global call counter, so every call we
do not make is failure surface that disappears. 43 requests reference 15
students, and a student_id that does not exist can appear in several of them:
without a negative cache, each of those requests would repeat the same 404
against the real API.

`calls` counts only the HTTP round trips this client started that ended in a
received response (success or a translated 404). It does not count the
internal retries `send` makes on a transient 429/503, nor cache hits,
positive or negative.
"""
import time

from .store import Store
from .config import Config, Env
from .http import NotFound, send
from .schemas import Student

TYPE_ALTERNATIVE_FIELD = "alternative_field"


class StudentNotFound(Exception):
    """The API confirms that student_id does not exist (or already confirmed it)."""


class StudentClient:
    """Wraps the student API with an in-memory cache and record normalisation.

    The hit cache lives in `_cache` (student_id -> Student); the confirmed
    miss cache lives in `_missing` (student_id already known not to exist).
    Both avoid a second HTTP request for the same student_id.
    """

    def __init__(self, client, env: Env, cfg: Config, store: Store,
                 sleep=time.sleep):
        """The `sleep` parameter exists because `send` binds its own
        `sleep=time.sleep` default when it is defined, so patching
        `triage.http.time.sleep` afterwards does not intercept it.
        """
        self._client = client
        self._env = env
        self._cfg = cfg
        self._store = store
        self._cache: dict[str, Student] = {}
        self._missing: set[str] = set()
        self.calls = 0
        # This parameter forwards a replaceable `sleep` to every `send` call,
        # so tests that pin `max_wait_s` or the backoff don't really sleep. The
        # default is still the real `time.sleep`, so that behaviour in
        # production (where nobody passes `sleep=`) is a real sleep and not a
        # silent no-op.
        self._sleep = sleep

    def get(self, student_id: str) -> Student:
        if student_id in self._cache:
            return self._cache[student_id]
        if student_id in self._missing:
            raise StudentNotFound(f"Student {student_id} does not exist")

        # This client's incidents are about a STUDENT, not a request: they are
        # namespaced so the request_id column is never mistaken for a student
        # id, and so two different student_ids never collide with each other
        # or with a real request_id.
        key = f"student:{student_id}"
        url = f"{self._env.student_api_url}/students/{student_id}"
        headers = {"Authorization": f"Bearer {self._env.token}"}
        try:
            response = send(
                self._client, "GET", url,
                headers=headers,
                timeout_s=self._cfg.http_timeout_s,
                retries=self._cfg.http_retries,
                backoff_base_s=self._cfg.http_backoff_base_s,
                max_wait_s=self._cfg.http_max_wait_s,
                sleep=self._sleep,
                record=lambda kind, detail: self._store.incident(
                    key, kind, detail),
            )
        except NotFound as e:
            self._missing.add(student_id)
            raise StudentNotFound(f"Student {student_id} does not exist") from e

        self.calls += 1
        data = response.json()

        if "status_enrolment" in data and "enrolment_status" not in data:
            # Student already accepts 'status_enrolment' as an alias through
            # AliasChoices (see schemas.py), so this is not a correction: it
            # records that the student record arrived with the alternative name.
            self._store.incident(
                key, TYPE_ALTERNATIVE_FIELD,
                "The student record arrives with 'status_enrolment' instead of "
                "'enrolment_status'.",
            )

        student = Student(**data)
        self._cache[student_id] = student
        return student

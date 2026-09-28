"""Domain schemas and input validation.

Nothing enters the system without passing through here. The types are strict
on purpose: an urgency that arrives as a string is an error, not something to
fix on our own initiative.
"""
import hashlib
import json
from datetime import datetime

from pydantic import (
    AliasChoices, BaseModel, ConfigDict, Field, StrictFloat, StrictInt,
)

REQUIRED_FIELDS = ("id", "received_at", "channel", "student_id", "subject", "message")
ALT_DATE_FORMAT = "%d/%m/%Y %H:%M"


class Request(BaseModel):
    id: str
    received_at: str
    channel: str
    student_id: str
    subject: str
    message: str


class Student(BaseModel):
    # A student record is a record, not a draft: the student client returns
    # the SAME object on every cache hit, so anyone mutating it would corrupt
    # the cache for everyone after them. Frozen, that mistake fails where it
    # is made.
    model_config = ConfigDict(frozen=True)

    student_id: str
    name: str
    email: str
    programme: str
    # One record in the student API arrives with this field named the other
    # way round. The alias is accepted and an incident is recorded; the
    # process does not break.
    enrolment_status: str = Field(
        validation_alias=AliasChoices("enrolment_status", "status_enrolment")
    )
    start_date: str
    progress_pct: int
    tutor: str
    language: str


class Classification(BaseModel):
    category: str
    urgency: StrictInt = Field(ge=1, le=3)
    confidence: StrictFloat | StrictInt = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


def _normalise_date(value: str) -> tuple[str | None, bool]:
    """Returns (iso_date, used_alternative_format)."""
    try:
        return datetime.fromisoformat(value).isoformat(), False
    except ValueError:
        pass
    try:
        return datetime.strptime(value, ALT_DATE_FORMAT).isoformat(), True
    except ValueError:
        return None, False


def validate_request(raw: dict) -> tuple[Request | None, str | None, bool]:
    """Validates one raw record from the input file.

    Returns (request, invalid_reason, used_alternative_date_format).
    An invalid request is not processed: it is recorded with the reason.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        return None, f"missing fields: {', '.join(missing)}", False

    empty = [f for f in REQUIRED_FIELDS if not str(raw[f]).strip()]
    if empty:
        return None, f"empty fields: {', '.join(empty)}", False

    date, fallback = _normalise_date(str(raw["received_at"]).strip())
    if date is None:
        return None, f"malformed field: received_at={raw['received_at']!r}", False

    data = {f: str(raw[f]).strip() for f in REQUIRED_FIELDS}
    data["received_at"] = date
    return Request(**data), None, fallback


def record_hash(raw: dict) -> str:
    """Fingerprint of the content, independent of key order.

    It separates a benign duplicate (same id, same content) from a conflict
    (same id, different content), which cannot be resolved automatically.
    """
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

import pytest
from pydantic import ValidationError
from triage.schemas import Student, Classification, validate_request, record_hash

BASE = {"id": "REQ-1", "received_at": "2026-09-10T09:15:00", "channel": "email",
        "student_id": "STU-1", "subject": "a", "message": "m"}

def test_student_accepts_the_alternative_field_name():
    s = Student(student_id="STU-03871", name="n", email="e", programme="p",
                status_enrolment="active", start_date="2026-02-01",
                progress_pct=55, tutor="t", language="en")
    assert s.enrolment_status == "active"

def test_student_is_immutable_once_built():
    """The student client hands out the SAME object on every cache hit; if
    Student were not frozen, one caller's mutation would silently corrupt
    the cache for everyone else.
    """
    s = Student(student_id="STU-1", name="n", email="e", programme="p",
                enrolment_status="active", start_date="2026-02-01",
                progress_pct=55, tutor="t", language="en")
    with pytest.raises(ValidationError):
        s.tutor = "someone else"

def test_urgency_as_text_is_rejected():
    with pytest.raises(ValidationError):
        Classification(category="content", urgency="high", confidence=0.9, reason="x")

def test_urgency_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        Classification(category="content", urgency=7, confidence=0.9, reason="x")

def test_request_without_student_id_is_invalid():
    raw = {k: v for k, v in BASE.items() if k != "student_id"}
    req, reason, _ = validate_request(raw)
    assert req is None
    assert "student_id" in reason

def test_request_with_empty_field_is_invalid():
    req, reason, _ = validate_request({**BASE, "subject": "   "})
    assert req is None
    assert "subject" in reason

def test_non_iso_date_is_normalised_and_flagged():
    req, reason, fallback = validate_request({**BASE, "received_at": "10/09/2026 09:15"})
    assert reason is None
    assert fallback is True
    assert req.received_at == "2026-09-10T09:15:00"

def test_hash_is_stable_and_independent_of_key_order():
    assert record_hash(BASE) == record_hash(dict(reversed(list(BASE.items()))))
    assert record_hash(BASE) != record_hash({**BASE, "message": "other"})

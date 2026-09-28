"""Orchestration of the steps, one request at a time.

Each step commits its state to the database before moving on to the next. A
failure stops that request, never the run: it is marked as failed, the
incident is recorded and processing moves on to the next one.

What to do with an arrival is decided by the store (classify_arrival), not by
this module: new, duplicate or conflict. The only distinction made here,
among duplicates, is between ignoring it (already finished, or already seen
earlier in this same run) and resuming it (left half done by a previous run).

Resuming does not pay twice: if the row already has a committed
classification, it is reused instead of calling the model again. Validating
(pure) and enriching (cached) are repeated, because the draft needs the
student's record.
"""
import json
import logging

from .store import (
    FAILED_STATE, TERMINAL_STATES, TYPE_DRAFT_RESUMED)
from .students import StudentNotFound
from .http import HttpError
from .schemas import Classification, record_hash, validate_request
from .drafter import NoTemplate, write_draft
from .model_output import InvalidOutput

log = logging.getLogger(__name__)

# Results of process_one that are not states: the request was not processed.
RESULT_DUPLICATE = "duplicate"
RESULT_CONFLICT = "conflict"

# Incident types born in this module.
TYPE_INVALID_REQUEST = "invalid_request"
TYPE_ALTERNATIVE_DATE = "alternative_date_format"
TYPE_STUDENT_NOT_FOUND = "student_not_found"
TYPE_NO_TEMPLATE = "no_template"
TYPE_EXTERNAL_FAILURE = "external_failure"
TYPE_UNEXPECTED_FAILURE = "unexpected_failure"

# Columns that, when all present, prove the classification was already
# committed (and paid for). Reuse is decided by them and not by the state,
# because a failed row no longer says which step it reached.
CLASSIFICATION_COLUMNS = ("category", "urgency", "confidence", "classification_reason")


def _reference(raw) -> tuple[str, str]:
    """Returns (reference, hash) for a record, whether or not it is an object.

    A record with no id, or that is not even an object, is stored as well:
    with a reference derived from its content, so running it again is as
    idempotent as running a valid one.
    """
    content = raw if isinstance(raw, dict) else {"record": repr(raw)}
    rec_hash = record_hash(content)
    request_id = str(raw.get("id") or "").strip() if isinstance(raw, dict) else ""
    return request_id or f"no-id:{rec_hash[:12]}", rec_hash


def process_one(raw, store, student_client, classifier, cfg, *,
                seen=None) -> str:
    """Processes one raw record.

    Returns the final state reached, or RESULT_DUPLICATE / RESULT_CONFLICT if
    it was not processed. `seen` holds the references already handled in this
    run: a failed request that reappears in the same file is not retried now,
    it is left for the next run.
    """
    if seen is None:
        seen = set()
    ref, rec_hash = _reference(raw)

    arrival = store.classify_arrival(ref, rec_hash)
    if arrival == "conflict":
        # The store already recorded the incident. The work done on the
        # original content is still valid: it is not touched.
        seen.add(ref)
        log.warning("%s: same id with different content; left for human review", ref)
        return RESULT_CONFLICT
    if arrival == "duplicate":
        previous_state, _ = store.state(ref)
        if ref in seen or previous_state in TERMINAL_STATES:
            seen.add(ref)
            store.duplicate(ref)
            log.info("%s already handled (%s); ignored", ref, previous_state)
            return RESULT_DUPLICATE
        log.info("%s: resuming from %s", ref, previous_state)
    seen.add(ref)

    try:
        return _steps(raw, ref, store, student_client, classifier, cfg)
    # If the store itself fails inside these handlers, the exception propagates
    # and stops the run: carrying on unable to record would make every later
    # failure silent.
    except HttpError as e:
        store.mark(ref, FAILED_STATE, reason=f"external_failure: {e}")
        store.incident(ref, TYPE_EXTERNAL_FAILURE, str(e))
        log.error("%s failed against an external service: %s", ref, e)
        return FAILED_STATE
    except Exception as e:  # nothing passes silently, not even the unforeseen
        store.mark(ref, FAILED_STATE, reason=f"unexpected_failure: {e!r}")
        store.incident(ref, TYPE_UNEXPECTED_FAILURE, repr(e))
        log.exception("%s failed unexpectedly", ref)
        return FAILED_STATE


def _steps(raw, ref, store, student_client, classifier, cfg) -> str:
    # Step 1: validate.
    if isinstance(raw, dict):
        request, reason, alternative_date = validate_request(raw)
    else:
        request, reason, alternative_date = None, "the record is not a JSON object", False
    if request is None:
        store.mark(ref, "invalid", reason=reason)
        store.incident(ref, TYPE_INVALID_REQUEST, reason)
        log.warning("%s invalid: %s", ref, reason)
        return "invalid"
    if alternative_date:
        store.incident(
            ref, TYPE_ALTERNATIVE_DATE,
            f"received_at arrived as {raw['received_at']!r}; normalised to ISO-8601.")
    store.mark(ref, "validated", student_id=request.student_id)

    # Step 2: enrich with the student's real record.
    try:
        student = student_client.get(request.student_id)
    except StudentNotFound as e:
        store.mark(ref, "invalid", reason=f"student_not_found: {request.student_id}")
        store.incident(ref, TYPE_STUDENT_NOT_FOUND, str(e))
        return "invalid"
    store.mark(ref, "enriched")

    # Steps 3 and 4: classify, or reuse the classification already paid for.
    row = store.row(ref)
    if all(row[c] is not None for c in CLASSIFICATION_COLUMNS):
        classif = Classification(
            category=row["category"], urgency=row["urgency"],
            confidence=row["confidence"], reason=row["classification_reason"])
        # No cost_eur: it accumulates in the store and is already counted.
        store.mark(ref, "classified")
        log.info("%s: reusing the already committed classification", ref)
    else:
        try:
            classif, model, cost = classifier.classify(request)
        except InvalidOutput as e:
            # The money was spent even though no model gave a valid output.
            store.mark(ref, "escalated", reason=f"invalid_model_output: {e}",
                       cost_eur=getattr(e, "cost_eur", 0.0))
            log.warning("%s escalated: invalid model output: %s", ref, e)
            return "escalated"
        store.mark(
            ref, "classified",
            category=classif.category, urgency=classif.urgency,
            confidence=classif.confidence, classification_reason=classif.reason,
            model_used=model, cost_eur=cost,
        )

    # Step 6: below the threshold a person decides, with no draft. The
    # classifier has already tried the precise model; if it is still low, the
    # request is escalated.
    if classif.confidence < cfg.confidence_threshold:
        store.mark(
            ref, "escalated",
            reason=(f"confidence {classif.confidence} below the threshold "
                    f"{cfg.confidence_threshold}"))
        log.warning("%s escalated: confidence %s below the threshold %s",
                    ref, classif.confidence, cfg.confidence_threshold)
        return "escalated"

    # Step 5: a draft in the student's language, with their real data. If one
    # already exists on resume, it is not even rewritten: the student already
    # has theirs, and a later template change cannot escalate finished work.
    already_existed = store.draft(ref) is not None
    if not already_existed:
        try:
            body = write_draft(request, student, classif, cfg)
        except NoTemplate as e:
            store.mark(ref, "escalated", reason=f"no_template: {e}")
            store.incident(ref, TYPE_NO_TEMPLATE, str(e))
            log.warning("%s escalated: no template: %s", ref, e)
            return "escalated"
        # The drafts primary key is the final guarantee against another
        # process that saved one between the check and the write.
        already_existed = not store.save_draft(ref, student.language, body)
    if already_existed:
        store.incident(
            ref, TYPE_DRAFT_RESUMED,
            "A draft already existed for this request; no new one is generated.")
    store.mark(ref, "drafted")

    # Step 7: result persisted. The reason from an earlier failure no longer
    # describes the row; its history stays in incidents.
    store.mark(ref, "completed", reason=None)
    return "completed"


def process_all(path, store, student_client, classifier, cfg) -> dict:
    """Processes the whole file and returns the store's summary.

    The summary includes "processed_this_run": how many records THIS run
    actually handled (new or resumed), not counting duplicates or conflicts.
    It is the honest denominator for the cost per 1,000 requests: a second
    run over an already-processed file pays for nothing new, so neither
    "completed" nor the store's total can be used to spread the cost
    measured in this run.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    log.info("Loaded %d records from %s", len(records), path)
    seen: set[str] = set()
    processed_this_run = 0
    for i, raw in enumerate(records, 1):
        result = process_one(raw, store, student_client, classifier, cfg,
                             seen=seen)
        if result not in (RESULT_DUPLICATE, RESULT_CONFLICT):
            processed_this_run += 1
        log.info("[%d/%d] %s -> %s", i, len(records), _reference(raw)[0], result)
    summary = store.summary()
    summary["processed_this_run"] = processed_this_run
    return summary

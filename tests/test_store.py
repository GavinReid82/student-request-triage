import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from triage.store import (
    STATES,
    TYPE_DRAFT_RESUMED,
    TYPE_CONFLICT,
    Store,
    UnknownRequest,
)

# The real value lives in config/config.yaml; a test value is injected here,
# the same way process start-up does it.
LOCK_WAIT_MS = 5000


def new(tmp_path):
    s = Store(str(tmp_path / "t.db"), LOCK_WAIT_MS)
    s.initialise()
    return s


def test_register_is_idempotent(tmp_path):
    """Registering the same id twice repeats no work and overwrites nothing.

    The second arrival carries a DIFFERENT hash on purpose: with the same
    hash, a store that silently overwrites (INSERT OR REPLACE) would be
    indistinguishable from one that ignores, and the test would pin nothing.
    Here, if it overwrote, the hash would become 'h2' and the state would go
    back to 'received'.
    """
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "validated", student_id="STU-1")
    s.register("R1", "h2")  # same id, different content
    assert s.state("R1") == ("validated", "h1")
    assert s.rows()[0]["student_id"] == "STU-1"
    assert len(s.rows()) == 1


def test_register_without_id_fails(tmp_path):
    """SQLite allows NULL in a TEXT PRIMARY KEY, so the primary key alone does
    not prevent three rows with a NULL id. The explicit NOT NULL does."""
    s = new(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        s.register(None, "h1")
    assert s.rows() == []


def test_register_without_hash_fails_instead_of_being_ignored(tmp_path):
    """Only the primary-key conflict is ignored. Any other violation has to
    blow up: with INSERT OR IGNORE the request vanished without a sound and
    could then be marked through the whole state machine without a row."""
    s = new(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        s.register("R9", None)
    assert s.rows() == []
    assert s.summary()["total"] == 0


def test_mark_updates_state_and_fields(tmp_path):
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "completed", category="technical", urgency=3, confidence=0.9)
    state, _ = s.state("R1")
    assert state == "completed"
    assert s.rows()[0]["category"] == "technical"


def test_row_returns_the_request_or_none(tmp_path):
    s = new(tmp_path)
    s.register("R1", "h1")
    s.register("R2", "h2")
    s.mark("R1", "classified", category="technical", classification_reason="ok")
    row = s.row("R1")
    assert (row["id"], row["state"], row["category"], row["classification_reason"]) == (
        "R1", "classified", "technical", "ok")
    assert s.row("R-missing") is None


def test_marking_an_unknown_request_fails(tmp_path):
    """The UPDATE found no row and returned silently: a request could walk the
    whole state machine without leaving a trace."""
    s = new(tmp_path)
    with pytest.raises(UnknownRequest):
        s.mark("DOES-NOT-EXIST", "completed", category="technical")
    assert s.summary()["total"] == 0


@pytest.mark.parametrize("invalid_state", ["compelted", "", "COMPLETED"])
def test_a_state_outside_the_closed_set_fails(tmp_path, invalid_state):
    """A terminal state with a typo does not match TERMINAL_STATES, so the
    request would be reprocessed (and paid for again) on every run, and
    summary() would undercount it. The set of states is closed and enforced
    by the database, not by a check in Python."""
    s = new(tmp_path)
    s.register("R1", "h1")
    with pytest.raises(sqlite3.IntegrityError):
        s.mark("R1", invalid_state)
    assert s.state("R1") == ("received", "h1")


def test_cost_accumulates_instead_of_being_overwritten(tmp_path):
    """If the process dies after 'classified' and resumes, the model really is
    called twice and paid for twice. Overwriting the cost understated it
    exactly when there had been a crash."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "classified", cost_eur=0.002, model_used="sim-fast")
    s.mark("R1", "classified", cost_eur=0.002, model_used="sim-precise")
    assert s.summary()["cost_eur"] == 0.004
    # The other fields are replaced.
    assert s.rows()[0]["model_used"] == "sim-precise"


def test_a_second_draft_is_rejected(tmp_path):
    s = new(tmp_path)
    s.register("R1", "h1")
    assert s.save_draft("R1", "en", "hello") is True
    assert s.save_draft("R1", "en", "hello again") is False
    assert s.count_drafts() == 1


def test_a_different_second_draft_does_not_overwrite_the_original(tmp_path):
    """Nobody gets two drafts, and nobody gets a draft different from the one
    already accepted. The primary key of 'drafts' guarantees it."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.save_draft("R1", "en", "original body")
    assert s.save_draft("R1", "es", "un cuerpo completamente distinto") is False
    row = s.draft("R1")
    assert row["body"] == "original body"
    assert row["language"] == "en"
    assert s.count_drafts() == 1


def test_summary_counts_by_state_and_category(tmp_path):
    s = new(tmp_path)
    for rid, cat in [("R1", "technical"), ("R2", "technical"), ("R3", "withdrawal")]:
        s.register(rid, "h")
        s.mark(rid, "completed", category=cat)
    s.register("R4", "h")
    s.mark("R4", "escalated", reason="low confidence")
    r = s.summary()
    assert r["completed"] == 3
    assert r["escalated"] == 1
    assert r["by_category"]["technical"] == 2


def test_incidents_are_stored(tmp_path):
    s = new(tmp_path)
    s.incident("R1", "alternative_field", "status_enrolment")
    assert s.summary()["incidents"] == 1


# --- Benign duplicate vs. content conflict ---
#
# The content hash (triage.schemas.record_hash) is the only thing that can
# separate these two cases, which look identical at the "same id" level. The
# decision lives in classify_arrival(), inside the store: if it lived in the
# caller, a test that did the escalation itself would be checking its own work
# and not the store.

def test_classify_arrival_new_registers_the_request(tmp_path):
    s = new(tmp_path)
    assert s.classify_arrival("R1", "h1") == "new"
    assert s.state("R1") == ("received", "h1")
    assert s.summary()["total"] == 1


def test_classify_arrival_duplicate_does_not_touch_the_original_row(tmp_path):
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "completed", category="technical")

    assert s.classify_arrival("R1", "h1") == "duplicate"

    r = s.summary()
    assert r["completed"] == 1
    assert r["escalated"] == 0
    assert r["conflicts"] == 0
    assert r["incidents"] == 0
    assert s.state("R1") == ("completed", "h1")


def test_benign_duplicate_is_counted_and_does_not_touch_the_original_row(tmp_path):
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "completed", category="technical")

    # The same request arrives again, with the same content: same hash.
    assert s.classify_arrival("R1", "h1") == "duplicate"
    s.duplicate("R1")

    r = s.summary()
    assert r["duplicates_ignored"] == 1
    assert r["completed"] == 1  # the original row is still intact
    assert r["escalated"] == 0
    assert s.state("R1") == ("completed", "h1")


def test_repeated_duplicate_does_not_stack(tmp_path):
    """The same id arriving again several times (for example because the file
    is reprocessed more than once) is the same fact observed again, not a new
    fact each time. Without a uniqueness constraint on `duplicates`, every
    repeat added another row and `duplicates_ignored` climbed without limit."""
    s = new(tmp_path)
    s.register("R1", "h1")

    s.duplicate("R1")
    s.duplicate("R1")
    s.duplicate("R1")

    assert s.summary()["duplicates_ignored"] == 1


def test_duplicates_of_different_ids_are_counted_separately(tmp_path):
    """The uniqueness constraint is per request, not global: a repeated id not
    stacking cannot mean the count gets stuck at 1 however many distinct ids
    reappear."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.register("R2", "h2")

    s.duplicate("R1")
    s.duplicate("R1")
    s.duplicate("R2")
    s.duplicate("R2")

    assert s.summary()["duplicates_ignored"] == 2


def test_a_conflict_records_an_incident_and_does_not_destroy_completed_work(tmp_path):
    """Completed work really was completed: paid for, classified and with a
    draft sent. What needs a human is the NEW content. Overwriting the state
    with 'escalated' took one away from completed and left category/urgency/
    confidence/cost describing the old content next to a state that talked
    about the new one. Both facts are kept: the row intact, and the incident."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "completed", category="technical", urgency=3,
           confidence=0.9, cost_eur=0.002)

    assert s.classify_arrival("R1", "h2") == "conflict"

    r = s.summary()
    assert r["conflicts"] == 1
    assert r["incidents"] == 1
    assert r["completed"] == 1  # the completed work still counts
    assert r["escalated"] == 0
    assert r["duplicates_ignored"] == 0  # a conflict is not a duplicate
    assert s.state("R1") == ("completed", "h1")

    row = s.rows()[0]
    assert row["category"] == "technical"
    assert row["confidence"] == 0.9

    # A human has to be able to resolve it by reading only the database.
    inc = s.incidents()[0]
    assert inc["type"] == TYPE_CONFLICT
    assert "h1" in inc["detail"] and "h2" in inc["detail"]


def test_a_repeated_conflict_leaves_a_single_incident(tmp_path):
    """Running the same file again has to leave the database identical.
    Without the UNIQUE constraint on incidents, the same conflict stacked once
    per run, without limit."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "completed", category="technical")

    for _ in range(3):
        assert s.classify_arrival("R1", "h2") == "conflict"

    r = s.summary()
    assert r["incidents"] == 1
    assert r["conflicts"] == 1
    assert r["completed"] == 1


# --- The rest of the interface ---

def test_state_returns_none_if_the_request_does_not_exist(tmp_path):
    s = new(tmp_path)
    assert s.state("R-missing") is None


def test_escalated_returns_only_the_escalated_rows(tmp_path):
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "completed", category="technical")
    s.register("R2", "h2")
    s.mark("R2", "escalated", reason="low confidence")

    escalated = s.escalated()
    assert len(escalated) == 1
    assert escalated[0]["id"] == "R2"
    assert escalated[0]["reason"] == "low confidence"


def test_the_set_of_states_and_the_schema_cannot_drift_apart(tmp_path):
    """The schema's CHECK is generated from STATES, so every declared state
    has to be acceptable and no other."""
    s = new(tmp_path)
    s.register("R1", "h1")
    for state in STATES:
        s.mark("R1", state)
        assert s.state("R1")[0] == state


def test_the_connection_sets_wal_and_the_lock_wait(tmp_path):
    """Two processes writing at once cannot rely on SQLite's defaults. The
    wait comes from the configuration, not from the code."""
    s = Store(str(tmp_path / "t.db"), 7000)
    s.initialise()
    # The connection is queried directly: these are connection settings, not domain.
    assert s._con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert s._con.execute("PRAGMA busy_timeout").fetchone()[0] == 7000
    s.close()


def test_state_persists_after_reopening_the_connection(tmp_path):
    """A durability proxy, not a crash: the connection is closed cleanly and
    reopened. It checks that a second connection sees the last state the
    first one committed, because each step commits immediately and not at the
    end of the request. The real hard crash is covered by the next test."""
    path = str(tmp_path / "t.db")

    s1 = Store(path, LOCK_WAIT_MS)
    s1.initialise()
    s1.register("R1", "h1")
    s1.mark("R1", "classified", category="technical", urgency=2, confidence=0.8)
    s1.close()

    s2 = Store(path, LOCK_WAIT_MS)
    s2.initialise()  # CREATE TABLE IF NOT EXISTS: does not touch existing data
    assert s2.state("R1") == ("classified", "h1")
    assert s2.rows()[0]["category"] == "technical"
    s2.close()


HARD_CRASH_SCRIPT = """
import os, sys
from triage.store import Store

s = Store(sys.argv[1], {wait})
s.initialise()
s.register("R1", "h1")
s.mark("R1", "validated", student_id="STU-1")
s.mark("R1", "classified", category="technical", cost_eur=0.002)
os._exit(1)  # hard kill: no close(), no final commit, no atexit
"""


def test_a_hard_process_crash_does_not_lose_committed_steps(tmp_path):
    """The real test of committing after EVERY step.

    A subprocess writes three steps and is killed with os._exit(1), which
    neither closes the connection nor runs any clean-up. The parent reopens
    the file and has to see all three. If the commit() calls are removed,
    SQLite discards the open transaction when the process dies and this test
    fails.
    """
    path = str(tmp_path / "t.db")
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", HARD_CRASH_SCRIPT.format(wait=LOCK_WAIT_MS), path],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr

    # Reopened without initialising: everything, schema included, was committed by the child.
    s = Store(path, LOCK_WAIT_MS)
    assert s.state("R1") == ("classified", "h1")
    row = s.rows()[0]
    assert row["student_id"] == "STU-1"
    assert row["category"] == "technical"
    assert s.summary()["cost_eur"] == 0.002
    s.close()


def test_a_failed_request_is_counted_separately_and_not_as_in_progress(tmp_path):
    """'failed' was a loose literal inside summary() while its three siblings
    lived in TERMINAL_STATES. It has to be a storable state and be counted
    separately: neither finished nor still in progress."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "failed", reason="the model did not respond")
    s.register("R2", "h2")
    s.mark("R2", "classified", category="technical")

    r = s.summary()
    assert r["failed"] == 1
    assert r["in_progress"] == 1  # only R2
    assert r["completed"] == 0


def test_accessors_return_empty_when_there_is_nothing(tmp_path):
    s = new(tmp_path)
    assert s.draft("R-missing") is None
    assert s.incidents() == []
    assert s.summary()["conflicts"] == 0


def test_running_the_same_file_again_leaves_the_database_identical(tmp_path):
    """The idempotency requirement, end to end: process the batch, process it
    again in full and check the database has not changed by a single field. A
    store that silently overwrote would reset the rows to the initial state
    and lose the classification and cost already paid for."""
    s = new(tmp_path)
    batch = [("R1", "h1", "technical"), ("R2", "h2", "withdrawal")]
    for rid, h, cat in batch:
        s.classify_arrival(rid, h)
        s.mark(rid, "completed", category=cat, confidence=0.9, cost_eur=0.002)
        s.save_draft(rid, "en", f"body of {rid}")

    rows_before, summary_before = s.rows(), s.summary()

    # Second run of the same file: same ids, same content.
    for rid, h, _ in batch:
        assert s.classify_arrival(rid, h) == "duplicate"

    assert s.rows() == rows_before
    assert s.summary() == summary_before
    assert s.summary()["cost_eur"] == 0.004  # not paid for again


def test_registering_again_does_not_destroy_the_draft_already_sent(tmp_path):
    """The draft hangs off the request through a foreign key. If register()
    replaced the row instead of ignoring the clash, it would delete it and
    take the already-accepted draft with it."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.save_draft("R1", "en", "original body")
    s.mark("R1", "completed", category="technical")

    s.register("R1", "h1")

    assert s.count_drafts() == 1
    assert s.draft("R1")["body"] == "original body"
    assert s.state("R1") == ("completed", "h1")


def test_registering_again_does_not_wipe_the_cost_already_paid(tmp_path):
    """Cost is a measured deliverable. If re-inserting the same id overwrote
    the row, the spend already declared would go back to zero and the summary
    would understate exactly the figure that has to be accounted for."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.mark("R1", "completed", category="technical", cost_eur=0.002)

    s.register("R1", "h1")

    r = s.summary()
    assert r["cost_eur"] == 0.002
    assert r["completed"] == 1
    assert r["in_progress"] == 0


def test_another_connection_sees_each_step_as_soon_as_it_is_committed(tmp_path):
    """In WAL two processes work on the same file at once. Each step commits
    immediately, so a second process sees progress without waiting for the
    first to finish the request or close the connection."""
    path = str(tmp_path / "t.db")
    writer = Store(path, LOCK_WAIT_MS)
    writer.initialise()
    reader = Store(path, LOCK_WAIT_MS)

    writer.register("R1", "h1")
    assert reader.state("R1") == ("received", "h1")

    writer.mark("R1", "classified", category="technical", cost_eur=0.002)
    assert reader.state("R1") == ("classified", "h1")
    assert reader.summary()["cost_eur"] == 0.002

    writer.close()
    reader.close()


def test_resuming_after_a_crash_does_not_send_a_second_draft(tmp_path):
    """A crash between save_draft() and mark('drafted'): on resume, the draft
    is generated again and may come out different. It is rejected and the one
    already accepted stands. This is not a duplicate in the input file but a
    resume, and the incident type the store offers says so."""
    s = new(tmp_path)
    s.register("R1", "h1")
    s.save_draft("R1", "en", "body already sent")
    # (the process dies here, before marking 'drafted')

    assert s.state("R1") == ("received", "h1")
    assert s.save_draft("R1", "en", "regenerated body, slightly different") is False
    assert s.draft("R1")["body"] == "body already sent"
    assert s.count_drafts() == 1

    # The incident vocabulary cannot call a resume a "duplicate".
    assert "duplicate" not in TYPE_DRAFT_RESUMED


def _dump(path: str) -> str:
    """Full SQL dump of the file, internal tables included.

    rows() and summary() only look at what the store chooses to show. The
    dump sees everything written, which is what someone opening the file
    would see.
    """
    con = sqlite3.connect(path)
    try:
        return "\n".join(con.iterdump())
    finally:
        con.close()


def test_the_full_dump_is_identical_after_repeating_the_run(tmp_path):
    """The same check as above, but against the whole file.

    Both runs deliberately include the only things that write to each table:
    a benign duplicate and a content conflict. Both go through 'ON CONFLICT
    DO NOTHING' paths, which write no row the second time. With AUTOINCREMENT
    that was not enough: SQLite advances sqlite_sequence even when the INSERT
    writes nothing, so the dump changed between a run and its repeat without
    a single piece of data changing. No store accessor would have seen it.
    """
    path = str(tmp_path / "t.db")
    s = Store(path, LOCK_WAIT_MS)
    s.initialise()

    def run():
        # As the process does: the work that costs money is only done if the
        # arrival is new. A duplicate arrival is counted and left alone.
        if s.classify_arrival("R1", "h1") == "new":
            s.mark("R1", "completed", category="technical", confidence=0.9,
                   cost_eur=0.002)
            s.save_draft("R1", "en", "body of R1")
        else:
            s.duplicate("R1")
        s.classify_arrival("R2", "h2")
        s.classify_arrival("R2", "h-different")   # content conflict

    # The first run cannot equal the second: in the first the request is new
    # and in the second it has arrived again, which is a different fact and
    # is recorded. What must stay still is from the second run onwards.
    run()
    run()
    before = _dump(path)
    run()
    assert _dump(path) == before

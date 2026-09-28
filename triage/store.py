"""SQLite persistence and the state machine.

Idempotency is not guaranteed by a check in Python, it is guaranteed by the
database: the primary keys of `requests` and `drafts` are what make it safe to
run an already-processed file again, or to resume a request left half done by
a crash. Re-inserting a known id does not touch the existing row, and a second
draft for the same id is rejected.

State is committed after every step. The intermediate states (`validated`,
`enriched`, `classified`, `drafted`) are not read by this module: they are an
already-committed audit trail that a later stage consults to skip work that
was already done and already paid for. There is no resume logic here; the
only promise this file makes is that the last committed step survives a hard
kill of the process.

The content hash (triage.schemas.record_hash) separates two cases that look
identical at the "same id" level: a benign duplicate (same id, same content)
is ignored; a conflict (same id, different content) is never silently
overwritten. That decision lives in classify_arrival(), inside the store, not
in the caller: a conflict records an incident and does NOT overwrite the
committed state or classification, because the completed work really was
completed and what needs a human is the new content.

The set of states is closed and enforced by a CHECK generated from STATES, so
a typo such as 'compelted' cannot be stored: if it were, the request would not
match TERMINAL_STATES and would be reprocessed (and paid for again) on every
run.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Closed set of states. The schema's CHECK is generated from here, so the
# constant and the schema cannot drift apart.
STATES = (
    "received",
    "validated",
    "enriched",
    "classified",
    "drafted",
    "completed",
    "invalid",
    "escalated",
    "failed",
)

INITIAL_STATE = "received"
TERMINAL_STATES = ("completed", "invalid", "escalated")
# A failed request is not in progress either, but it is not a valid outcome:
# it is counted separately so it is not confused with finished work.
FAILED_STATE = "failed"

# Precision used to report the accumulated cost, kept next to the states so it
# does not hide as a magic number inside summary().
COST_DECIMALS = 6

# Incident types born in this module or that other stages must use by name,
# so the text that ends up in the database is not a loose literal.
TYPE_CONFLICT = "content_conflict"
TYPE_DRAFT_RESUMED = "draft_resumed_not_repeated"

UPDATABLE_FIELDS = (
    "reason", "student_id", "category", "urgency", "confidence",
    "classification_reason", "model_used", "cost_eur",
)

# Cost is added, not replaced: if the process dies after calling the model and
# resumes, the model really is called twice and paid for twice.
ACCUMULATING_FIELDS = ("cost_eur",)

_STATES_SQL = ", ".join(f"'{state}'" for state in STATES)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS requests (
  id                    TEXT PRIMARY KEY NOT NULL,
  record_hash           TEXT NOT NULL,
  state                 TEXT NOT NULL CHECK (state IN ({_STATES_SQL})),
  reason                TEXT,
  student_id            TEXT,
  category              TEXT,
  urgency               INTEGER,
  confidence            REAL,
  classification_reason TEXT,
  model_used            TEXT,
  cost_eur              REAL DEFAULT 0,
  updated_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drafts (
  request_id TEXT PRIMARY KEY REFERENCES requests(id),
  language   TEXT NOT NULL,
  body       TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- No AUTOINCREMENT, for the same reason as in 'duplicates': the ON CONFLICT
-- DO NOTHING of a repeated incident writes no row, but AUTOINCREMENT still
-- advances sqlite_sequence, and the dump stops being identical between a run
-- and its repeat.
CREATE TABLE IF NOT EXISTS incidents (
  id          INTEGER PRIMARY KEY,
  request_id  TEXT,
  type        TEXT NOT NULL,
  detail      TEXT NOT NULL,
  occurred_at TEXT NOT NULL,
  UNIQUE (request_id, type, detail)
);

-- UNIQUE(request_id): "duplicates_ignored" counts distinct requests that
-- have arrived again with the same content, not the number of times they
-- arrived. A third, fourth... appearance of the same id is the same fact
-- observed again, not a new fact each time.
-- No AUTOINCREMENT, as in 'incidents'. No rows are deleted here, so the
-- ordinary rowid is enough.
CREATE TABLE IF NOT EXISTS duplicates (
  id         INTEGER PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE,
  seen_at    TEXT NOT NULL
);
"""


class UnknownRequest(Exception):
    """An update was attempted on a request that is not registered.

    The UPDATE found no row and returned quietly, so a request could walk the
    whole state machine without leaving a trace in the database. This
    module's job is to make silence impossible.
    """

    def __init__(self, request_id):
        super().__init__(f"Request {request_id!r} does not exist in the store.")
        self.request_id = request_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """All of the process's persistence lives behind this class.

    Every method that writes commits before returning. There is no final
    commit when a request finishes: each step is its own transaction, so a
    crash between two steps leaves a queryable state, never a half-written one.

    The lock wait comes in as an argument because its value lives in
    config/config.yaml, not in the code: two processes writing at the same
    time cannot rely on SQLite's defaults.
    """

    def __init__(self, path: str, lock_wait_ms: int):
        self.path = path
        self.lock_wait_ms = int(lock_wait_ms)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(path)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA foreign_keys = ON")
        # WAL: a reader does not block the writer. The wait is set explicitly
        # on every connection; the value comes from the configuration.
        self._con.execute("PRAGMA journal_mode = WAL")
        self._con.execute(f"PRAGMA busy_timeout = {self.lock_wait_ms:d}")

    def initialise(self) -> None:
        self._con.executescript(SCHEMA)
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    def _write(self, sql: str, params: tuple) -> sqlite3.Cursor:
        """Runs a write and commits it. On failure, rolls back and re-raises.

        Nothing is swallowed here: a constraint violation has to reach the
        caller, not vanish and leave the database without the row.
        """
        try:
            cur = self._con.execute(sql, params)
        except sqlite3.Error:
            self._con.rollback()
            raise
        self._con.commit()
        return cur

    def state(self, request_id: str):
        row = self._con.execute(
            "SELECT state, record_hash FROM requests WHERE id = ?", (request_id,)
        ).fetchone()
        return (row["state"], row["record_hash"]) if row else None

    def register(self, request_id: str, rec_hash: str) -> None:
        self._register_if_missing(request_id, rec_hash)

    def _register_if_missing(self, request_id: str, rec_hash: str) -> bool:
        """Inserts the request if it did not exist. Returns True if it inserted.

        ON CONFLICT(id) DO NOTHING rather than INSERT OR IGNORE: only the
        primary-key clash is ignored, which is the expected duplicate. Any
        other violation (a null id, a null hash) blows up instead of making
        the request disappear with neither a row nor an incident.
        """
        cur = self._write(
            "INSERT INTO requests (id, record_hash, state, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
            (request_id, rec_hash, INITIAL_STATE, _now()),
        )
        return cur.rowcount == 1

    def mark(self, request_id: str, state: str, **fields) -> None:
        """Moves an already-registered request to a new state.

        Raises UnknownRequest if the id does not exist: an UPDATE that finds no
        row is not a silent success. Fields are replaced, except the
        accumulating ones (the cost), which are added.
        """
        unknown = set(fields) - set(UPDATABLE_FIELDS)
        if unknown:
            raise ValueError(f"Non-updatable fields: {sorted(unknown)}")
        assignments = []
        for field in fields:
            if field in ACCUMULATING_FIELDS:
                assignments.append(f"{field} = COALESCE({field}, 0) + ?")
            else:
                assignments.append(f"{field} = ?")
        sql = "UPDATE requests SET state = ?, updated_at = ?"
        if assignments:
            sql += ", " + ", ".join(assignments)
        sql += " WHERE id = ?"
        cur = self._write(sql, (state, _now(), *fields.values(), request_id))
        if cur.rowcount == 0:
            raise UnknownRequest(request_id)

    def classify_arrival(self, request_id: str, new_hash: str) -> str:
        """Decides what this arrival is: 'new', 'duplicate' or 'conflict'.

        'new'        — the request does not exist; it is now registered.
        'duplicate'  — same id and same content; no work is repeated.
        'conflict'   — same id, different content; an incident is recorded and
                       the committed state and classification are NOT touched.

        The conflict is idempotent: the UNIQUE constraint on incidents means
        that running the same file again leaves the database identical instead
        of stacking one incident per run.
        """
        stored = self.state(request_id)
        if stored is None:
            if self._register_if_missing(request_id, new_hash):
                return "new"
            stored = self.state(request_id)

        _, stored_hash = stored
        if stored_hash == new_hash:
            return "duplicate"

        self.incident(
            request_id,
            TYPE_CONFLICT,
            f"same id with different content: stored hash {stored_hash}, "
            f"incoming hash {new_hash}",
        )
        return "conflict"

    def save_draft(self, request_id: str, language: str, body: str) -> bool:
        """Returns False if one already existed. Nobody gets two drafts."""
        cur = self._write(
            "INSERT INTO drafts (request_id, language, body, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(request_id) DO NOTHING",
            (request_id, language, body, _now()),
        )
        return cur.rowcount == 1

    def draft(self, request_id: str):
        """Returns the stored draft, or None if there is none."""
        row = self._con.execute(
            "SELECT * FROM drafts WHERE request_id = ?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    def incident(self, request_id, kind: str, detail: str) -> None:
        """Records an incident. The same incident repeated does not stack."""
        self._write(
            "INSERT INTO incidents (request_id, type, detail, occurred_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (request_id, kind, detail, _now()),
        )

    def incidents(self) -> list[dict]:
        return [dict(r) for r in self._con.execute(
            "SELECT * FROM incidents ORDER BY id")]

    def duplicate(self, request_id: str) -> None:
        """Records a benign duplicate: same id, same hash. Does not touch the row.

        `duplicates_ignored` counts distinct requests that have reappeared
        with the same content, not the number of times they reappeared: hence
        the ON CONFLICT DO NOTHING. A second, third... arrival of the same id
        is the same fact observed again, not a new fact, so it cannot push the
        counter up on every repeat of the same file.
        """
        self._write(
            "INSERT INTO duplicates (request_id, seen_at) VALUES (?, ?) "
            "ON CONFLICT(request_id) DO NOTHING",
            (request_id, _now()),
        )

    def row(self, request_id: str):
        """Returns a request's full row, or None if it does not exist."""
        row = self._con.execute(
            "SELECT * FROM requests WHERE id = ?", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_drafts(self) -> int:
        return self._con.execute("SELECT COUNT(*) AS n FROM drafts").fetchone()["n"]

    def rows(self) -> list[dict]:
        return [dict(r) for r in self._con.execute(
            "SELECT * FROM requests ORDER BY id")]

    def escalated(self) -> list[dict]:
        return [dict(r) for r in self._con.execute(
            "SELECT * FROM requests WHERE state = ? ORDER BY id", ("escalated",))]

    def summary(self) -> dict:
        count = {r["state"]: r["n"] for r in self._con.execute(
            "SELECT state, COUNT(*) AS n FROM requests GROUP BY state")}
        by_category = {r["category"]: r["n"] for r in self._con.execute(
            "SELECT category, COUNT(*) AS n FROM requests "
            "WHERE category IS NOT NULL GROUP BY category")}
        by_urgency = {r["urgency"]: r["n"] for r in self._con.execute(
            "SELECT urgency, COUNT(*) AS n FROM requests "
            "WHERE urgency IS NOT NULL GROUP BY urgency")}
        one = lambda sql, p=(): self._con.execute(sql, p).fetchone()["n"]
        return {
            "total": one("SELECT COUNT(*) AS n FROM requests"),
            "completed": count.get("completed", 0),
            "escalated": count.get("escalated", 0),
            "invalid": count.get("invalid", 0),
            "failed": count.get(FAILED_STATE, 0),
            "in_progress": sum(n for s, n in count.items()
                               if s not in TERMINAL_STATES and s != FAILED_STATE),
            "duplicates_ignored": one("SELECT COUNT(*) AS n FROM duplicates"),
            "drafts": self.count_drafts(),
            "by_category": by_category,
            "by_urgency": by_urgency,
            "incidents": one("SELECT COUNT(*) AS n FROM incidents"),
            # A conflict has to be acted on by a person: it cannot be visible
            # only by reading the incidents table.
            "conflicts": one(
                "SELECT COUNT(*) AS n FROM incidents WHERE type = ?", (TYPE_CONFLICT,)),
            "cost_eur": round(self._con.execute(
                "SELECT COALESCE(SUM(cost_eur), 0) AS n FROM requests"
            ).fetchone()["n"], COST_DECIMALS),
        }

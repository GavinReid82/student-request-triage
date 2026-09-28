import csv

import httpx
import respx

from triage.store import TYPE_CONFLICT, Store
from triage.config import Config, Env
from triage.summary import (
    REVIEW_COLUMNS,
    RESULTS_COLUMNS,
    provider_cost,
    cost_difference,
    export_csv,
    format_cost,
    format_summary,
)

# Values deliberately different from the real ones in config/config.yaml and
# from every default (0.75, sim-fast/sim-precise, 10/3/0.5, 5000/30.0): a
# hardcode of any of them in summary.py would show up here as a failing test.
TEMPLATES = {c: {"ca": "Hola {name}."} for c in ("content", "technical")}
CFG = Config(
    categories=("content", "technical"),
    confidence_threshold=0.55,
    fast_model="mod-x",
    precise_model="mod-y",
    http_timeout_s=2.2,
    http_retries=4,
    http_backoff_base_s=0.1,
    draft_engine="template",
    templates=TEMPLATES,
    db_lock_wait_ms=9999,
    http_max_wait_s=12.0,
)
ENV = Env("tok-x", "http://api-x", "http://mod-x", "r.json", "t.db", "output")


def _store(tmp_path):
    st = Store(str(tmp_path / "t.db"), CFG.db_lock_wait_ms)
    st.initialise()
    return st


def _value_of(text: str, label: str) -> str:
    """Returns what follows the label on its own line, or fails."""
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith(label):
            return clean[len(label):].lstrip(": ").strip()
    raise AssertionError(f"label {label!r} does not appear in:\n{text}")


def _line_starting_with(text: str, prefix: str) -> str:
    """Returns the full (trimmed) line that starts with `prefix`."""
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith(prefix):
            return clean
    raise AssertionError(f"no line starts with {prefix!r} in:\n{text}")


def test_each_summary_value_appears_on_its_own_label_line():
    """A swap between two fields has to fail: every value is different and is
    looked up on ITS label's line, not in the whole text."""
    r = {
        "total": 97, "completed": 61, "escalated": 13, "invalid": 7,
        "failed": 2, "in_progress": 4, "duplicates_ignored": 5,
        "conflicts": 3, "drafts": 59, "incidents": 23,
        "cost_eur": 0.045678,
        "by_category": {"technical": 31, "content": 19, "admin": 11},
        "by_urgency": {3: 17, 2: 29, 1: 43},
    }
    text = format_summary(r)
    labels = {
        "Processed": r["total"],
        "completed": r["completed"],
        "escalated": r["escalated"],
        "invalid": r["invalid"],
        "failed": r["failed"],
        "Duplicates ignored": r["duplicates_ignored"],
        "Conflicts": r["conflicts"],
        "In progress": r["in_progress"],
        "Drafts generated": r["drafts"],
        "Incidents recorded": r["incidents"],
    }
    assert len(labels) == 10
    for label, value in labels.items():
        assert str(value) in _value_of(text, label)
    for category, n in r["by_category"].items():
        assert str(n) == _value_of(text, category)
    for urgency, n in r["by_urgency"].items():
        assert str(n) == _value_of(text, f"urgency {urgency}")
    assert str(r["cost_eur"]) in text


def test_exports_two_csvs_with_exact_headers_and_overwrites_on_re_export(tmp_path):
    st = _store(tmp_path)
    st.register("R1", "h1")
    st.mark("R1", "completed", category="technical")
    st.register("R2", "h2")
    st.mark("R2", "escalated", reason="low confidence")

    output_dir = str(tmp_path / "output")
    paths = export_csv(st, output_dir)
    assert len(paths) == 2
    assert paths[0].endswith("results.csv")
    assert paths[1].endswith("review_queue.csv")

    with open(paths[0], encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = tuple(next(reader))
        rows = list(reader)
    assert header == RESULTS_COLUMNS
    assert len(rows) == 2

    with open(paths[1], encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        review_header = tuple(next(reader))
        review_rows = list(reader)
    assert review_header == REVIEW_COLUMNS
    assert len(review_rows) == 1

    # Re-exporting overwrites: it does not duplicate rows by appending.
    export_csv(st, output_dir)
    with open(paths[0], encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_review_queue_includes_requests_with_a_content_conflict(tmp_path):
    st = _store(tmp_path)
    st.register("R1", "h1")
    st.mark("R1", "completed", category="technical")
    st.register("R2", "h2")
    st.mark("R2", "escalated", reason="low confidence")
    # An incident unrelated to the conflict cannot leak into the reason of a
    # real escalation.
    st.incident("R2", "alternative_date_format", "received_at normalised")
    st.register("R3", "h3")
    st.mark("R3", "completed", category="technical")
    assert st.classify_arrival("R3", "h3-different-content") == "conflict"
    # R4 is completed with no conflict: ANY incident is not enough to put it
    # in the human review queue.
    st.register("R4", "h4")
    st.mark("R4", "completed", category="technical")
    st.incident("R4", "alternative_date_format", "received_at normalised")

    paths = export_csv(st, str(tmp_path / "output"))
    with open(paths[1], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert {r["id"] for r in rows} == {"R2", "R3"}
    conflict_row = next(r for r in rows if r["id"] == "R3")
    assert TYPE_CONFLICT in conflict_row["review_reason"]
    escalated_row = next(r for r in rows if r["id"] == "R2")
    assert escalated_row["review_reason"] == "low confidence"


@respx.mock
def test_provider_cost_gets_stats_with_the_token():
    route = respx.get("http://mod-x/v1/stats").mock(return_value=httpx.Response(
        200, json={"total_calls": 9, "total_cost_eur": 0.1, "by_model": {}}))
    with httpx.Client() as c:
        stats = provider_cost(c, ENV, CFG, sleep=lambda s: None)
    assert stats == {"total_calls": 9, "total_cost_eur": 0.1, "by_model": {}}
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {ENV.token}"


@respx.mock
def test_provider_cost_bounds_the_wait_with_the_configuration():
    """Retry-After asks for 9999s; the real cap is http_max_wait_s (12.0),
    not the default (30.0) nor any other hand-picked cap."""
    route = respx.get("http://mod-x/v1/stats")
    route.side_effect = [
        httpx.Response(503, headers={"Retry-After": "9999"}),
        httpx.Response(200, json={"total_calls": 0, "total_cost_eur": 0.0,
                                  "by_model": {}}),
    ]
    waits = []
    with httpx.Client() as c:
        provider_cost(c, ENV, CFG, sleep=waits.append)
    assert waits == [CFG.http_max_wait_s]


def test_cost_difference_subtracts_the_two_snapshots_rather_than_repeating_them():
    """Costs on the order of €1e-5, like the real per-call ones: with only 2
    decimals of precision the difference would round to 0.0 and the whole
    cost of the run would vanish. The values also carry floating-point noise
    (0.000037 - 0.000012 is not exactly 0.000025 in binary) so the rounding
    to 6 decimals is really exercised, not just copied."""
    before = {
        "total_calls": 12, "total_cost_eur": 0.000012,
        "by_model": {"model-alpha": {"calls": 10, "cost_eur": 0.000012}},
    }
    after = {
        "total_calls": 20, "total_cost_eur": 0.000037,
        "by_model": {
            "model-alpha": {"calls": 15, "cost_eur": 0.000037},
            "model-beta": {"calls": 5, "cost_eur": 0.000025},
        },
    }
    d = cost_difference(before, after)
    assert d["total_calls"] == 8
    assert d["total_cost_eur"] == 0.000025
    assert d["by_model"]["model-alpha"]["calls"] == 5
    assert d["by_model"]["model-alpha"]["cost_eur"] == 0.000025
    # model-beta did not exist "before": it counts from zero rather than crashing.
    assert d["by_model"]["model-beta"]["calls"] == 5
    assert d["by_model"]["model-beta"]["cost_eur"] == 0.000025


def test_format_cost_shows_the_per_thousand_division():
    run = {
        "total_calls": 12, "total_cost_eur": 0.0123,
        "by_model": {
            "model-alpha": {"calls": 7, "cost_eur": 0.008},
            "model-beta": {"calls": 5, "cost_eur": 0.0043},
        },
    }
    text = format_cost(run, processed=40, recorded_cost_eur=0.0456)
    assert "0.012300" in text
    assert "40" in text
    assert "0.3075" in text
    assert "0.0456" in text
    # Two models with different costs: each line carries ITS own cost, not
    # the run's total repeated on both.
    alpha_line = _line_starting_with(text, "model-alpha")
    beta_line = _line_starting_with(text, "model-beta")
    assert "cost=€0.008000" in alpha_line
    assert "cost=€0.004300" in beta_line
    assert "cost=€0.008000" not in beta_line
    assert "cost=€0.004300" not in alpha_line


def test_format_cost_with_nothing_processed_does_not_divide():
    run = {"total_calls": 0, "total_cost_eur": 0.02, "by_model": {}}
    text = format_cost(run, processed=0, recorded_cost_eur=0.09)
    assert "0.020000" in text
    assert "1,000" not in text


def test_review_queue_puts_what_could_not_be_judged_first_then_the_most_urgent(tmp_path):
    st = _store(tmp_path)
    for rid, urgency in (("R1", 1), ("R2", 3), ("R3", None), ("R4", 2)):
        st.register(rid, f"h-{rid}")
        fields = {"urgency": urgency} if urgency is not None else {}
        st.mark(rid, "escalated", reason=f"reason {rid}", **fields)

    paths = export_csv(st, str(tmp_path / "output"))
    with open(paths[1], encoding="utf-8-sig") as f:
        order = [row["id"] for row in csv.DictReader(f)]
    assert order == ["R3", "R2", "R4", "R1"]

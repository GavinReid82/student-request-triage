"""Final summary, CSV export and the cost measured by the provider.

The CSVs exist so student services staff can look at the human-review queue
without knowing SQL. SQLite is still the source of truth: idempotency depends
on a primary key and a transaction, and a spreadsheet offers neither.

The model service's `/v1/stats` accumulates from the moment the service
started, not from the start of this run. A second run over an
already-processed file does not call the model once, so showing the whole
running total as if this run had caused it would misreport the one figure
that most needs to be honest. That is why this module separates taking a
snapshot (`provider_cost`) from subtracting two snapshots (`cost_difference`):
the cost of the run is always that difference, never the raw total.
"""
import csv
import os
import time

from .store import COST_DECIMALS, TYPE_CONFLICT
from .http import send

# The CSVs are read by student services, not an engineer: only what helps
# them act. Hash, model, cost and timestamps stay in SQLite, the source of
# truth; here they would be noise.
RESULTS_COLUMNS = ("id", "state", "student_id", "category", "urgency",
                   "confidence", "classification_reason", "reason")
# The review queue: a single reason (why a person has to look at it) instead
# of two columns that almost always repeat each other.
REVIEW_COLUMNS = ("id", "student_id", "urgency", "category", "confidence",
                  "review_reason")


def format_summary(r: dict) -> str:
    """Text of the run's final summary.

    "Processed" shows the total of registered requests and, below it, their
    outcome (completed, escalated, invalid, failed): the word alone is
    ambiguous, so both readings are given. A content conflict and a request
    still in progress are also shown separately: both need someone to look at
    them, and neither is visible from the incidents alone.
    """
    categories = "\n".join(
        f"    {cat:<20}: {n}" for cat, n in sorted(r["by_category"].items())
    ) or "    (none)"
    urgencies = "\n".join(
        f"    {'urgency ' + str(u):<20}: {n}"
        for u, n in sorted(r["by_urgency"].items(), reverse=True)
    ) or "    (none)"
    return (
        "\n===== RUN SUMMARY =====\n"
        f"  Processed               : {r['total']}\n"
        f"    completed             : {r['completed']}\n"
        f"    escalated             : {r['escalated']}\n"
        f"    invalid               : {r['invalid']}\n"
        f"    failed                : {r['failed']}\n"
        f"  Duplicates ignored      : {r['duplicates_ignored']}\n"
        f"  Conflicts (need human review): {r['conflicts']}\n"
        f"  In progress             : {r['in_progress']}\n"
        f"  Drafts generated        : {r['drafts']}\n"
        f"  Incidents recorded      : {r['incidents']}\n"
        f"  Recorded cost (€)       : {r['cost_eur']}\n"
        "  By category:\n"
        f"{categories}\n"
        "  By urgency (3 = most urgent):\n"
        f"{urgencies}\n"
        "=================================\n"
    )


def _review_queue(store) -> list[dict]:
    """One row per request that needs a person to look at it.

    `store.escalated()` only sees state='escalated'. A content conflict
    leaves the original state intact (the work done is still valid) and only
    records an incident, so without this the human queue would silently lose
    every conflict.
    """
    details_by_id: dict[str, list[str]] = {}
    for incident in store.incidents():
        if incident["type"] == TYPE_CONFLICT:
            details_by_id.setdefault(incident["request_id"], []).append(
                incident["detail"])

    queue = []
    for row in store.rows():
        reasons = []
        if row["state"] == "escalated":
            reasons.append(row["reason"] or "")
        for detail in details_by_id.get(row["id"], ()):
            reasons.append(f"{TYPE_CONFLICT}: {detail}")
        if reasons:
            queue.append({**row, "review_reason": "; ".join(reasons)})
    # Most urgent first. Rows with no urgency go at the very top: the system
    # could not judge them (the model returned no valid classification), so a
    # person has to read them before anyone knows whether they are urgent.
    queue.sort(key=lambda r: (r["urgency"] is not None, -(r["urgency"] or 0), r["id"]))
    return queue


def export_csv(store, output_dir: str) -> list[str]:
    """Writes results.csv and review_queue.csv in `output_dir` and returns their paths.

    Every call opens the files in write mode: a re-export replaces the
    previous content, it never piles on top of it.
    """
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.csv")
    with open(results_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(store.rows())

    review_path = os.path.join(output_dir, "review_queue.csv")
    with open(review_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_review_queue(store))

    return [results_path, review_path]


def provider_cost(client, env, cfg, sleep=time.sleep) -> dict:
    """A snapshot of the usage and cost accumulated, according to the provider.

    It is not the cost of this run: it is what the service has accumulated
    since it started. Two snapshots (before and after) and `cost_difference`
    give the figure that does belong to this run.
    """
    response = send(
        client, "GET", f"{env.model_url}/v1/stats",
        headers={"Authorization": f"Bearer {env.token}"},
        timeout_s=cfg.http_timeout_s,
        retries=cfg.http_retries,
        backoff_base_s=cfg.http_backoff_base_s,
        max_wait_s=cfg.http_max_wait_s,
        sleep=sleep,
    )
    return response.json()


def cost_difference(before: dict, after: dict) -> dict:
    """What changed between two `/v1/stats` snapshots: the cost of this run.

    A model missing from `before` (never used when that snapshot was taken)
    counts from zero rather than crashing. Each snapshot's `detail` is ignored
    here: it has already been paid for, and what matters for this purpose is
    the aggregate.
    """
    models_before = before.get("by_model", {})
    models_after = after.get("by_model", {})
    by_model = {}
    for model, data in models_after.items():
        base = models_before.get(model, {"calls": 0, "cost_eur": 0.0})
        by_model[model] = {
            "calls": data.get("calls", 0) - base.get("calls", 0),
            "cost_eur": round(
                data.get("cost_eur", 0.0) - base.get("cost_eur", 0.0),
                COST_DECIMALS),
        }
    return {
        "total_calls": (
            after.get("total_calls", 0) - before.get("total_calls", 0)),
        "total_cost_eur": round(
            after.get("total_cost_eur", 0.0) - before.get("total_cost_eur", 0.0),
            COST_DECIMALS),
        "by_model": by_model,
    }


def format_cost(run: dict, processed: int, recorded_cost_eur: float) -> str:
    """What this run cost, with the arithmetic shown.

    `run` is the output of `cost_difference`: what the provider measured
    between the start and the end of this run, not its historical total.
    `processed` is the number of requests this run actually handled (new or
    resumed), not duplicates or conflicts, which triggered no new model call.
    """
    total = round(run.get("total_cost_eur", 0.0), COST_DECIMALS)
    calls = run.get("total_calls", 0)
    lines = ["", "===== COST OF THIS RUN ====="]
    lines.append(f"  Model calls in this run            : {calls}")
    lines.append(f"  Cost measured by the provider (€)  : {total:.6f}")
    if processed:
        per_thousand = round(total / processed * 1000, 4)
        lines.append(
            f"  Cost per 1,000 requests: €{total:.6f} / {processed} "
            f"requests × 1,000 = €{per_thousand}")
    else:
        lines.append(
            "  This run handled no new requests; there is no denominator for "
            "a cost per thousand. The cost measured over the interval was "
            f"€{total:.6f}.")
    lines.append("  By model in this run:")
    for model, data in sorted(run.get("by_model", {}).items()):
        model_cost = round(data.get("cost_eur", 0.0), COST_DECIMALS)
        lines.append(
            f"    {model:<14} calls={data.get('calls', 0):<4} "
            f"cost=€{model_cost:.6f}")
    lines.append(
        "  Our records vs the provider's meter (€): "
        f"{recorded_cost_eur} (our accounting) / {total:.6f} (provider)")
    lines.append("=================================\n")
    return "\n".join(lines)

#!/usr/bin/env bash
# Measures the real cost of three model strategies over data/requests.json.
#
# Each strategy starts from an empty database. The provider accumulates its
# statistics from the moment it starts, but the pipeline already reports only
# the difference for its own run, so there is no need to restart the services.
#
# config/config.yaml is not touched: each strategy writes its own copy into
# output/ (mounted read-write) and passes it to the container through
# TRIAGE_CONFIG_PATH.
#
# The single-model strategies use the same model on both tiers and threshold
# 0.0, so each request makes exactly one call to that model (plus a retry when
# the output is invalid). The number of requests that need a person is not
# read from the run (with threshold 0.0 nothing escalates for low
# confidence): it is counted afterwards from the stored confidences, against
# the real threshold in config.yaml, plus the requests escalated because no
# model gave a valid output.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
DB="output/triage.db"
REAL_THRESHOLD=$(grep '^confidence_threshold:' config/config.yaml | awk '{print $2}')

measure () {
  local name="$1" fast="$2" precise="$3" threshold="$4"
  local cfg="output/config_${name}.yaml"
  echo
  echo "=============== ${name} (fast=${fast}, precise=${precise}, threshold=${threshold}) ==============="
  rm -f "$DB" "$DB-wal" "$DB-shm"
  sed -e "s/^confidence_threshold: .*/confidence_threshold: ${threshold}/" \
      -e "s/^  fast: .*/  fast: ${fast}/" \
      -e "s/^  precise: .*/  precise: ${precise}/" \
      config/config.yaml > "$cfg"
  docker compose run --rm -e TRIAGE_CONFIG_PATH="/app/${cfg}" triage 2>&1 \
    | sed -n '/===== RUN SUMMARY/,$p'
  "$PY" - "$DB" "$REAL_THRESHOLD" "$name" <<'PYEOF'
import sqlite3, sys
db, threshold, name = sys.argv[1], float(sys.argv[2]), sys.argv[3]
con = sqlite3.connect(db)
low = con.execute(
    "SELECT COUNT(*) FROM requests WHERE confidence IS NOT NULL AND confidence < ?",
    (threshold,)).fetchone()[0]
no_output = con.execute(
    "SELECT COUNT(*) FROM requests WHERE state = 'escalated' AND confidence IS NULL"
).fetchone()[0]
print(f"  [{name}] below {threshold}: {low}; no valid model output: {no_output}; "
      f"would need a person: {low + no_output}")
PYEOF
}

measure "all_fast"    sim-fast    sim-fast    0.0
measure "all_precise" sim-precise sim-precise 0.0
measure "two_tier"    sim-fast    sim-precise "$REAL_THRESHOLD"

docker compose down >/dev/null 2>&1 || true

"""Entry point.

Waits for the external services to respond, runs the full pass, exports the
CSVs and prints the summary and the measured cost of this run.

The cost this module shows is never the raw running total from `/v1/stats`
(that has been counting since the model service started, not since this run
began): it is the difference between a snapshot taken before `process_all`
and another taken after. A failure taking either snapshot must not bring the
run down or hide it: it is logged as a WARNING and the run carries on without
the cost block.
"""
import logging
import os
import sys
import time

import httpx

from .store import Store
from .students import StudentClient
from .classifier import Classifier
from .config import ConfigError, load_config, load_env
from .pipeline import process_all
from .summary import (
    cost_difference, export_csv, format_cost, format_summary, provider_cost,
)

logging.basicConfig(
    level=os.environ.get("TRIAGE_LOG", "INFO"),
    format="%(asctime)s %(levelname)-7s %(message)s",
)
log = logging.getLogger("triage")

# On first start, this container and the two external services install their
# dependencies with three concurrent `pip install`s: that can take much longer
# than the few seconds an already-installed `uvicorn` needs to come up. 120
# attempts at 1 s intervals leave plenty of room for that cold start without
# slowing down the normal case, where `/health` answers almost immediately.
SERVICE_WAIT_ATTEMPTS = 120
SERVICE_WAIT_INTERVAL_S = 1.0


def wait_for(client, url: str, attempts: int = SERVICE_WAIT_ATTEMPTS,
             interval_s: float = SERVICE_WAIT_INTERVAL_S,
             sleep=time.sleep) -> bool:
    """Polls `{url}/health` until it answers 200 or the attempts run out.

    Returns True or False; it does not raise. What to do with a service that
    never answered, and which exit code the process ends with, is decided by
    `main`, not by this function.
    """
    for attempt in range(attempts):
        try:
            if client.get(f"{url}/health", timeout=2).status_code == 200:
                log.info("%s available (attempt %d/%d)", url, attempt + 1, attempts)
                return True
        except httpx.HTTPError:
            pass
        sleep(interval_s)
    return False


def main(sleep=time.sleep) -> int:
    """Runs one full pass. Returns the process exit code.

    0 the run reached the end (even if some requests ended up `failed`:
      they are recorded and the next run picks them up).
    2 invalid or incomplete configuration.
    3 a required external service never answered `/health`.
    1 an exception escaped `process_all`; by design, only a failure of the
      store itself does that, because every other failure is already caught
      request by request inside the pipeline.
    """
    try:
        env = load_env()
        cfg = load_config(os.environ.get("TRIAGE_CONFIG_PATH", "config/config.yaml"))
    except ConfigError as e:
        log.error("Invalid configuration: %s", e)
        return 2

    store = Store(env.db_path, cfg.db_lock_wait_ms)
    store.initialise()

    try:
        with httpx.Client() as client:
            for url in (env.student_api_url, env.model_url):
                if not wait_for(client, url, sleep=sleep):
                    log.error(
                        "%s did not answer /health after %d attempts.",
                        url, SERVICE_WAIT_ATTEMPTS)
                    return 3

            # Production calls StudentClient and Classifier with their default
            # `sleep` (the real time.sleep): `main`'s injectable `sleep` is only
            # for the /health polling, which is the only thing the tests need
            # to not really sleep.
            student_client = StudentClient(client, env, cfg, store)
            classifier = Classifier(client, env, cfg, store)

            before = None
            try:
                before = provider_cost(client, env, cfg)
            except Exception as e:
                log.warning(
                    "Could not take the cost snapshot before the run; "
                    "this run will have no measured cost: %s", e)

            try:
                start = time.monotonic()
                summary = process_all(
                    env.requests_path, store, student_client, classifier, cfg)
                duration = time.monotonic() - start
            except Exception:
                log.exception(
                    "The run was interrupted by an unexpected store failure.")
                return 1

            paths = export_csv(store, env.output_dir)

            print(format_summary(summary))
            print(f"  Run duration: {duration:.1f} s")
            print(f"  Student API calls: {student_client.calls}")
            print("  Exported to: " + ", ".join(paths))

            if summary["in_progress"] > 0 or summary["failed"] > 0:
                log.warning(
                    "%d requests still in progress and %d failed; the next "
                    "run will pick them up.", summary["in_progress"], summary["failed"])

            if before is None:
                log.warning(
                    "No cost snapshot from before the run; the cost block "
                    "is skipped.")
            else:
                try:
                    after = provider_cost(client, env, cfg)
                except Exception as e:
                    log.warning(
                        "Could not take the cost snapshot after the run; "
                        "this run will have no measured cost: %s", e)
                else:
                    print(format_cost(
                        cost_difference(before, after),
                        summary["processed_this_run"], summary["cost_eur"]))
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

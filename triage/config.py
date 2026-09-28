"""Loading of configuration and environment variables.

Business configuration (categories, thresholds, templates) lives in YAML.
Secrets and paths live in the environment. None of it is in the code.
"""
import os
from dataclasses import dataclass
import yaml


class ConfigError(Exception):
    """The configuration is invalid or incomplete."""


# Default wait on an SQLite write lock, in milliseconds. It applies when the
# YAML has no 'db' block.
DEFAULT_DB_LOCK_WAIT_MS = 5000

# Cap, in seconds, on how long the HTTP client sleeps between attempts. It
# applies when the YAML has no 'http.max_wait_s' key.
DEFAULT_HTTP_MAX_WAIT_S = 30.0


@dataclass(frozen=True)
class Config:
    categories: tuple[str, ...]
    confidence_threshold: float
    fast_model: str
    precise_model: str
    http_timeout_s: float
    http_retries: int
    http_backoff_base_s: float
    draft_engine: str
    templates: dict[str, dict[str, str]]
    # New fields ALWAYS go at the end: some tests build Config by position
    # and anywhere else would break those callers.
    db_lock_wait_ms: int
    http_max_wait_s: float


@dataclass(frozen=True)
class Env:
    token: str
    student_api_url: str
    model_url: str
    requests_path: str
    db_path: str
    output_dir: str


def load_config(path: str) -> Config:
    try:
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
    except OSError as e:
        raise ConfigError(f"Could not read the configuration at {path}: {e}") from e

    try:
        cfg = Config(
            categories=tuple(d["categories"]),
            confidence_threshold=float(d["confidence_threshold"]),
            fast_model=d["models"]["fast"],
            precise_model=d["models"]["precise"],
            http_timeout_s=float(d["http"]["timeout_s"]),
            http_retries=int(d["http"]["retries"]),
            http_backoff_base_s=float(d["http"]["backoff_base_s"]),
            draft_engine=d["draft"]["engine"],
            templates=d["templates"],
            # With a default: a YAML written before this block still loads.
            db_lock_wait_ms=int(
                (d.get("db") or {}).get("lock_wait_ms", DEFAULT_DB_LOCK_WAIT_MS)
            ),
            http_max_wait_s=float(
                d["http"].get("max_wait_s", DEFAULT_HTTP_MAX_WAIT_S)
            ),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"Incomplete or malformed configuration: {e}") from e

    # Values the YAML allows you to write but that would leave the HTTP client
    # with no possible behaviour. They are rejected here, at load time, not in
    # the middle of a request.
    if cfg.http_retries < 1:
        raise ConfigError(
            f"http.retries must be 1 or more, and it is {cfg.http_retries}: "
            "with 0 no request would ever be made."
        )
    if cfg.http_max_wait_s <= 0:
        raise ConfigError(
            f"http.max_wait_s must be greater than 0, and it is "
            f"{cfg.http_max_wait_s}: with no room to wait, no retry is useful."
        )
    return cfg


def _var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"Missing environment variable {name}. See .env.example.")
    return value


def load_env() -> Env:
    return Env(
        token=_var("TRIAGE_TOKEN"),
        student_api_url=_var("TRIAGE_STUDENT_API_URL"),
        model_url=_var("TRIAGE_MODEL_URL"),
        requests_path=_var("TRIAGE_REQUESTS_PATH"),
        db_path=_var("TRIAGE_DB_PATH"),
        output_dir=_var("TRIAGE_OUTPUT_DIR"),
    )

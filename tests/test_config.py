import pytest
from triage.config import load_config, load_env, ConfigError

def test_loads_threshold_and_categories(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "categories: [content, technical]\n"
        "confidence_threshold: 0.75\n"
        "models: {fast: sim-fast, precise: sim-precise}\n"
        "http: {timeout_s: 10, retries: 3, backoff_base_s: 0.5}\n"
        "draft: {engine: template}\n"
        "templates:\n"
        "  content: {en: 'hi {name}'}\n",
        encoding="utf-8",
    )
    cfg = load_config(str(path))
    assert cfg.categories == ("content", "technical")
    assert cfg.confidence_threshold == 0.75
    assert cfg.http_retries == 3
    assert cfg.templates["content"]["en"] == "hi {name}"
    # With no 'db' block the default applies: an older YAML still loads.
    assert cfg.db_lock_wait_ms == 5000
    # The same for the http wait cap: a YAML without the key still loads.
    assert cfg.http_max_wait_s == 30.0

def test_loads_the_database_lock_wait(tmp_path):
    """The wait on an SQLite lock is configuration, not code: the store
    receives it as an argument and its value comes from here."""
    path = tmp_path / "c.yaml"
    path.write_text(
        "categories: [content]\n"
        "confidence_threshold: 0.75\n"
        "models: {fast: sim-fast, precise: sim-precise}\n"
        "http: {timeout_s: 10, retries: 3, backoff_base_s: 0.5}\n"
        "draft: {engine: template}\n"
        "db: {lock_wait_ms: 7000}\n"
        "templates:\n"
        "  content: {en: 'hi {name}'}\n",
        encoding="utf-8",
    )
    assert load_config(str(path)).db_lock_wait_ms == 7000

def test_incomplete_environment_fails(monkeypatch):
    monkeypatch.delenv("TRIAGE_TOKEN", raising=False)
    with pytest.raises(ConfigError):
        load_env()

def _yaml_with(http: str) -> str:
    return (
        "categories: [content]\n"
        "confidence_threshold: 0.75\n"
        "models: {fast: sim-fast, precise: sim-precise}\n"
        f"http: {http}\n"
        "draft: {engine: template}\n"
        "templates:\n"
        "  content: {en: 'hi {name}'}\n"
    )

def test_zero_retries_is_a_configuration_error(tmp_path):
    """0 looks like a legal value in the YAML but leaves `send` with no
    attempt at all: it has to fail as configuration, not blow up in the
    middle of a request."""
    path = tmp_path / "c.yaml"
    path.write_text(
        _yaml_with("{timeout_s: 10, retries: 0, backoff_base_s: 0.5}"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(str(path))

def test_non_positive_max_wait_is_a_configuration_error(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        _yaml_with(
            "{timeout_s: 10, retries: 3, backoff_base_s: 0.5, max_wait_s: 0}"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(str(path))

def test_loads_max_wait_from_the_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        _yaml_with(
            "{timeout_s: 10, retries: 3, backoff_base_s: 0.5, max_wait_s: 45}"
        ),
        encoding="utf-8",
    )
    assert load_config(str(path)).http_max_wait_s == 45.0

"""Compression with no timeout override allows 900s, including on an isolated profile.

Hinata-style profiles load DEFAULT_CONFIG plus their own config.yaml and do not
inherit another home's ``auxiliary.compression`` keys. The built-in request
budget and no-progress window are both 900s. An explicit shorter value is kept.
Other auxiliary tasks stay on their own timeouts and the shared 60s no-progress
default. No live API calls and no real 900s wait — the Codex guard is stopped
before it blocks.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from agent.auxiliary_client import (
    _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS,
    _COMPRESSION_BUILTIN_NO_PROGRESS_TIMEOUT_SECONDS,
    _COMPRESSION_BUILTIN_TIMEOUT_SECONDS,
    _CodexStreamGuard,
    _effective_aux_timeout,
    _get_task_no_progress_timeout,
    _get_task_timeout,
    call_llm,
)
from agent.conversation_compression import resolve_context_compression_timeouts
from hermes_cli.config_defaults import DEFAULT_CONFIG


def _write_home(home, body: str) -> None:
    home.mkdir()
    (home / "config.yaml").write_text(body)


def _guard_budgets(monkeypatch, task: str) -> tuple[float, float]:
    """(no_progress, total) the Codex stream guard would arm for one aux call."""
    from agent.auxiliary_client import CodexAuxiliaryClient

    captured: dict[str, float] = {}

    class _Stop(Exception):
        pass

    def _start(self):
        captured["window"] = self.no_progress_timeout
        captured["total"] = self.total_timeout
        raise _Stop()

    real_client = SimpleNamespace(
        api_key="k", base_url="https://chatgpt.com/backend-api/codex/", close=lambda: None,
        responses=SimpleNamespace(create=lambda **kw: None),
    )
    client = CodexAuxiliaryClient(real_client, "gpt-5.6-sol")
    monkeypatch.setattr(_CodexStreamGuard, "start", _start)
    monkeypatch.setattr(
        "agent.auxiliary_client._get_cached_client", lambda *a, **k: (client, "gpt-5.6-sol"),
    )
    with pytest.raises(_Stop):
        call_llm(
            task=task, provider="openai-codex", model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "summarize"}],
        )
    return captured["window"], captured["total"]


def test_schema_builtin_matches_the_runtime_constants():
    compression = DEFAULT_CONFIG["auxiliary"]["compression"]
    assert compression["timeout"] == _COMPRESSION_BUILTIN_TIMEOUT_SECONDS == 900
    assert compression["no_progress_timeout"] == _COMPRESSION_BUILTIN_NO_PROGRESS_TIMEOUT_SECONDS == 900
    assert DEFAULT_CONFIG["auxiliary"]["vision"]["timeout"] == 120
    assert DEFAULT_CONFIG["auxiliary"]["title_generation"]["timeout"] == 30
    assert "no_progress_timeout" not in DEFAULT_CONFIG["auxiliary"]["vision"]
    assert "no_progress_timeout" not in DEFAULT_CONFIG["auxiliary"]["title_generation"]


def test_isolated_profile_without_compression_keys_allows_900s(tmp_path, monkeypatch):
    """No auxiliary.compression block: Codex no-progress and the request budget are 900s."""
    home = tmp_path / "hinata"
    _write_home(home, "model:\n  provider: openai-codex\n  default: gpt-5\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert _get_task_timeout("compression") == 900
    assert _effective_aux_timeout("compression", None) == 900
    assert _get_task_no_progress_timeout("compression") == 900
    assert resolve_context_compression_timeouts() == (900.0, 900.0)

    window, total = _guard_budgets(monkeypatch, "compression")
    assert window == 900
    assert total == 900

    guard = _CodexStreamGuard(None, total, no_progress_timeout=window)
    assert guard.no_progress_timeout == 900
    assert guard.total_timeout == 900
    assert guard.hard_deadline - guard._start >= 900


def test_partial_auxiliary_block_still_uses_the_compression_builtin(tmp_path, monkeypatch):
    """A profile that configures vision but not compression does not inherit a short budget."""
    home = tmp_path / "profile"
    _write_home(
        home,
        "auxiliary:\n  vision:\n    timeout: 15\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert _effective_aux_timeout("compression", None) == 900
    assert _get_task_no_progress_timeout("compression") == 900
    assert _effective_aux_timeout("vision", None) == 15
    window, total = _guard_budgets(monkeypatch, "compression")
    assert (window, total) == (900, 900)


def test_explicit_shorter_compression_override_is_kept(tmp_path, monkeypatch):
    home = tmp_path / "short"
    _write_home(
        home,
        "auxiliary:\n  compression:\n    timeout: 100\n    no_progress_timeout: 40\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert _effective_aux_timeout("compression", None) == 100
    assert _get_task_no_progress_timeout("compression") == 40
    # Below the historical 300s floor, and not replaced by the 900s built-in.
    window, total = _guard_budgets(monkeypatch, "compression")
    assert total == 100
    assert window == 40


def test_shorter_timeout_caps_the_builtin_no_progress_window(tmp_path, monkeypatch):
    """no_progress stays at the built-in until the explicit request budget caps it."""
    home = tmp_path / "cap"
    _write_home(home, "auxiliary:\n  compression:\n    timeout: 100\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert _get_task_no_progress_timeout("compression") == 900
    window, total = _guard_budgets(monkeypatch, "compression")
    assert total == 100
    assert window == 100


def test_per_call_timeout_is_not_replaced_by_the_builtin():
    assert _effective_aux_timeout("compression", 25) == 25


def test_other_auxiliary_tasks_keep_their_budgets(tmp_path, monkeypatch):
    home = tmp_path / "other"
    _write_home(home, "model:\n  provider: openai-codex\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS == 60
    assert _get_task_no_progress_timeout("vision") is None
    assert _get_task_no_progress_timeout("title_generation") is None
    assert _get_task_no_progress_timeout("approval") is None
    assert _effective_aux_timeout("vision", None) == 120
    assert _effective_aux_timeout("title_generation", None) == 30
    assert _effective_aux_timeout("approval", None) == 30

    window, total = _guard_budgets(monkeypatch, "title_generation")
    assert total == 30
    assert window == 30  # shared 60s window, capped at this task's 30s timeout — not 900


def test_invalid_compression_no_progress_warns_and_uses_900(tmp_path, monkeypatch, caplog):
    home = tmp_path / "bad"
    _write_home(
        home,
        "auxiliary:\n  compression:\n    timeout: 2000\n    no_progress_timeout: -3\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
        assert _get_task_no_progress_timeout("compression") == 900
    assert any("no_progress_timeout" in r.getMessage() for r in caplog.records)

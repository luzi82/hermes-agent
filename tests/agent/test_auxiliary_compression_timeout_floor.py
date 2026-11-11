"""Compression's built-in auxiliary request budget is 900s (#54915 follow-up).

A reasoning summary can sit silent well past the old 120s config timeout and
the 300s floor that used to raise it. The built-in budget is now 900s for
compression only:

  * Other auxiliary tasks keep their own timeouts.
  * An explicit per-call ``timeout=`` is used as written.
  * A positive ``auxiliary.compression.timeout`` is used as written, including
    a value shorter than 900s and shorter than the historical 300s floor.
  * Sync ``call_llm`` and async ``async_call_llm`` share ``_effective_aux_timeout``.

These tests exercise the real call paths with a mocked LLM client and assert
the timeout that actually reaches ``client.chat.completions.create``.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import call_llm, async_call_llm

# Built-in compression request budget when the profile does not set one.
COMPRESSION_BUILTIN_TIMEOUT = 900.0

# An explicit config value below the built-in and below the historical 300s floor.
COMPRESSION_SHORT_OVERRIDE = 45.0


def _ok_response():
    return {"ok": True}


def _client_sync():
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    client.chat.completions.create.return_value = _ok_response()
    return client


def _client_async():
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    client.chat.completions.create = AsyncMock(return_value=_ok_response())
    return client


def _patches(client, *, task_timeout):
    """Common mocks: provider resolution, cached client, response validation,
    and the config-derived task timeout."""
    return (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("openai-codex", "gpt-5.5", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client",
              return_value=(client, "gpt-5.5")),
        patch("agent.auxiliary_client._validate_llm_response",
              side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._get_task_timeout",
              return_value=task_timeout),
    )


class TestCompressionTimeoutFloorSync:
    """Sync ``call_llm`` uses the 900s built-in unless config sets a timeout."""

    def test_builtin_compression_timeout_reaches_the_client(self):
        client = _client_sync()
        p1, p2, p3, p4 = _patches(client, task_timeout=30.0)
        with p1, p2, p3, p4, patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={},
        ):
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarise this"}],
            )
        timeout = client.chat.completions.create.call_args.kwargs["timeout"]
        assert timeout == COMPRESSION_BUILTIN_TIMEOUT

    def test_explicit_shorter_config_timeout_is_not_raised(self):
        client = _client_sync()
        p1, p2, p3, p4 = _patches(client, task_timeout=COMPRESSION_SHORT_OVERRIDE)
        with p1, p2, p3, p4, patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"timeout": COMPRESSION_SHORT_OVERRIDE},
        ):
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarise this"}],
            )
        timeout = client.chat.completions.create.call_args.kwargs["timeout"]
        assert timeout == COMPRESSION_SHORT_OVERRIDE


    def test_non_compression_task_is_not_floored(self):
        """Layer 4: only ``compression`` gets the floor; another auxiliary
        task with the same low config timeout must pass it through."""
        client = _client_sync()
        low = 30.0
        p1, p2, p3, p4 = _patches(client, task_timeout=low)
        with p1, p2, p3, p4:
            call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "x"}],
            )
        timeout = client.chat.completions.create.call_args.kwargs["timeout"]
        assert timeout == low, (
            f"non-compression task timeout must stay {low}, got {timeout}"
        )



class TestCompressionTimeoutFloorAsync:
    """Async ``async_call_llm`` mirrors the sync floor (Layer 2)."""

    @pytest.mark.asyncio
    async def test_async_builtin_compression_timeout_reaches_the_client(self):
        client = _client_async()
        p1, p2, p3, p4 = _patches(client, task_timeout=30.0)
        with p1, p2, p3, p4, patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={},
        ):
            await async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarise this"}],
            )
        timeout = client.chat.completions.create.call_args.kwargs["timeout"]
        assert timeout == COMPRESSION_BUILTIN_TIMEOUT

    @pytest.mark.asyncio
    async def test_async_explicit_per_call_timeout_is_not_floored(self):
        client = _client_async()
        explicit = 45.0
        p1, p2, p3, p4 = _patches(client, task_timeout=COMPRESSION_BUILTIN_TIMEOUT)
        with p1, p2, p3, p4:
            await async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "x"}],
                timeout=explicit,
            )
        timeout = client.chat.completions.create.call_args.kwargs["timeout"]
        assert timeout == explicit

    @pytest.mark.asyncio
    async def test_async_non_compression_task_is_not_floored(self):
        client = _client_async()
        low = 30.0
        p1, p2, p3, p4 = _patches(client, task_timeout=low)
        with p1, p2, p3, p4:
            await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "x"}],
            )
        timeout = client.chat.completions.create.call_args.kwargs["timeout"]
        assert timeout == low

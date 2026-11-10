"""Discord REST and Gateway honor NO_PROXY without dropping an explicit DISCORD_PROXY.

Empty ``DISCORD_PROXY`` used to fall through to ``HTTPS_PROXY`` and ignore
``NO_PROXY`` / ``no_proxy``, so the gateway logged ``Using proxy for Discord``
for hosts the operator had already exempted. An explicit ``DISCORD_PROXY``
stays the Discord route.
"""

import asyncio
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock() -> None:
    """Let connect() run when discord.py is not installed. A real install is left alone."""
    existing = sys.modules.get("discord")
    if existing is not None and getattr(existing, "__file__", None):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object, button=lambda *a, **k: (lambda fn: fn), Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, danger=3, green=1, blurple=2, red=3, grey=4, secondary=5,
    )
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(is_loaded=lambda: True)
    discord_mod.AllowedMentions = lambda **kwargs: SimpleNamespace(**kwargs)
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod = MagicMock()
    ext_mod.commands = commands_mod
    sys.modules["discord"] = discord_mod
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import resolve_discord_proxy_url

_PROXY_KEYS = (
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
    "NO_PROXY", "no_proxy", "DISCORD_PROXY",
)
# The production exemption list: apex and leading-dot forms for every Discord family.
_DISCORD_NO_PROXY = (
    "discord.com,.discord.com,discord.gg,.discord.gg,"
    "discordapp.com,.discordapp.com,discordapp.net,.discordapp.net"
)
_HTTPS_PROXY = "http://127.0.0.1:8080"
_DISCORD_PROXY = "http://discord-proxy.example:3128"


@pytest.fixture
def proxy_env(monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("gateway.platforms.base.gateway_trust_env", lambda: True)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    return monkeypatch


@pytest.mark.parametrize("no_proxy", [
    "discord.com",
    ".discord.com",
    "*.discord.com",
    "discord.gg",
    ".discord.gg",
    "gateway.discord.gg",
    "discordapp.com",
    ".discordapp.com",
    "cdn.discordapp.com",
    "discordapp.net",
    ".discordapp.net",
    "media.discordapp.net",
    _DISCORD_NO_PROXY,
])
def test_no_proxy_bypasses_generic_proxy_for_discord_hosts(proxy_env, no_proxy):
    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    proxy_env.setenv("NO_PROXY", no_proxy)
    assert resolve_discord_proxy_url() is None


def test_lowercase_no_proxy_bypasses_generic_proxy(proxy_env):
    proxy_env.setenv("https_proxy", _HTTPS_PROXY)
    proxy_env.setenv("no_proxy", _DISCORD_NO_PROXY)
    assert resolve_discord_proxy_url() is None


def test_discord_proxy_wins_when_set(proxy_env):
    """An explicit Discord proxy is the Discord route, including when NO_PROXY lists Discord."""
    proxy_env.setenv("DISCORD_PROXY", _DISCORD_PROXY)
    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    proxy_env.setenv("NO_PROXY", _DISCORD_NO_PROXY)
    assert resolve_discord_proxy_url() == _DISCORD_PROXY
    assert resolve_discord_proxy_url(target_hosts="https://cdn.discordapp.com/a.png") == _DISCORD_PROXY


def test_socks_discord_proxy_is_normalized(proxy_env):
    proxy_env.setenv("DISCORD_PROXY", "socks://127.0.0.1:1080")
    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    proxy_env.setenv("NO_PROXY", "discord.com")
    assert resolve_discord_proxy_url() == "socks5://127.0.0.1:1080"


@pytest.mark.parametrize("no_proxy", ["", "example.com", "notdiscord.com", "discord.com.evil.test"])
def test_https_proxy_used_when_discord_host_is_not_excluded(proxy_env, no_proxy):
    proxy_env.setenv("DISCORD_PROXY", "")
    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    if no_proxy:
        proxy_env.setenv("NO_PROXY", no_proxy)
    assert resolve_discord_proxy_url() == _HTTPS_PROXY


def test_attachment_url_uses_its_own_host_for_no_proxy(proxy_env):
    """A Discord exemption must not send an unrelated download around the proxy."""
    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    proxy_env.setenv("NO_PROXY", _DISCORD_NO_PROXY)
    assert resolve_discord_proxy_url(target_hosts="https://example.com/a.png") == _HTTPS_PROXY
    assert resolve_discord_proxy_url(target_hosts="https://cdn.discordapp.com/attachments/a.png") is None
    assert resolve_discord_proxy_url(target_hosts="https://media.discordapp.net/attachments/a.mp4") is None


def test_discord_no_proxy_does_not_change_llm_proxy(proxy_env):
    from agent.process_bootstrap import _get_proxy_for_base_url

    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    proxy_env.setenv("NO_PROXY", _DISCORD_NO_PROXY)
    assert resolve_discord_proxy_url() is None
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == _HTTPS_PROXY


def _bot_kwargs_from_connect(monkeypatch) -> dict:
    captured: dict = {}

    def _stop(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before any network I/O")

    adapter = discord_platform.DiscordAdapter(PlatformConfig(enabled=True, token="bot-token"))
    monkeypatch.setattr(discord_platform.discord.opus, "is_loaded", lambda: True)
    monkeypatch.setattr(discord_platform.commands, "Bot", _stop)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    monkeypatch.setattr(adapter, "_snapshot_gate_env", lambda: None)
    monkeypatch.setattr(adapter, "_get_allowed_users", lambda: set())
    monkeypatch.setattr(adapter, "_get_allowed_roles", lambda: set())
    asyncio.run(adapter.connect())
    return captured


def test_connect_omits_proxy_when_no_proxy_excludes_discord(proxy_env, caplog):
    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    proxy_env.setenv("NO_PROXY", _DISCORD_NO_PROXY)
    with caplog.at_level(logging.INFO):
        kwargs = _bot_kwargs_from_connect(proxy_env)
    assert "proxy" not in kwargs
    assert "connector" not in kwargs
    assert not any("Using proxy for Discord" in r.getMessage() for r in caplog.records)


def test_connect_passes_discord_proxy_when_set(proxy_env, caplog):
    proxy_env.setenv("DISCORD_PROXY", _DISCORD_PROXY)
    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    proxy_env.setenv("NO_PROXY", _DISCORD_NO_PROXY)
    with caplog.at_level(logging.INFO):
        kwargs = _bot_kwargs_from_connect(proxy_env)
    assert kwargs.get("proxy") == _DISCORD_PROXY
    assert any("Using proxy for Discord" in r.getMessage() for r in caplog.records)
    assert all(_HTTPS_PROXY not in r.getMessage() for r in caplog.records)


def test_connect_passes_https_proxy_when_discord_is_not_excluded(proxy_env, caplog):
    proxy_env.setenv("HTTPS_PROXY", _HTTPS_PROXY)
    proxy_env.setenv("NO_PROXY", "example.com")
    with caplog.at_level(logging.INFO):
        kwargs = _bot_kwargs_from_connect(proxy_env)
    assert kwargs.get("proxy") == _HTTPS_PROXY
    assert any("Using proxy for Discord" in r.getMessage() and "127.0.0.1:8080" in r.getMessage()
               for r in caplog.records)

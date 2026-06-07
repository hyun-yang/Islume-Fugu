"""FuguModelRouter — task-aware selection + degraded fallback.

These tests stub out `get_available_models`/`get_default_model`/`get_system_model`
so we never depend on the host's env-configured provider keys.
"""

from __future__ import annotations

import pytest

import shared.fugu.router as R
from shared.fugu.router import FuguModelRouter


@pytest.fixture
def stub_env(monkeypatch):
    def _apply(available: list[str], default: str = "", system: str = ""):
        monkeypatch.setattr(R, "get_available_models", lambda: list(available))
        monkeypatch.setattr(R, "get_default_model", lambda: default)
        monkeypatch.setattr(R, "get_system_model", lambda: system)
    return _apply


def test_hint_wins_when_available(stub_env) -> None:
    stub_env(["claude-sonnet-4-5", "claude-haiku-4-5"], default="claude-haiku-4-5")
    router = FuguModelRouter(policy="balanced")
    d = router.select("conversation_turn", hint="claude-sonnet-4-5")
    assert d.model == "claude-sonnet-4-5"
    assert d.degraded is False


def test_hint_ignored_when_not_available_falls_to_policy_ladder(stub_env) -> None:
    stub_env(["claude-haiku-4-5"], default="claude-haiku-4-5")
    router = FuguModelRouter(policy="balanced")
    d = router.select("conversation_turn", hint="openai/gpt-4o")
    # Default chat resolves to haiku here, which is the first balanced entry.
    assert d.model == "claude-haiku-4-5"
    assert d.degraded is False


def test_ladder_falls_through_unavailable_candidates(stub_env) -> None:
    stub_env(
        ["openai/gpt-4o-mini", "gemini/gemini-2.0-flash"],
        default="openai/gpt-4o-mini",
        system="openai/gpt-4o-mini",
    )
    router = FuguModelRouter(policy="quality")
    d = router.select("safety_review")
    # quality ladder: haiku → gemini-flash → system. haiku missing, flash wins.
    assert d.model == "gemini/gemini-2.0-flash"
    assert d.degraded is False


def test_degraded_when_no_candidate_matches(stub_env) -> None:
    # negotiation_judge ladder is fully explicit (no default/system sentinels)
    # so none of "mystery/foo" matches and we fall through to the degraded
    # default.
    stub_env(["mystery/foo"], default="mystery/foo", system="mystery/foo")
    router = FuguModelRouter(policy="quality")
    d = router.select("negotiation_judge")
    assert d.degraded is True
    assert d.model == "mystery/foo"


def test_negative_budget_forces_cheap_lane(stub_env) -> None:
    stub_env(
        [
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "openai/gpt-4o-mini",
            "gemini/gemini-2.0-flash",
        ],
        default="claude-haiku-4-5",
        system="claude-haiku-4-5",
    )
    router = FuguModelRouter(policy="quality")
    d = router.select("conversation_turn", budget_remaining_usd=0.0)
    # cheap ladder for conversation_turn starts with haiku.
    assert d.model == "claude-haiku-4-5"
    assert d.policy == "cheap"


def test_system_sentinel_resolves(stub_env) -> None:
    stub_env(
        ["openai/gpt-4o-mini"],
        default="openai/gpt-4o-mini",
        system="openai/gpt-4o-mini",
    )
    router = FuguModelRouter(policy="balanced")
    d = router.select("affinity_analysis")
    assert d.model == "openai/gpt-4o-mini"
    assert d.degraded is False

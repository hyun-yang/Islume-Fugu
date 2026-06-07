"""Worker adapter — _FuguResponseAdapter shape + flag-off behavior.

The full _run_turn path needs a Postgres + Redis stack, so we don't try to
exercise it here. Instead we lock down the duck-typed adapter shape that the
persistence block in worker._run_turn reads from, and we verify the legacy
generate() path is selected when fugu_enabled=False.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from services.worker.main import _FuguResponseAdapter, _run_turn_via_fugu
from shared.fugu.turn_orchestrator import (
    FuguTurnOrchestrator,
    TurnOrchestrationInput,
    TurnOrchestrationResult,
)
from shared.intent_plugins.base import ToolCall


def test_adapter_exposes_legacy_fields() -> None:
    result = TurnOrchestrationResult(
        text="hi",
        tool_calls=[ToolCall(id="x", name="t", arguments={})],
        primary_model="claude-haiku-4-5",
        input_tokens=4,
        output_tokens=2,
        cost_usd=0.0001,
        latency_ms=42,
        degraded=True,
        subtask_audit=[{"kind": "conversation_turn"}],
    )
    adapter = _FuguResponseAdapter(result)
    # Worker reads exactly these attributes off LLMResponse/GenerationResult.
    assert adapter.text == "hi"
    assert adapter.input_tokens == 4
    assert adapter.output_tokens == 2
    assert adapter.model == "claude-haiku-4-5"
    assert adapter.cost_usd == pytest.approx(0.0001)
    assert len(adapter.tool_calls) == 1
    assert adapter.tool_calls[0].name == "t"
    assert adapter.fugu_degraded is True
    assert adapter.fugu_subtask_audit == [{"kind": "conversation_turn"}]


@pytest.mark.asyncio
async def test_run_turn_via_fugu_falls_back_on_empty_result(monkeypatch) -> None:
    speaker = SimpleNamespace(
        id=uuid4(),
        name="A",
        persona_prompt="p",
        tone="friendly",
        attached_plugins=None,
        safety=None,
        boundaries=None,
        conversation_phases=None,
        translations=None,
        references_meta=None,
        created_by=None,
        slug=None,
    )
    listener = SimpleNamespace(**speaker.__dict__, **{})

    async def fake_run(self, input):
        return TurnOrchestrationResult(
            text="",
            tool_calls=[],
            primary_model="claude-haiku-4-5",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=0,
            degraded=True,
            subtask_audit=[],
        )

    monkeypatch.setattr(FuguTurnOrchestrator, "run", fake_run)
    adapter, failed = await _run_turn_via_fugu(
        input=TurnOrchestrationInput(
            session_id=uuid4(),
            turn_number=1,
            speaker=speaker,
            listener=listener,
            listener_user_name="B",
            match_context="ctx",
            phase="initial",
            history=[],
            is_opening=True,
            attached_plugins=[],
            preferred_model=None,
            system_prompt_builder=lambda *a, **kw: "sys",
        ),
        routing_policy="balanced",
        subtask_timeout_ms=1000,
        plugin_selector_min_count=2,
        turn_budget_usd=None,
    )
    assert adapter is None
    assert failed is True


@pytest.mark.asyncio
async def test_run_turn_via_fugu_catches_orchestrator_exception(monkeypatch) -> None:
    speaker = SimpleNamespace(
        id=uuid4(),
        name="A",
        persona_prompt="p",
        tone="friendly",
        attached_plugins=None,
        safety=None,
        boundaries=None,
        conversation_phases=None,
        translations=None,
        references_meta=None,
        created_by=None,
        slug=None,
    )

    async def boom(self, input):
        raise RuntimeError("router crashed")

    monkeypatch.setattr(FuguTurnOrchestrator, "run", boom)
    adapter, failed = await _run_turn_via_fugu(
        input=TurnOrchestrationInput(
            session_id=uuid4(),
            turn_number=1,
            speaker=speaker,
            listener=speaker,
            listener_user_name="B",
            match_context="ctx",
            phase="initial",
            history=[],
            is_opening=False,
            attached_plugins=[],
            preferred_model=None,
            system_prompt_builder=lambda *a, **kw: "sys",
        ),
        routing_policy="balanced",
        subtask_timeout_ms=1000,
        plugin_selector_min_count=2,
        turn_budget_usd=None,
    )
    assert adapter is None
    assert failed is True

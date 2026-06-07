"""FuguTurnOrchestrator — end-to-end persona/safety/plugin-selector flow.

All LLM calls are stubbed via `shared.fugu.executor.generate`. We assert on
subtask audit, the persisted-shape fields, and safety/revise behavior.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

import shared.fugu.executor as E
import shared.fugu.router as R
from shared.fugu.executor import FuguExecutor
from shared.fugu.router import FuguModelRouter
from shared.fugu.turn_orchestrator import (
    FuguTurnOrchestrator,
    TurnOrchestrationInput,
)
from shared.intent_plugins import get_plugin
from shared.llm import GenerationResult, LLMResponse


@pytest.fixture(autouse=True)
def _stub_router_env(monkeypatch):
    monkeypatch.setattr(R, "get_available_models", lambda: ["claude-haiku-4-5"])
    monkeypatch.setattr(R, "get_default_model", lambda: "claude-haiku-4-5")
    monkeypatch.setattr(R, "get_system_model", lambda: "claude-haiku-4-5")


def _speaker(**overrides) -> SimpleNamespace:
    base = dict(
        id=uuid4(),
        name="Alice",
        persona_prompt="You love jazz.",
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
    base.update(overrides)
    return SimpleNamespace(**base)


def _builder(speaker, listener_user_name, match_context, *, phase, plugins):
    """Mimic worker.build_system_prompt minus the file/DB dependencies."""
    plug_ids = [p.id for p, _ in (plugins or [])]
    return (
        f"persona={speaker.persona_prompt}|tone={speaker.tone}|"
        f"listener={listener_user_name}|ctx={match_context}|phase={phase}|"
        f"plugins={','.join(plug_ids)}"
    )


def _input(speaker, *, attached_plugins=(), is_opening=False):
    listener = _speaker(name="Bob")
    return TurnOrchestrationInput(
        session_id=uuid4(),
        turn_number=1,
        speaker=speaker,
        listener=listener,
        listener_user_name="Bob",
        match_context="both love jazz",
        phase="initial",
        history=[{"role": "user", "content": "hello"}],
        is_opening=is_opening,
        attached_plugins=list(attached_plugins),
        preferred_model=None,
        system_prompt_builder=_builder,
        max_tokens=120,
    )


@pytest.mark.asyncio
async def test_persona_only_flow_no_safety_no_plugins(monkeypatch) -> None:
    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        assert tools is None
        return LLMResponse(text="hey!", input_tokens=5, output_tokens=2, model=model)
    monkeypatch.setattr(E, "generate", fake_generate)

    orch = FuguTurnOrchestrator(executor=FuguExecutor(router=FuguModelRouter()))
    result = await orch.run(_input(_speaker()))
    assert result.text == "hey!"
    assert result.tool_calls == []
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.primary_model == "claude-haiku-4-5"
    kinds = [a["kind"] for a in result.subtask_audit]
    assert kinds == ["conversation_turn"]
    assert result.degraded is False


@pytest.mark.asyncio
async def test_plugin_selector_filters_tools(monkeypatch) -> None:
    bartering = get_plugin("bartering")
    assert bartering is not None
    # Build a fake second plugin by reusing bartering with a different id-like
    # marker — we only need len(attached_plugins) >= 2 to trigger the selector.
    plug2 = bartering  # reused; the selector only deals in ids
    speaker = _speaker(
        attached_plugins=[
            {"plugin": "bartering", "policy": {"role": "seller"}},
            {"plugin": "bartering", "policy": {"role": "buyer"}},
        ]
    )

    calls: list[dict] = []

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        calls.append({"system": system, "has_tools": tools is not None})
        if "routing classifier" in system:
            return LLMResponse(
                text=json.dumps({"plugins": ["bartering"]}),
                input_tokens=1,
                output_tokens=1,
                model=model,
            )
        # persona draft — must have tools attached
        return LLMResponse(text="ok", input_tokens=1, output_tokens=1, model=model)

    monkeypatch.setattr(E, "generate", fake_generate)

    orch = FuguTurnOrchestrator(
        executor=FuguExecutor(router=FuguModelRouter()),
        plugin_selector_min_count=2,
    )
    input = _input(speaker, attached_plugins=[(bartering, {"role": "seller"}), (plug2, {"role": "buyer"})])
    result = await orch.run(input)
    assert result.text == "ok"
    kinds = [a["kind"] for a in result.subtask_audit]
    assert kinds[0] == "plugin_intent"
    assert kinds[1] == "conversation_turn"
    # persona draft must have received the tools list
    persona_call = [c for c in calls if "routing classifier" not in c["system"]][0]
    assert persona_call["has_tools"] is True


@pytest.mark.asyncio
async def test_safety_revise_overrides_draft_on_violation(monkeypatch) -> None:
    speaker = _speaker(safety={"redline_topics": ["politics"]})

    seq = iter(
        [
            ("draft", "let's talk politics"),
            ("review", json.dumps({"ok": False, "reasons": ["politics is redlined"]})),
            ("revise", "let's talk music instead"),
        ]
    )

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        kind, text = next(seq)
        return LLMResponse(text=text, input_tokens=2, output_tokens=2, model=model)

    monkeypatch.setattr(E, "generate", fake_generate)

    orch = FuguTurnOrchestrator(executor=FuguExecutor(router=FuguModelRouter()))
    result = await orch.run(_input(speaker))
    assert result.text == "let's talk music instead"
    kinds = [a["kind"] for a in result.subtask_audit]
    assert kinds == ["conversation_turn", "safety_review", "conversation_turn"]
    assert result.degraded is True
    # Tokens accumulate across draft + revise (review is separate, also counted in audit)
    assert result.input_tokens >= 4


@pytest.mark.asyncio
async def test_safety_pass_keeps_draft(monkeypatch) -> None:
    speaker = _speaker(safety={"refuse_personal_info_share": True})

    seq = iter(
        [
            ("draft", "what jazz album do you love?"),
            ("review", json.dumps({"ok": True, "reasons": []})),
        ]
    )

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        _, text = next(seq)
        return LLMResponse(text=text, input_tokens=2, output_tokens=2, model=model)

    monkeypatch.setattr(E, "generate", fake_generate)
    result = await FuguTurnOrchestrator(
        executor=FuguExecutor(router=FuguModelRouter())
    ).run(_input(speaker))
    assert result.text == "what jazz album do you love?"
    assert result.degraded is False


@pytest.mark.asyncio
async def test_safety_review_unparseable_text_fails_closed_into_revise(monkeypatch) -> None:
    """Before the fix this test asserted the opposite ("pass through").

    Production trace turn-18 showed Claude wrapping its JSON in a
    ```json``` fence; the old parser fell back to ``None`` which the
    orchestrator treated as "no review happened", and the draft was
    accepted unchecked. The safety gate is now fail-closed: an
    unparseable reviewer reply triggers a revise pass.
    """
    speaker = _speaker(safety={"redline_topics": ["politics"]})

    seq = iter(
        [
            ("draft", "all good"),
            ("review", "this is not json at all, just prose"),
            ("revise", "revised reply"),
        ]
    )

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        _, text = next(seq)
        return LLMResponse(text=text, input_tokens=2, output_tokens=2, model=model)

    monkeypatch.setattr(E, "generate", fake_generate)
    result = await FuguTurnOrchestrator(
        executor=FuguExecutor(router=FuguModelRouter())
    ).run(_input(speaker))
    assert result.text == "revised reply"
    kinds = [a["kind"] for a in result.subtask_audit]
    assert kinds == ["conversation_turn", "safety_review", "conversation_turn"]
    assert result.degraded is True


@pytest.mark.asyncio
async def test_safety_review_strips_markdown_code_fence(monkeypatch) -> None:
    """Reproduction of the turn-18 Langfuse trace: Claude emitted
    ```json
    {"ok": true, "reasons": []}
    ```
    plus a couple of trailing sentences. The parser must recover the
    embedded JSON and treat it as a clean pass — no revise.
    """
    speaker = _speaker(safety={"redline_topics": ["politics"]})

    fenced_review = (
        "```json\n"
        '{"ok": true, "reasons": []}\n'
        "```\n\n"
        "The draft is a safe response about music preferences."
    )
    seq = iter(
        [
            ("draft", "Lovely chat about jazz."),
            ("review", fenced_review),
        ]
    )

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        _, text = next(seq)
        return LLMResponse(text=text, input_tokens=2, output_tokens=2, model=model)

    monkeypatch.setattr(E, "generate", fake_generate)
    result = await FuguTurnOrchestrator(
        executor=FuguExecutor(router=FuguModelRouter())
    ).run(_input(speaker))
    assert result.text == "Lovely chat about jazz."
    kinds = [a["kind"] for a in result.subtask_audit]
    # No revise — the fenced JSON parsed cleanly as ok=True.
    assert kinds == ["conversation_turn", "safety_review"]
    assert result.degraded is False


@pytest.mark.asyncio
async def test_safety_review_extracts_json_object_after_prose(monkeypatch) -> None:
    """When the model prefixes the JSON with a sentence (no code fence),
    the regex extractor should still recover the object."""
    speaker = _speaker(safety={"redline_topics": ["politics"]})
    chatty_review = (
        "I think this draft is fine. Final verdict: "
        '{"ok": true, "reasons": []}'
    )
    seq = iter(
        [
            ("draft", "ok draft"),
            ("review", chatty_review),
        ]
    )

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        _, text = next(seq)
        return LLMResponse(text=text, input_tokens=2, output_tokens=2, model=model)

    monkeypatch.setattr(E, "generate", fake_generate)
    result = await FuguTurnOrchestrator(
        executor=FuguExecutor(router=FuguModelRouter())
    ).run(_input(speaker))
    assert result.text == "ok draft"
    assert result.degraded is False


@pytest.mark.asyncio
async def test_persona_failure_returns_empty_so_worker_can_fall_back(monkeypatch) -> None:
    async def fail(*, system, messages, model, max_tokens, tools=None):
        raise RuntimeError("boom")
    monkeypatch.setattr(E, "generate", fail)
    result = await FuguTurnOrchestrator(
        executor=FuguExecutor(router=FuguModelRouter())
    ).run(_input(_speaker()))
    assert result.text == ""
    assert result.degraded is True


@pytest.mark.asyncio
async def test_opening_turn_uses_opener_prompt(monkeypatch) -> None:
    captured: dict = {}

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        captured["messages"] = messages
        return LLMResponse(text="hi!", input_tokens=1, output_tokens=1, model=model)
    monkeypatch.setattr(E, "generate", fake_generate)
    await FuguTurnOrchestrator(
        executor=FuguExecutor(router=FuguModelRouter())
    ).run(_input(_speaker(), is_opening=True))
    assert captured["messages"][0]["content"].startswith("(Start")


@pytest.mark.asyncio
async def test_persona_with_tool_call_propagates(monkeypatch) -> None:
    bartering = get_plugin("bartering")
    speaker = _speaker(
        attached_plugins=[{"plugin": "bartering", "policy": {"role": "seller"}}]
    )

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        from shared.intent_plugins.base import ToolCall
        return GenerationResult(
            text="",
            tool_calls=[ToolCall(id="x", name="propose_price", arguments={"amount": 10, "currency": "ISL", "item_name": "vinyl"})],
            input_tokens=3,
            output_tokens=2,
            model=model,
        )
    monkeypatch.setattr(E, "generate", fake_generate)
    input = _input(
        speaker,
        attached_plugins=[(bartering, {"role": "seller"})],
    )
    result = await FuguTurnOrchestrator(
        executor=FuguExecutor(router=FuguModelRouter())
    ).run(input)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "propose_price"

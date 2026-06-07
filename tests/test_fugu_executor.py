"""FuguExecutor — timeout, provider error, normalization."""

from __future__ import annotations

import asyncio

import pytest

import shared.fugu.executor as E
import shared.fugu.router as R
from shared.fugu.executor import FuguExecutor
from shared.fugu.router import FuguModelRouter
from shared.fugu.types import FuguSubtask
from shared.intent_plugins.base import ToolCall
from shared.llm import GenerationResult, LLMResponse


@pytest.fixture(autouse=True)
def _stub_router_env(monkeypatch):
    monkeypatch.setattr(R, "get_available_models", lambda: ["claude-haiku-4-5"])
    monkeypatch.setattr(R, "get_default_model", lambda: "claude-haiku-4-5")
    monkeypatch.setattr(R, "get_system_model", lambda: "claude-haiku-4-5")


def _make_subtask(kind="conversation_turn"):
    return FuguSubtask(
        id="t-1",
        kind=kind,
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
    )


@pytest.mark.asyncio
async def test_run_subtask_normalizes_llm_response(monkeypatch) -> None:
    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        return LLMResponse(
            text="hi back", input_tokens=4, output_tokens=2, model="claude-haiku-4-5"
        )
    monkeypatch.setattr(E, "generate", fake_generate)
    res = await FuguExecutor(router=FuguModelRouter()).run_subtask(_make_subtask())
    assert res.status == "ok"
    assert res.text == "hi back"
    assert res.input_tokens == 4
    assert res.output_tokens == 2
    assert res.model_used == "claude-haiku-4-5"
    assert res.tool_calls == []
    assert res.cost_usd > 0  # haiku is priced
    assert res.degraded is False


@pytest.mark.asyncio
async def test_run_subtask_normalizes_generation_result(monkeypatch) -> None:
    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        return GenerationResult(
            text="",
            tool_calls=[ToolCall(id="c1", name="propose_price", arguments={"amount": 9})],
            input_tokens=10,
            output_tokens=3,
            model="claude-haiku-4-5",
        )
    monkeypatch.setattr(E, "generate", fake_generate)
    sub = _make_subtask()
    res = await FuguExecutor(router=FuguModelRouter()).run_subtask(sub)
    assert res.status == "ok"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "propose_price"


@pytest.mark.asyncio
async def test_run_subtask_timeout_returns_soft_failure(monkeypatch) -> None:
    async def slow_generate(*, system, messages, model, max_tokens, tools=None):
        await asyncio.sleep(0.5)
        return LLMResponse(text="too late", input_tokens=1, output_tokens=1, model="claude-haiku-4-5")
    monkeypatch.setattr(E, "generate", slow_generate)
    sub = _make_subtask()
    sub.timeout_ms = 50
    res = await FuguExecutor(router=FuguModelRouter()).run_subtask(sub)
    assert res.status == "timeout"
    assert res.text == ""
    assert res.error and "timeout" in res.error


@pytest.mark.asyncio
async def test_run_subtask_provider_error_is_soft(monkeypatch) -> None:
    async def boom(*, system, messages, model, max_tokens, tools=None):
        raise RuntimeError("anthropic 500")
    monkeypatch.setattr(E, "generate", boom)
    res = await FuguExecutor(router=FuguModelRouter()).run_subtask(_make_subtask())
    assert res.status == "provider_error"
    assert res.error == "anthropic 500"


@pytest.mark.asyncio
async def test_bounded_gather_preserves_order_and_caps_concurrency() -> None:
    from shared.fugu.executor import bounded_gather

    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def task(i: int) -> int:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1
        return i

    results = await bounded_gather([task(i) for i in range(8)], max_concurrency=3)
    assert results == list(range(8))
    assert peak <= 3


@pytest.mark.asyncio
async def test_run_subtask_tags_span_as_langfuse_observation(monkeypatch) -> None:
    """Regression for the turn-18 trace: the Fugu wrapper span must carry
    `langfuse.observation.type` so the shared OTel exporter
    (`shared/telemetry.py::LLMSpanFilterProcessor`) doesn't drop it. Without
    this tag every `fugu.*` attribute (routing model, policy, degraded flag)
    is invisible in Langfuse even though the child llm.<provider> span is
    forwarded.
    """
    import shared.telemetry as T

    captured: list[dict] = []

    class _CaptureSpan:
        def __init__(self) -> None:
            self.attributes: dict = {}

        def set_attribute(self, k: str, v) -> None:
            self.attributes[k] = v

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _CaptureTracer:
        def start_as_current_span(self, name: str):
            span = _CaptureSpan()
            captured.append({"name": name, "span": span})
            return span

    monkeypatch.setattr(T, "get_tracer", lambda _name: _CaptureTracer())
    # E.get_tracer is the symbol Fugu actually calls (imported at module top).
    monkeypatch.setattr(E, "get_tracer", lambda _name: _CaptureTracer())

    async def fake_generate(*, system, messages, model, max_tokens, tools=None):
        return LLMResponse(text="hi", input_tokens=1, output_tokens=1, model="claude-haiku-4-5")

    monkeypatch.setattr(E, "generate", fake_generate)
    await FuguExecutor(router=FuguModelRouter()).run_subtask(_make_subtask())

    assert captured, "Fugu should have opened at least one span"
    fugu_span = captured[0]["span"]
    assert fugu_span.attributes.get("langfuse.observation.type") == "span", (
        "Fugu subtask span must be tagged so LLMSpanFilterProcessor forwards it "
        "to Langfuse; otherwise all fugu.* attrs are silently dropped."
    )
    # Sanity: the routing/kind attrs we care about are still present.
    assert fugu_span.attributes.get("fugu.subtask.kind") == "conversation_turn"
    assert "fugu.routing.model" in fugu_span.attributes

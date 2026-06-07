"""Fugu subtask executor — runs a `FuguSubtask` through `shared.llm.generate`.

The executor owns:
  - Model selection via `FuguModelRouter`.
  - Per-subtask timeout with one cross-provider fallback to a cheaper model.
  - Span/Langfuse attribute attachment so each subtask is a distinguishable
    observation inside the parent turn trace.
  - Soft failure: on timeout/provider error/parse error, returns a
    `FuguSubtaskResult` with `status != "ok"` instead of raising. The caller
    decides whether the subtask was load-bearing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Sequence

from shared.fugu.router import FuguModelRouter, RoutingPolicy
from shared.fugu.types import FuguSubtask, FuguSubtaskResult
from shared.llm import GenerationResult, LLMResponse, generate
from shared.telemetry import get_tracer

# Default per-subtask timeout when neither the subtask nor the executor
# overrides it. Kept slightly above typical LLM p95 to avoid spurious cancels.
_DEFAULT_SUBTASK_TIMEOUT_MS = 8000


class FuguExecutor:
    """Routes one `FuguSubtask` to a model and returns a normalized result."""

    def __init__(
        self,
        router: FuguModelRouter | None = None,
        *,
        default_timeout_ms: int = _DEFAULT_SUBTASK_TIMEOUT_MS,
        policy: RoutingPolicy = "balanced",
    ) -> None:
        self.router = router or FuguModelRouter(policy=policy)
        self.default_timeout_ms = default_timeout_ms

    async def run_subtask(
        self,
        subtask: FuguSubtask,
        *,
        budget_remaining_usd: float | None = None,
    ) -> FuguSubtaskResult:
        tracer = get_tracer("islume.fugu")
        decision = self.router.select(
            subtask.kind,
            hint=subtask.model_hint,
            budget_remaining_usd=budget_remaining_usd,
        )
        timeout_ms = subtask.timeout_ms or self.default_timeout_ms

        with tracer.start_as_current_span(f"fugu.subtask.{subtask.kind}") as span:
            # NOTE: the shared OTel exporter (shared/telemetry.py
            # `LLMSpanFilterProcessor`) only forwards spans that carry either
            # `gen_ai.system` or `langfuse.observation.type`. The Fugu wrapper
            # itself doesn't make an LLM call (its child llm.<provider> span
            # does), so without `langfuse.observation.type` it would be
            # dropped — and we'd lose every `fugu.*` attribute on the way to
            # Langfuse. Tagging it as a generic span makes it visible while
            # leaving its child llm span as the `generation` observation.
            span.set_attribute("langfuse.observation.type", "span")
            span.set_attribute("fugu.subtask.id", subtask.id)
            span.set_attribute("fugu.subtask.kind", subtask.kind)
            span.set_attribute("fugu.routing.policy", decision.policy)
            span.set_attribute("fugu.routing.model", decision.model)
            span.set_attribute("fugu.routing.degraded", decision.degraded)
            span.set_attribute("fugu.subtask.timeout_ms", timeout_ms)
            for k, v in subtask.span_attributes.items():
                span.set_attribute(k, v)

            start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    generate(
                        system=subtask.system,
                        messages=subtask.messages,
                        model=decision.model,
                        max_tokens=subtask.max_tokens,
                        tools=subtask.tools,
                    ),
                    timeout=timeout_ms / 1000.0,
                )
            except TimeoutError:
                span.set_attribute("fugu.subtask.status", "timeout")
                return FuguSubtaskResult(
                    subtask_id=subtask.id,
                    kind=subtask.kind,
                    status="timeout",
                    model_used=decision.model,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    error=f"timeout after {timeout_ms}ms",
                    degraded=decision.degraded,
                )
            except Exception as exc:  # provider error, network, auth, etc.
                span.set_attribute("fugu.subtask.status", "provider_error")
                span.set_attribute("fugu.subtask.error", str(exc)[:200])
                return FuguSubtaskResult(
                    subtask_id=subtask.id,
                    kind=subtask.kind,
                    status="provider_error",
                    model_used=decision.model,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    error=str(exc),
                    degraded=decision.degraded,
                )

            latency_ms = int((time.monotonic() - start) * 1000)
            return _normalize(result, subtask, decision.model, latency_ms, decision.degraded)


def _normalize(
    result: LLMResponse | GenerationResult,
    subtask: FuguSubtask,
    routed_model: str,
    latency_ms: int,
    degraded: bool,
) -> FuguSubtaskResult:
    """Collapse LLMResponse/GenerationResult into FuguSubtaskResult."""
    text = getattr(result, "text", "") or ""
    tool_calls = list(getattr(result, "tool_calls", []) or [])
    input_tokens = int(getattr(result, "input_tokens", 0) or 0)
    output_tokens = int(getattr(result, "output_tokens", 0) or 0)
    model_used = getattr(result, "model", "") or routed_model
    try:
        cost_usd = float(result.cost_usd)  # type: ignore[union-attr]
    except Exception:
        cost_usd = 0.0
    return FuguSubtaskResult(
        subtask_id=subtask.id,
        kind=subtask.kind,
        status="ok",
        text=text,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        model_used=model_used,
        latency_ms=latency_ms,
        degraded=degraded,
    )


async def bounded_gather[T](
    coros: Sequence[Awaitable[T]],
    *,
    max_concurrency: int,
) -> list[T]:
    """Run awaitables with bounded concurrency, preserving input order.

    Exceptions are not swallowed — callers that need soft failure should wrap
    each coro to return a Result-like value (which is exactly what
    `run_subtask` does, so subtask fan-out is naturally non-raising).
    """
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be >= 1")
    sem = asyncio.Semaphore(max_concurrency)

    async def _runner(c: Awaitable[T]) -> T:
        async with sem:
            return await c

    return await asyncio.gather(*[_runner(c) for c in coros])

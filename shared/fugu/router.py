"""Task-aware model router.

Maps a `TaskKind` to a prioritized list of candidate models per routing
policy (`quality` / `balanced` / `cheap`). The router consults the live
available-models list from `shared.llm.get_available_models()`, the explicit
chat default (`get_default_model()`), and the system model
(`get_system_model()`) to materialize the actual candidate set; it never
invents a model that isn't configured.

If no candidate is available, it falls back to whatever
`get_default_model()` returns and marks the decision as `degraded=True` so
the executor can surface the downgrade on the OTel span.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shared.fugu.types import TaskKind
from shared.llm import (
    PRICING,
    get_available_models,
    get_default_model,
    get_system_model,
)

RoutingPolicy = Literal["quality", "balanced", "cheap"]


# Sentinel tokens resolved lazily at decide-time, so the router stays in sync
# with whatever env-driven model list the process booted with.
_DEFAULT_CHAT = "<default_chat>"
_SYSTEM = "<system>"

# Per-(kind, policy) candidate ladder. Each entry is tried in order; the first
# one that is `_is_available` wins. Bare names are anthropic; prefixed names
# follow `shared.llm.parse_model` conventions.
DEFAULT_TASK_POLICY: dict[TaskKind, dict[RoutingPolicy, list[str]]] = {
    "conversation_turn": {
        "quality": [_DEFAULT_CHAT, "claude-sonnet-4-5", "openai/gpt-4o"],
        "balanced": [_DEFAULT_CHAT, "openai/gpt-5-mini", "claude-haiku-4-5"],
        "cheap": ["claude-haiku-4-5", "openai/gpt-4o-mini", "gemini/gemini-2.0-flash"],
    },
    "plugin_intent": {
        "quality": ["openai/gpt-5-mini", "claude-haiku-4-5", _SYSTEM],
        "balanced": ["openai/gpt-5-mini", "claude-haiku-4-5", _SYSTEM],
        "cheap": ["openai/gpt-5-nano", "gemini/gemini-2.0-flash", _SYSTEM],
    },
    "safety_review": {
        "quality": ["claude-haiku-4-5", "gemini/gemini-2.0-flash", _SYSTEM],
        "balanced": ["claude-haiku-4-5", "gemini/gemini-2.0-flash", _SYSTEM],
        "cheap": ["gemini/gemini-2.0-flash", "openai/gpt-5-nano", _SYSTEM],
    },
    "reference_select": {
        "quality": ["claude-haiku-4-5", _SYSTEM],
        "balanced": ["claude-haiku-4-5", _SYSTEM],
        "cheap": ["gemini/gemini-2.0-flash", _SYSTEM],
    },
    "affinity_analysis": {
        "quality": [_SYSTEM, "claude-haiku-4-5"],
        "balanced": [_SYSTEM, "claude-haiku-4-5"],
        "cheap": [_SYSTEM, "openai/gpt-4o-mini"],
    },
    "semantic_similarity": {
        "quality": [_SYSTEM, "openai/gpt-4o-mini"],
        "balanced": [_SYSTEM, "openai/gpt-4o-mini"],
        "cheap": [_SYSTEM, "gemini/gemini-2.0-flash"],
    },
    "negotiation_draft": {
        "quality": ["claude-sonnet-4-5", "openai/gpt-4o", "claude-haiku-4-5"],
        "balanced": ["claude-sonnet-4-5", "openai/gpt-5-mini", "claude-haiku-4-5"],
        "cheap": ["claude-haiku-4-5", "openai/gpt-4o-mini"],
    },
    "negotiation_judge": {
        "quality": ["claude-sonnet-4-5", "gemini/gemini-2.5-pro", "claude-haiku-4-5"],
        "balanced": ["claude-sonnet-4-5", "openai/gpt-5-mini", "claude-haiku-4-5"],
        "cheap": ["claude-haiku-4-5", "openai/gpt-4o-mini"],
    },
}


@dataclass(frozen=True)
class RoutingDecision:
    """Result of `FuguModelRouter.select`."""

    model: str
    kind: TaskKind
    policy: RoutingPolicy
    # True iff none of the kind's preferred candidates were available and the
    # router fell back to a generic chat/system default.
    degraded: bool
    # Cost ceiling (per 1M tokens, input rate) used to estimate budget impact.
    input_rate_per_1m: float


def _resolve_tokens(candidates: list[str]) -> list[str]:
    """Replace sentinel tokens with the live default chat/system models."""
    chat_default = get_default_model()
    sys_default = get_system_model()
    out: list[str] = []
    for c in candidates:
        if c == _DEFAULT_CHAT:
            out.append(chat_default)
        elif c == _SYSTEM:
            out.append(sys_default)
        else:
            out.append(c)
    # De-dup while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _input_rate(model: str) -> float:
    p = PRICING.get(model)
    return float(p["input"]) if p else 0.0


class FuguModelRouter:
    """Stateless selector over `DEFAULT_TASK_POLICY` + a current policy."""

    def __init__(
        self,
        policy: RoutingPolicy = "balanced",
        task_policy: dict[TaskKind, dict[RoutingPolicy, list[str]]] | None = None,
    ) -> None:
        self.policy: RoutingPolicy = policy
        self.task_policy = task_policy or DEFAULT_TASK_POLICY

    def select(
        self,
        kind: TaskKind,
        *,
        hint: str | None = None,
        budget_remaining_usd: float | None = None,
    ) -> RoutingDecision:
        """Pick a model for `kind`.

        Priority:
          1. `hint` if it's currently available.
          2. The kind's policy ladder (first available wins).
          3. `get_default_model()` (degraded=True).

        `budget_remaining_usd` is a soft signal: if non-None and <= 0, the
        router downshifts to the `cheap` ladder regardless of `self.policy`.
        """
        available = set(get_available_models())

        if hint and hint in available:
            return RoutingDecision(
                model=hint,
                kind=kind,
                policy=self.policy,
                degraded=False,
                input_rate_per_1m=_input_rate(hint),
            )

        effective_policy: RoutingPolicy = self.policy
        if budget_remaining_usd is not None and budget_remaining_usd <= 0:
            effective_policy = "cheap"

        ladder = self.task_policy.get(kind, {}).get(effective_policy, [])
        for candidate in _resolve_tokens(ladder):
            if candidate in available:
                return RoutingDecision(
                    model=candidate,
                    kind=kind,
                    policy=effective_policy,
                    degraded=False,
                    input_rate_per_1m=_input_rate(candidate),
                )

        # Last-ditch fallback. `get_default_model()` may itself return a model
        # not in `available` if the env is misconfigured — we still surface it
        # rather than raising, because the worker has its own retry path.
        fallback = get_default_model()
        return RoutingDecision(
            model=fallback,
            kind=kind,
            policy=effective_policy,
            degraded=True,
            input_rate_per_1m=_input_rate(fallback),
        )

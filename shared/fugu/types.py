"""Fugu subtask types — kind taxonomy, request/response shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from shared.intent_plugins.base import ToolCall, ToolDef

# Subtask taxonomy. Each kind has a router policy entry in `fugu.router`.
TaskKind = Literal[
    "conversation_turn",      # the main persona-driven user-facing reply
    "plugin_intent",          # decide which attached plugin(s) are active this turn
    "safety_review",          # cheap pass/fail review of a draft against safety rules
    "reference_select",       # pick which references to inject (reserved for v2; not LLM-called in v1)
    "affinity_analysis",      # session-level compatibility scoring (worker._analyze_affinity)
    "semantic_similarity",    # pairwise persona similarity (matching.similarity.llm_similarity)
    "negotiation_draft",      # bartering: side draft (reserved for Phase 4)
    "negotiation_judge",      # bartering: judge (reserved for Phase 4)
]

# Status for a single subtask execution.
SubtaskStatus = Literal["ok", "timeout", "parse_error", "provider_error", "skipped"]


@dataclass
class FuguSubtask:
    """One unit of work Fugu hands to an LLM.

    `model_hint` lets the caller pin a model (e.g. the speaker's preferred chat
    model). The router still validates availability; an unavailable hint is
    silently downgraded with `routing.degraded=True`.
    """

    id: str
    kind: TaskKind
    system: str
    messages: list[dict[str, Any]]
    tools: list[ToolDef] | None = None
    max_tokens: int = 300
    model_hint: str | None = None
    timeout_ms: int | None = None
    # Caller-provided cap; the executor refuses to start the subtask if the
    # cheapest available model's pessimistic cost would exceed this. None = no cap.
    cost_cap_usd: float | None = None
    # Free-form tags surfaced on the Langfuse span (e.g. session_id, turn_number).
    span_attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class FuguSubtaskResult:
    """What `FuguExecutor.run_subtask` returns.

    Mirrors the union of `LLMResponse` + `GenerationResult` fields so callers
    can treat both tool and non-tool subtasks uniformly. On non-OK status, the
    text/tool_calls fields are empty and `error` is set; callers decide
    whether to fall back or treat as soft-fail.
    """

    subtask_id: str
    kind: TaskKind
    status: SubtaskStatus
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_used: str = ""
    latency_ms: int = 0
    error: str | None = None
    # True iff the router downgraded from its preferred model for this subtask.
    degraded: bool = False

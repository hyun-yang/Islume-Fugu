"""Fugu in-process orchestration layer for Islume.

Fugu decomposes a single LLM request (e.g. one conversation turn) into a small
DAG of TaskKind-tagged subtasks, routes each subtask to the model best suited
for its cost/quality profile, and composes the results back into the legacy
shapes the rest of Islume already understands (`LLMResponse` /
`GenerationResult`-equivalent fields, plus normalized `ToolCall`s).

Side-effect ownership is unchanged: this package only generates and routes.
Database writes, Redis stream publishes, intent_plugins handlers, wallet
mutations, and next-turn enqueueing remain in `services/worker` and
`services/orchestrator`.

This module is feature-flagged off by default (`settings.fugu_enabled`); the
legacy code path in `services/worker/main.py:_run_turn` is the canonical
behavior and Fugu must be a byte-equivalent drop-in when disabled.
"""

from __future__ import annotations

from shared.fugu.executor import FuguExecutor, bounded_gather
from shared.fugu.router import (
    DEFAULT_TASK_POLICY,
    FuguModelRouter,
    RoutingDecision,
)
from shared.fugu.turn_orchestrator import (
    FuguTurnOrchestrator,
    TurnOrchestrationInput,
    TurnOrchestrationResult,
)
from shared.fugu.types import (
    FuguSubtask,
    FuguSubtaskResult,
    SubtaskStatus,
    TaskKind,
)

__all__ = [
    "DEFAULT_TASK_POLICY",
    "FuguExecutor",
    "FuguModelRouter",
    "FuguSubtask",
    "FuguSubtaskResult",
    "FuguTurnOrchestrator",
    "RoutingDecision",
    "SubtaskStatus",
    "TaskKind",
    "TurnOrchestrationInput",
    "TurnOrchestrationResult",
    "bounded_gather",
]

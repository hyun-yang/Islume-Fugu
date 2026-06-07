"""Compose one conversation turn from multiple specialized subtasks.

v1 flow (kept intentionally small — Phase 2 of the plan):

  1. Preflight (parallel, cheap):
       - plugin_intent selector: when the speaker has 2+ attached plugins,
         pick which ones (and which of their tools) are actually plausible
         this turn. With 0 or 1 plugin the selector is skipped — there's
         nothing to choose.
  2. Persona draft (quality lane): the canonical user-facing reply. The
     system prompt is exactly the same one the worker has always built
     (`build_system_prompt`), so persona/safety/references/phase behavior
     is unchanged. Only the `tools` list passed to the LLM is filtered by
     the selector's choices.
  3. Safety review (cheap lane): only when the speaker has populated
     `safety` fields. The reviewer returns `{"ok": bool, "reasons": [...]}`.
       - On `ok=True` (or non-OK status / parse failure): pass through.
       - On `ok=False` with reasons: one revise pass that injects the
         reasons back into the persona draft system prompt as a
         non-negotiable correction. If the revise still fails for any
         reason, we keep the original draft and flip `degraded=True` so
         the worker/Langfuse can flag it.

This module never writes to the DB, never publishes to Redis, never invokes
plugin handlers, and never enqueues next turns. Those remain in the worker.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from shared.fugu.executor import FuguExecutor
from shared.fugu.types import FuguSubtask, FuguSubtaskResult
from shared.intent_plugins.base import Plugin, ToolCall, ToolDef

Phase = Literal["initial", "extended", "offline"]


@dataclass
class TurnOrchestrationInput:
    """Everything Fugu needs to produce one turn.

    `system_prompt_builder` is intentionally a callable — the worker passes
    `build_system_prompt` so Fugu doesn't pull a hard dependency on the
    worker module. The callable signature matches
    `worker.build_system_prompt(speaker, listener_user_name, match_context,
    phase, plugins)`.
    """

    session_id: UUID
    turn_number: int
    speaker: Any           # shared.models.Agent — typed Any to avoid import cycle
    listener: Any          # shared.models.Agent
    listener_user_name: str
    match_context: str
    phase: Phase
    history: list[dict[str, Any]]
    is_opening: bool
    attached_plugins: list[tuple[Plugin, dict]]
    preferred_model: str | None
    system_prompt_builder: Any
    locale: str | None = None
    max_tokens: int = 300


@dataclass
class TurnOrchestrationResult:
    """Output of `FuguTurnOrchestrator.run`.

    Field names line up with what the worker reads from `LLMResponse` /
    `GenerationResult` today so the integration point is a 1-for-1 swap.
    """

    text: str
    tool_calls: list[ToolCall]
    # Primary = the persona draft subtask. This is what gets written to
    # `ConversationTurn.model_used` so existing dashboards keep working.
    primary_model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    # True iff any load-bearing subtask was degraded or any optional subtask
    # silently fell back (e.g. safety review returned non-OK status).
    degraded: bool = False
    # Per-subtask audit suitable for OTel/Langfuse and integration tests.
    subtask_audit: list[dict[str, Any]] = field(default_factory=list)


class FuguTurnOrchestrator:
    def __init__(
        self,
        executor: FuguExecutor | None = None,
        *,
        plugin_selector_min_count: int = 2,
        turn_budget_usd: float | None = None,
    ) -> None:
        self.executor = executor or FuguExecutor()
        self.plugin_selector_min_count = plugin_selector_min_count
        self.turn_budget_usd = turn_budget_usd

    async def run(self, input: TurnOrchestrationInput) -> TurnOrchestrationResult:
        t0 = time.monotonic()
        audit: list[dict[str, Any]] = []
        budget_remaining = self.turn_budget_usd

        # ---- Step 1: plugin intent selector (only when worth it) -----------
        active_plugins = input.attached_plugins
        if (
            len(input.attached_plugins) >= self.plugin_selector_min_count
            and not input.is_opening
        ):
            selector_subtask = _build_plugin_selector_subtask(input)
            selector_res = await self.executor.run_subtask(
                selector_subtask, budget_remaining_usd=budget_remaining
            )
            audit.append(_audit(selector_res))
            budget_remaining = _decrement_budget(budget_remaining, selector_res)
            chosen_ids = _parse_plugin_selection(
                selector_res, [p.id for p, _ in input.attached_plugins]
            )
            if chosen_ids is not None:
                active_plugins = [
                    (p, policy)
                    for p, policy in input.attached_plugins
                    if p.id in chosen_ids
                ]

        # ---- Step 2: persona draft (the user-facing reply) -----------------
        tools_for_turn = _flatten_tools(active_plugins)
        draft_subtask = _build_persona_subtask(input, active_plugins, tools_for_turn)
        draft_res = await self.executor.run_subtask(
            draft_subtask, budget_remaining_usd=budget_remaining
        )
        audit.append(_audit(draft_res))
        budget_remaining = _decrement_budget(budget_remaining, draft_res)

        # The persona draft is load-bearing — if it failed outright, surface
        # an empty turn and let the worker decide (caller fallback path).
        if draft_res.status != "ok":
            return TurnOrchestrationResult(
                text="",
                tool_calls=[],
                primary_model=draft_res.model_used,
                input_tokens=draft_res.input_tokens,
                output_tokens=draft_res.output_tokens,
                cost_usd=draft_res.cost_usd,
                latency_ms=int((time.monotonic() - t0) * 1000),
                degraded=True,
                subtask_audit=audit,
            )

        text = draft_res.text
        tool_calls = list(draft_res.tool_calls)
        primary_model = draft_res.model_used
        input_tokens = draft_res.input_tokens
        output_tokens = draft_res.output_tokens
        cost_usd = draft_res.cost_usd
        degraded = draft_res.degraded

        # ---- Step 3: safety review + optional revise -----------------------
        if _has_safety_rules(input.speaker) and text.strip():
            review_subtask = _build_safety_review_subtask(input, text)
            review_res = await self.executor.run_subtask(
                review_subtask, budget_remaining_usd=budget_remaining
            )
            audit.append(_audit(review_res))
            budget_remaining = _decrement_budget(budget_remaining, review_res)

            review = _parse_safety_review(review_res)
            if review is not None and not review.ok and review.reasons:
                # One revise pass with reasons appended to the original
                # persona system prompt as a non-negotiable correction.
                revise_subtask = _build_revise_subtask(
                    input,
                    active_plugins,
                    tools_for_turn,
                    prior_draft=text,
                    reasons=review.reasons,
                )
                revise_res = await self.executor.run_subtask(
                    revise_subtask, budget_remaining_usd=budget_remaining
                )
                audit.append(_audit(revise_res))
                if revise_res.status == "ok" and revise_res.text.strip():
                    text = revise_res.text
                    tool_calls = list(revise_res.tool_calls)
                    # Accumulate usage; primary_model still tracks the
                    # original draft for dashboard continuity.
                    input_tokens += revise_res.input_tokens
                    output_tokens += revise_res.output_tokens
                    cost_usd += revise_res.cost_usd
                degraded = True

        return TurnOrchestrationResult(
            text=text,
            tool_calls=tool_calls,
            primary_model=primary_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=int((time.monotonic() - t0) * 1000),
            degraded=degraded,
            subtask_audit=audit,
        )


# ---------------------------------------------------------------------------
# Subtask builders & helpers
# ---------------------------------------------------------------------------


def _flatten_tools(plugins: list[tuple[Plugin, dict]]) -> list[ToolDef]:
    out: list[ToolDef] = []
    for plugin, _ in plugins:
        out.extend(plugin.tools)
    return out


def _build_persona_subtask(
    input: TurnOrchestrationInput,
    active_plugins: list[tuple[Plugin, dict]],
    tools: list[ToolDef],
) -> FuguSubtask:
    system = input.system_prompt_builder(
        input.speaker,
        input.listener_user_name,
        input.match_context,
        phase=input.phase,
        plugins=active_plugins,
    )
    messages: list[dict[str, Any]]
    if input.is_opening:
        messages = [
            {"role": "user", "content": "(Start the conversation with a friendly opener.)"}
        ]
    else:
        messages = input.history

    return FuguSubtask(
        id=f"turn-{input.turn_number}-persona",
        kind="conversation_turn",
        system=system,
        messages=messages,
        tools=tools or None,
        max_tokens=input.max_tokens,
        model_hint=input.preferred_model,
        span_attributes={
            "fugu.session_id": str(input.session_id),
            "fugu.turn_number": str(input.turn_number),
            "fugu.is_opening": "1" if input.is_opening else "0",
        },
    )


def _build_plugin_selector_subtask(input: TurnOrchestrationInput) -> FuguSubtask:
    """Ask a cheap model which attached plugins are plausible this turn.

    Output contract: a JSON object `{"plugins": ["bartering", ...]}`.
    Anything malformed is treated as "select all" (defensive default).
    """
    plugin_list = [
        {
            "id": p.id,
            "description": p.description or "",
            "tools": [t.name for t in p.tools],
        }
        for p, _ in input.attached_plugins
    ]
    convo_tail = _format_history_tail(input.history, n=6)
    system = (
        "You are a routing classifier. The agent below has multiple intent "
        "plugins attached; pick the subset that is *plausibly* useful for the "
        "very next turn of the conversation. Default to including a plugin "
        "when uncertain. Respond with ONLY a JSON object of the form "
        '{"plugins": ["<plugin_id>", ...]}.'
    )
    user_payload = {
        "available_plugins": plugin_list,
        "match_context": input.match_context,
        "recent_turns": convo_tail,
    }
    return FuguSubtask(
        id=f"turn-{input.turn_number}-plugin-select",
        kind="plugin_intent",
        system=system,
        messages=[{"role": "user", "content": json.dumps(user_payload)}],
        max_tokens=120,
        span_attributes={
            "fugu.session_id": str(input.session_id),
            "fugu.turn_number": str(input.turn_number),
        },
    )


def _build_safety_review_subtask(
    input: TurnOrchestrationInput, draft_text: str
) -> FuguSubtask:
    safety = getattr(input.speaker, "safety", None) or {}
    redlines = list(safety.get("redline_topics") or [])
    confirm_for = list(safety.get("require_owner_confirmation_for") or [])
    refuse_pii = bool(safety.get("refuse_personal_info_share"))
    rules = {
        "refuse_personal_info_share": refuse_pii,
        "redline_topics": redlines,
        "require_owner_confirmation_for": confirm_for,
    }
    system = (
        "You are a safety reviewer. Given a draft reply and a set of safety "
        "rules, decide if the draft violates any of them. Respond with ONLY "
        "a raw JSON object — no markdown, no code fences, no commentary, no "
        "leading or trailing text. The object MUST be exactly: "
        '{"ok": true|false, "reasons": ["..."]}. '
        "If `ok` is true, `reasons` must be empty. Personal info refers to "
        "the agent's owner — real name, address, phone, email, or other "
        "contact info — only when `refuse_personal_info_share` is true."
    )
    user_payload = {"rules": rules, "draft": draft_text}
    return FuguSubtask(
        id=f"turn-{input.turn_number}-safety",
        kind="safety_review",
        system=system,
        messages=[{"role": "user", "content": json.dumps(user_payload)}],
        max_tokens=200,
        span_attributes={
            "fugu.session_id": str(input.session_id),
            "fugu.turn_number": str(input.turn_number),
        },
    )


def _build_revise_subtask(
    input: TurnOrchestrationInput,
    active_plugins: list[tuple[Plugin, dict]],
    tools: list[ToolDef],
    *,
    prior_draft: str,
    reasons: list[str],
) -> FuguSubtask:
    base_system = input.system_prompt_builder(
        input.speaker,
        input.listener_user_name,
        input.match_context,
        phase=input.phase,
        plugins=active_plugins,
    )
    correction = (
        "\n\nYour previous draft was flagged by safety review for the "
        "following reasons:\n- "
        + "\n- ".join(reasons)
        + "\n\nRewrite your reply addressing these issues. Keep your "
        "persona, tone, and topic. Do NOT explain the correction to the "
        "user; just speak naturally as the same character."
    )
    system = base_system + correction

    # Use the conversation history (or opener prompt) plus a hint of the
    # rejected draft so the model doesn't lose its place.
    if input.is_opening:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "(Start the conversation with a friendly opener.)",
            },
        ]
    else:
        messages = list(input.history)
    messages.append(
        {
            "role": "assistant",
            "content": prior_draft,
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "(safety reviewer requested a revision — please rewrite the above.)",
        }
    )
    return FuguSubtask(
        id=f"turn-{input.turn_number}-revise",
        kind="conversation_turn",
        system=system,
        messages=messages,
        tools=tools or None,
        max_tokens=input.max_tokens,
        # Force cheap lane for the revise to keep latency/cost bounded.
        model_hint=None,
        span_attributes={
            "fugu.session_id": str(input.session_id),
            "fugu.turn_number": str(input.turn_number),
            "fugu.revise": "1",
        },
    )


def _has_safety_rules(speaker: Any) -> bool:
    safety = getattr(speaker, "safety", None)
    if not safety:
        return False
    return bool(
        safety.get("refuse_personal_info_share")
        or safety.get("redline_topics")
        or safety.get("require_owner_confirmation_for")
    )


def _format_history_tail(history: list[dict[str, Any]], *, n: int) -> list[dict[str, str]]:
    tail = history[-n:] if len(history) > n else history
    return [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))[:400]}
        for m in tail
    ]


@dataclass
class _SafetyReview:
    ok: bool
    reasons: list[str]


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort recovery of a JSON object from a chatty LLM reply.

    Models that have been told "respond with ONLY JSON" still wrap output in
    ```json … ``` fences (Anthropic Claude in particular) or append a sentence
    or two of explanation. We strip a leading code fence first, then fall back
    to extracting the first ``{...}`` block. Returns None if no dict is found.
    """
    stripped = text.strip()
    # Drop a single leading code fence + optional ``json`` tag.
    if stripped.startswith("```"):
        # Remove the opening fence + optional language tag on the same line.
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        # Remove a trailing closing fence (with or without trailing prose).
        end = stripped.find("```")
        if end != -1:
            stripped = stripped[:end]
        stripped = stripped.strip()

    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    match = _JSON_OBJECT_RE.search(stripped)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parse_safety_review(res: FuguSubtaskResult) -> _SafetyReview | None:
    """Parse the reviewer's reply into a `_SafetyReview`.

    Returns None ONLY when the subtask itself failed (timeout / provider
    error) — in that case the caller treats the review as "absent" and the
    draft passes through unchanged. On *parsable but malformed* output we
    fail closed: we synthesize ``ok=False`` with a sentinel reason so the
    revise step runs instead of silently shipping an un-reviewed draft. This
    is the conservative choice for a safety gate; we'd rather waste one
    cheap revise call than let a violation through because the reviewer
    decided to wrap its JSON in a markdown fence.
    """
    if res.status != "ok" or not res.text.strip():
        return None

    data = _extract_json_object(res.text)
    if data is None:
        # Fail closed — treat as "needs revise".
        return _SafetyReview(ok=False, reasons=["safety review parse failed"])

    # ``ok`` defaults to False on a malformed object (e.g. missing key) so
    # we still err on the side of running the revise pass.
    ok = bool(data.get("ok", False))
    raw_reasons = data.get("reasons") or []
    reasons = [str(r) for r in raw_reasons if isinstance(r, (str, int, float))]
    if not ok and not reasons:
        reasons = ["safety review reported violation without reasons"]
    return _SafetyReview(ok=ok, reasons=reasons)


def _parse_plugin_selection(
    res: FuguSubtaskResult, valid_ids: list[str]
) -> set[str] | None:
    """Return the intersection of (parsed selection) ∩ (valid_ids).

    Returns None when parsing fails — caller falls back to "all plugins
    active", which matches today's behavior.
    """
    if res.status != "ok" or not res.text.strip():
        return None
    try:
        data = json.loads(res.text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("plugins")
    if not isinstance(raw, list):
        return None
    valid = set(valid_ids)
    chosen = {str(p) for p in raw if isinstance(p, str)} & valid
    # An empty selection is suspect (model wanted to deactivate everything);
    # we prefer the legacy "all on" behavior in that case.
    if not chosen:
        return None
    return chosen


def _decrement_budget(
    remaining: float | None, res: FuguSubtaskResult
) -> float | None:
    if remaining is None:
        return None
    return max(0.0, remaining - res.cost_usd)


def _audit(res: FuguSubtaskResult) -> dict[str, Any]:
    return {
        "subtask_id": res.subtask_id,
        "kind": res.kind,
        "status": res.status,
        "model": res.model_used,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "cost_usd": res.cost_usd,
        "latency_ms": res.latency_ms,
        "degraded": res.degraded,
        "error": res.error,
    }

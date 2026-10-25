"""A minimal, provider-neutral agent loop in which every tool call goes through SAW.

Use it directly, or read it as the reference for wiring SAW into your own
loop. It exists in the package so that the easy path is also the safe one:

* Every tool call goes through ``security.execute_tool``. The loop never
  calls a tool function itself.
* The model only sees ``outcome.for_model()``: results of executed calls,
  or a status and a safe reason. A call waiting for approval is reported as
  *not performed*.
* The model is only told about tools the agent is allowed to use
  (``security.tool_definitions()``).
* When a call needs approval, the loop **waits** for ``approval_handler``
  (your application, asking a person) and records the decision through the
  :class:`~saw.ApprovalDesk`. The model has no way to make that decision.
* Every tool use the model asks for gets an answer, so provider rules such
  as "each tool use needs a tool result" always hold.

A model is any object with ``next_step(messages, tools)`` returning a
:class:`ModelTurn` (text and/or tool uses) or a :class:`FinalAnswer`. See
``saw.adapters`` for Anthropic and OpenAI(-compatible) models, and
:class:`ScriptedModel` for tests and offline demos.

Messages are provider-neutral dicts:

* ``{"role": "user", "content": str}``
* ``{"role": "assistant", "content": str, "tool_uses": [{"id", "name", "arguments"}]}``
* ``{"role": "tool_results", "results": [{"tool_use_id", "name", "content", "is_error"}]}``
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, Union

from saw.approvals import ApprovalDesk, ApprovalRequest
from saw.calls import CallStatus, ToolCallOutcome
from saw.exceptions import ConfigurationError
from saw.runtime import SAW


@dataclass(frozen=True)
class ToolUse:
    """The model asks for one tool.

    ``arguments`` is whatever the model produced. If it is not an object
    (for example malformed JSON passed through as a string), SAW reports
    ``invalid_arguments`` and nothing runs.
    """

    name: Any
    arguments: Any = None
    id: str = ""


@dataclass(frozen=True)
class ModelTurn:
    """A model reply that asks for one or more tools, optionally with text."""

    tool_uses: tuple[ToolUse, ...]
    text: str = ""


@dataclass(frozen=True)
class FinalAnswer:
    """The model is done."""

    text: str


Step = Union[ModelTurn, FinalAnswer]


class Model(Protocol):
    """Anything that chooses the next step from the conversation so far."""

    def next_step(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Step: ...


@dataclass
class ScriptedModel:
    """Replays fixed steps; records what it was shown. For tests and demos.

    Steps may be a :class:`ToolUse`, a list of them (several tools in one
    turn), a :class:`ModelTurn` or a :class:`FinalAnswer`.
    """

    steps: Sequence[Any]
    seen_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    seen_tools: list[list[dict[str, Any]]] = field(default_factory=list)
    _index: int = 0

    def next_step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Step:
        self.seen_messages.append([dict(m) for m in messages])
        self.seen_tools.append(list(tools))
        if self._index >= len(self.steps):
            return FinalAnswer("(script finished)")
        step = self.steps[self._index]
        self._index += 1
        if isinstance(step, ToolUse):
            return ModelTurn((step,))
        if isinstance(step, (list, tuple)):
            return ModelTurn(tuple(step))
        return step


#: Decides an approval request for the application: returns (approve?, reason).
ApprovalHandler = Callable[[ApprovalRequest], "tuple[bool, str]"]


@dataclass
class Turn:
    """One tool call in a run: what the model asked for and what happened."""

    tool_use: ToolUse
    outcome: ToolCallOutcome | None
    approval: str | None = None  # "approved", "denied", or None
    skipped: bool = False  # not attempted because max_tool_calls was reached


@dataclass
class Run:
    """The record of one agent run."""

    turns: list[Turn]
    answer: str
    messages: list[dict[str, Any]]
    stopped_early: bool = False


_NOT_ATTEMPTED = {
    "status": "not_attempted",
    "error": "The tool-call limit for this task was reached.",
    "note": "This action was not performed.",
}


def run_agent(
    security: SAW,
    model: Model,
    user_message: str,
    *,
    approval_desk: ApprovalDesk | None = None,
    approval_handler: ApprovalHandler | None = None,
    approver: str = "user",
    max_steps: int = 20,
    max_tool_calls: int = 50,
    on_turn: Callable[[Turn], None] | None = None,
) -> Run:
    """Run ``model`` until it answers, sending every tool call through SAW.

    Args:
        security: The agent's SAW runtime.
        model: See :class:`Model`.
        user_message: The user's request.
        approval_desk: The desk given to ``security``. Needed only with
            ``approval_handler``.
        approval_handler: Called when a call needs approval; the loop waits
            for its answer. Without one, such calls stay pending on the desk
            (or are denied if the agent has no desk) and the model is told
            the action was not performed.
        approver: Name recorded for decisions made by ``approval_handler``.
            Put the name your application authenticated here.
        max_steps: Maximum model turns.
        max_tool_calls: Maximum tool calls in the run. Further tool uses are
            answered as "not attempted" and the run stops.
        on_turn: Called after each tool call, for display or logging.

    Raises:
        ConfigurationError: ``approval_handler`` without ``approval_desk``.
    """
    if approval_handler is not None and approval_desk is None:
        raise ConfigurationError("approval_handler needs the agent's approval_desk")

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    turns: list[Turn] = []
    tool_calls = 0

    for _ in range(max_steps):
        step = model.next_step(messages, security.tool_definitions())
        if isinstance(step, FinalAnswer):
            messages.append({"role": "assistant", "content": step.text, "tool_uses": []})
            return Run(turns, step.text, messages)
        if not isinstance(step, ModelTurn) or not step.tool_uses:
            text = step.text if isinstance(step, ModelTurn) else ""
            messages.append({"role": "assistant", "content": text, "tool_uses": []})
            return Run(turns, text, messages)

        uses = [
            use if use.id else ToolUse(use.name, use.arguments, f"call_{tool_calls + i + 1}")
            for i, use in enumerate(step.tool_uses)
        ]
        messages.append({
            "role": "assistant",
            "content": step.text,
            "tool_uses": [{"id": u.id, "name": u.name, "arguments": u.arguments} for u in uses],
        })

        results = []
        limit_reached = False
        for use in uses:
            if tool_calls >= max_tool_calls:
                limit_reached = True
                turn = Turn(use, None, skipped=True)
                content, is_error = dict(_NOT_ATTEMPTED), True
            else:
                tool_calls += 1
                turn = _call_tool(security, use, approval_desk, approval_handler, approver)
                content, is_error = turn.outcome.for_model(), not turn.outcome.ok
            turns.append(turn)
            if on_turn is not None:
                on_turn(turn)
            results.append({
                "tool_use_id": use.id,
                "name": use.name if isinstance(use.name, str) else str(use.name),
                "content": content,
                "is_error": is_error,
            })
        messages.append({"role": "tool_results", "results": results})
        if limit_reached:
            return Run(turns, "(stopped: tool-call limit reached)", messages, stopped_early=True)

    return Run(turns, "(stopped: step limit reached)", messages, stopped_early=True)


def _call_tool(
    security: SAW,
    use: ToolUse,
    desk: ApprovalDesk | None,
    handler: ApprovalHandler | None,
    approver: str,
) -> Turn:
    outcome = security.execute_tool(use.name, use.arguments)
    if outcome.status is not CallStatus.APPROVAL_REQUIRED or handler is None:
        return Turn(use, outcome)

    request = outcome.approval_request
    approved, why = handler(request)  # the loop waits here for the person
    why = why if isinstance(why, str) and why.strip() else "no reason given"
    if approved is True:
        return Turn(use, desk.approve(request.request_id, approver=approver, note=why), "approved")
    return Turn(use, desk.deny(request.request_id, approver=approver, reason=why), "denied")


def describe_turn(turn: Turn, width: int = 60) -> str:
    """A one-line description of a turn, for console output."""
    use = turn.tool_use
    if isinstance(use.arguments, dict):
        args = ", ".join(f"{k}={v!r}" for k, v in use.arguments.items())
    else:
        args = repr(use.arguments)
    label = f"{use.name}({args})"
    if len(label) > width:
        label = label[: width - 3] + "..."
    if turn.skipped:
        return f"{label:<{width}} -> not attempted (tool-call limit)"
    outcome = turn.outcome
    note = f" [{turn.approval}]" if turn.approval else ""
    detail = "" if outcome.ok else f" - {outcome.reason}"
    return f"{label:<{width}} -> {outcome.status.value}{note}{detail}"


__all__ = [
    "ToolUse", "ModelTurn", "FinalAnswer", "Model", "ScriptedModel",
    "Turn", "Run", "run_agent", "describe_turn", "ApprovalHandler",
]

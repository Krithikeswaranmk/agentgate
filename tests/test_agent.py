"""The packaged agent loop (saw.agent) and tool definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pytest

from saw import SAW, AgentConfig, ApprovalDesk, CallStatus, ConfigurationError
from saw.agent import FinalAnswer, ModelTurn, ScriptedModel, ToolUse, describe_turn, run_agent
from conftest import RecordingLogger


@dataclass
class EmailArgs:
    to: str
    body: str
    priority: Literal["low", "high"] = "low"
    cc: Optional[list[str]] = None


@dataclass
class DayArgs:
    day: str


def assistant(desk=None, **config):
    config.setdefault("allowed_tools", ["calendar", "send_email"])
    config.setdefault("approval_required", ["send_email"])
    sent, ran = [], []
    security = SAW(
        config=AgentConfig(agent_id="pa", **config), logger=RecordingLogger(), approval_desk=desk,
    )

    def calendar(day: str) -> list:
        """List events for a day.

        Longer explanation that is not part of the description."""
        ran.append(day)
        return ["09:00 stand-up"]

    security.register_tool("calendar", schema=DayArgs, function=calendar)
    security.register_tool(
        "send_email", schema=EmailArgs, function=lambda to, body, priority="low", cc=None: sent.append(to),
        description="Send an email.",
    )
    security.register_tool("delete_all", schema=DayArgs, function=lambda day: "gone")
    return security, sent, ran


# ---------------------------------------------------------- tool definitions


def test_tool_definitions_list_only_allowed_registered_tools():
    security, _, _ = assistant()
    definitions = security.tool_definitions()
    assert [d["name"] for d in definitions] == ["calendar", "send_email"]  # not delete_all
    calendar = definitions[0]
    assert calendar["description"] == "List events for a day."
    assert calendar["parameters"] == {
        "type": "object", "properties": {"day": {"type": "string"}},
        "required": ["day"], "additionalProperties": False,
    }
    email = definitions[1]["parameters"]
    assert email["required"] == ["to", "body"]
    assert email["properties"]["priority"] == {"enum": ["low", "high"]}
    assert email["properties"]["cc"] == {
        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
    }


def test_allowed_but_unregistered_tools_are_not_listed():
    security, _, _ = assistant(allowed_tools=["calendar", "send_email", "ghost"])
    assert "ghost" not in [d["name"] for d in security.tool_definitions()]


def test_tool_definitions_are_independent_copies():
    security, _, _ = assistant()
    security.tool_definitions()[0]["parameters"]["properties"].clear()
    assert security.tool_definitions()[0]["parameters"]["properties"] == {"day": {"type": "string"}}


def test_description_validation():
    security, _, _ = assistant()
    with pytest.raises(ConfigurationError, match="1024"):
        security.register_tool("x", schema=DayArgs, function=lambda day: 1, description="d" * 1025)
    with pytest.raises(ConfigurationError, match="must be a string"):
        security.register_tool("y", schema=DayArgs, function=lambda day: 1, description=5)


# ------------------------------------------------------------- the loop


def test_loop_runs_tools_through_saw_and_answers():
    security, sent, ran = assistant()
    model = ScriptedModel([
        ToolUse("calendar", {"day": "mon"}, "t1"),
        ToolUse("delete_all", {"day": "now"}, "t2"),
        FinalAnswer("done"),
    ])
    run = run_agent(security, model, "check")

    assert [t.outcome.status for t in run.turns] == [CallStatus.EXECUTED, CallStatus.DENIED]
    assert ran == ["mon"] and run.answer == "done"
    # The model was told about allowed tools only.
    assert [t["name"] for t in model.seen_tools[0]] == ["calendar", "send_email"]
    assert run.messages[1] == {
        "role": "assistant", "content": "",
        "tool_uses": [{"id": "t1", "name": "calendar", "arguments": {"day": "mon"}}],
    }
    assert run.messages[2]["results"][0]["content"] == {"status": "executed", "result": ["09:00 stand-up"]}
    assert run.messages[2]["results"][0]["is_error"] is False


def test_several_tool_uses_in_one_turn_all_get_results():
    security, sent, ran = assistant()
    model = ScriptedModel([
        [ToolUse("calendar", {"day": "mon"}), ToolUse("calendar", {"day": "tue"}), ToolUse("nope", {})],
        FinalAnswer("ok"),
    ])
    run = run_agent(security, model, "check")
    results = run.messages[2]["results"]
    assert [r["tool_use_id"] for r in results] == ["call_1", "call_2", "call_3"]
    assert [r["is_error"] for r in results] == [False, False, True]
    assert ran == ["mon", "tue"]


def test_malformed_model_output_never_runs_anything():
    security, sent, ran = assistant(approval_required=[])
    model = ScriptedModel([
        [ToolUse(None, {}), ToolUse(["calendar"], {}), ToolUse("calendar", '{"day": "mon"'),
         ToolUse("calendar", None), ToolUse("send_email", {"to": 1, "body": "x"})],
        FinalAnswer("x"),
    ])
    run = run_agent(security, model, "go")
    assert [t.outcome.status for t in run.turns] == [
        CallStatus.UNKNOWN_TOOL, CallStatus.UNKNOWN_TOOL, CallStatus.INVALID_ARGUMENTS,
        CallStatus.INVALID_ARGUMENTS, CallStatus.INVALID_ARGUMENTS,
    ]
    assert sent == [] and ran == []


def test_approval_handler_decides_through_the_desk():
    desk = ApprovalDesk(approvers={"alice"})
    security, sent, _ = assistant(desk)
    decisions = iter([(True, "fine"), (False, "no")])
    asked = []

    def handler(request):
        asked.append(request.arguments["to"])
        return next(decisions)

    model = ScriptedModel([
        ToolUse("send_email", {"to": "a@example.com", "body": "1"}),
        ToolUse("send_email", {"to": "b@example.com", "body": "2"}),
        FinalAnswer("done"),
    ])
    run = run_agent(security, model, "send", approval_desk=desk, approval_handler=handler, approver="alice")
    assert [t.approval for t in run.turns] == ["approved", "denied"]
    assert sent == ["a@example.com"] and asked == ["a@example.com", "b@example.com"]
    assert desk.pending() == ()


def test_handler_must_return_true_to_approve():
    desk = ApprovalDesk()
    security, sent, _ = assistant(desk)
    model = ScriptedModel([ToolUse("send_email", {"to": "a@example.com", "body": "1"}), FinalAnswer("x")])
    run = run_agent(security, model, "go", approval_desk=desk, approval_handler=lambda r: ("yes", ""))
    assert run.turns[0].approval == "denied" and sent == []


def test_handler_without_desk_is_a_configuration_error():
    security, _, _ = assistant()
    with pytest.raises(ConfigurationError, match="approval_desk"):
        run_agent(security, ScriptedModel([]), "x", approval_handler=lambda r: (True, ""))


def test_handler_exception_leaves_request_pending_and_unrun():
    desk = ApprovalDesk()
    security, sent, _ = assistant(desk)

    def broken(request):
        raise RuntimeError("UI crashed")

    with pytest.raises(RuntimeError):
        run_agent(security, ScriptedModel([ToolUse("send_email", {"to": "a@example.com", "body": "1"})]),
                  "go", approval_desk=desk, approval_handler=broken)
    assert sent == [] and len(desk.pending()) == 1


def test_step_limit():
    security, _, _ = assistant()
    model = ScriptedModel([ToolUse("calendar", {"day": "x"}) for _ in range(5)])
    run = run_agent(security, model, "loop", max_steps=3)
    assert len(run.turns) == 3 and run.stopped_early


def test_tool_call_limit_answers_every_tool_use():
    security, _, ran = assistant()
    model = ScriptedModel([[ToolUse("calendar", {"day": str(i)}) for i in range(5)]])
    run = run_agent(security, model, "many", max_tool_calls=2)
    assert ran == ["0", "1"]
    results = run.messages[2]["results"]
    assert len(results) == 5
    assert [r["content"]["status"] for r in results[2:]] == ["not_attempted"] * 3
    assert run.stopped_early and run.turns[-1].skipped


def test_model_turn_without_tools_ends_the_run():
    security, _, _ = assistant()
    run = run_agent(security, ScriptedModel([ModelTurn((), "just text")]), "x")
    assert run.answer == "just text" and run.turns == []


def test_describe_turn():
    security, _, _ = assistant()
    run = run_agent(security, ScriptedModel([ToolUse("delete_all", {"day": "now"})]), "x")
    line = describe_turn(run.turns[0])
    assert line.startswith("delete_all(day='now')") and "denied" in line

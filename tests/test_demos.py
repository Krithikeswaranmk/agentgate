"""Three demo agents on one package, run through saw.agent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).resolve().parent.parent / "examples" / "agents"
sys.path.insert(0, str(AGENTS_DIR))

import coding_agent  # noqa: E402
import personal_assistant  # noqa: E402
import research_agent  # noqa: E402

import saw  # noqa: E402
from saw import ApprovalDesk, CallStatus, EventType  # noqa: E402
from saw.agent import FinalAnswer, ScriptedModel, ToolUse, run_agent  # noqa: E402
from conftest import RecordingLogger  # noqa: E402

S = CallStatus


@pytest.fixture(autouse=True)
def empty_outbox():
    personal_assistant.OUTBOX.clear()
    yield
    personal_assistant.OUTBOX.clear()


def statuses(run):
    return [t.outcome.status for t in run.turns]


def test_research_agent():
    security = research_agent.build(RecordingLogger())
    run = run_agent(security, research_agent.model(), research_agent.TASK)
    assert statuses(run) == [S.EXECUTED, S.EXECUTED, S.DENIED, S.DENIED, S.DENIED]
    assert "not in allowed_tools" in run.turns[4].outcome.reason


def test_coding_agent():
    security = coding_agent.build(RecordingLogger())
    run = run_agent(security, coding_agent.model(), coding_agent.TASK)
    assert statuses(run) == [
        S.EXECUTED, S.EXECUTED, S.UNKNOWN_TOOL, S.DENIED, S.INVALID_ARGUMENTS, S.RESULT_REJECTED,
    ]
    assert "shell" not in security.registered_tools


def test_personal_assistant_approval_flow():
    logger = RecordingLogger()
    desk = ApprovalDesk()
    security = personal_assistant.build(logger, desk)
    user = personal_assistant.ScriptedUser()
    run = run_agent(
        security, personal_assistant.model(), personal_assistant.TASK,
        approval_desk=desk, approval_handler=user,
    )

    assert statuses(run) == [S.EXECUTED, S.EXECUTED, S.DENIED, S.DENIED]
    assert [t.approval for t in run.turns] == [None, "approved", "denied", None]
    assert len(user.asked) == 2
    assert personal_assistant.OUTBOX == [
        {"to": "user@example.com", "body": "Meeting update: design review at 14:00."}
    ]
    assert desk.pending() == ()
    kinds = [e.event_type for e in logger.events]
    assert kinds.count(EventType.APPROVAL_GRANTED) == 1
    assert kinds.count(EventType.APPROVAL_DENIED) == 1


def test_personal_assistant_without_desk_denies_email():
    security = personal_assistant.build(RecordingLogger())
    run = run_agent(security, personal_assistant.model(), personal_assistant.TASK)
    assert statuses(run) == [S.EXECUTED, S.DENIED, S.DENIED, S.DENIED]
    assert personal_assistant.OUTBOX == []


def test_without_a_handler_approval_calls_stay_pending_and_never_run():
    desk = ApprovalDesk()
    security = personal_assistant.build(RecordingLogger(), desk)
    model = ScriptedModel([ToolUse("send_email", {"to": "user@example.com", "body": "hi"}, "x1"), FinalAnswer("done")])
    run = run_agent(security, model, "send it", approval_desk=desk)

    assert statuses(run) == [S.APPROVAL_REQUIRED]
    assert personal_assistant.OUTBOX == []
    assert len(desk.pending()) == 1
    result = run.messages[2]["results"][0]
    assert result["is_error"] is True
    assert "NOT been performed" in result["content"]["note"]
    assert "result" not in result["content"]


def test_agents_are_independent():
    agents = [research_agent.build(), coding_agent.build(), personal_assistant.build(desk=ApprovalDesk())]
    assert {type(a) for a in agents} == {saw.SAW}
    assert len({a.agent_id for a in agents}) == 3
    assert agents[0].config.allowed_tools == {"web_search", "read_document"}
    assert agents[1].config.allowed_tools == {"read_file", "run_tests"}
    assert agents[2].config.allowed_tools == {"calendar", "send_email"}
    assert agents[2].config.approval_required == {"send_email"}
    assert len({id(a.registry) for a in agents}) == 3
    assert len({type(a.permission_policy) for a in agents}) == 3
    # Tool definitions only advertise allowed tools: the research agent's
    # registered-but-forbidden send_email is never shown to its model.
    assert [d["name"] for d in agents[0].tool_definitions()] == ["read_document", "web_search"]
    assert agents[0].execute_tool("send_email", {"to": "a@example.com", "body": "x"}).status is S.DENIED


@pytest.mark.parametrize("answer, approved", [("y", True), ("n", False), ("", False), ("yes", False)])
def test_console_user(monkeypatch, answer, approved):
    desk = ApprovalDesk()
    security = personal_assistant.build(RecordingLogger(), desk)
    request = security.execute_tool("send_email", {"to": "user@example.com", "body": "hi"}).approval_request
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    assert personal_assistant.console_user(request)[0] is approved


def test_run_demos_script():
    result = subprocess.run(
        [sys.executable, str(AGENTS_DIR / "run_demos.py")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    out = result.stdout
    for heading in ("=== Research agent ===", "=== Coding agent ===", "=== Personal assistant ==="):
        assert heading in out
    assert "-> executed [approved]" in out
    assert "-> denied [denied]" in out
    assert "Approval requests still pending: 0" in out

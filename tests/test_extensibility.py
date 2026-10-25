"""Sprint 4: the three developer scenarios, and mandatory checks that survive
replacing any component."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from saw import (
    SAW,
    AgentConfig,
    ApprovalDesk,
    CallStatus,
    DefaultPermissionPolicy,
    DefaultResultValidator,
    DefaultSecurityLogger,
    EventType,
    PermissionPolicy,
    PolicyDecision,
    ResultValidator,
    SecurityLogger,
    ValidationResult,
)
from conftest import RecordingLogger

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TextArgs:
    text: str


@dataclass
class Counts:
    passed: bool
    failures: int

    def __post_init__(self):
        if self.failures < 0:
            raise ValueError("failures must be >= 0")


# ----------------------------------------------------- deliberately permissive


class AllowEverything(PermissionPolicy):
    def evaluate(self, call, config):
        return PolicyDecision.allow("allow everything")


class AcceptEverything(ResultValidator):
    def validate(self, call, result, config):
        return ValidationResult.accept()


class FileLikeLogger(SecurityLogger):
    def __init__(self):
        self.lines: list[str] = []

    def log(self, event):
        self.lines.append(json.dumps(event.to_dict(), default=str))


# ------------------------------------------------------------ the three devs


def test_research_developer_custom_logger(tmp_path):
    class ResearchLogger(SecurityLogger):
        def __init__(self, path):
            self.path = path

        def log(self, event):
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")

    path = tmp_path / "events.jsonl"
    security = SAW(agent_id="research-agent", logger=ResearchLogger(path))
    security.register_tool("web_search", schema=TextArgs, function=lambda text: [text])

    # agent_id only: no tools allowed, so permission enforcement is unchanged.
    outcome = security.execute_tool("web_search", {"text": "q"})

    assert outcome.status is CallStatus.DENIED
    kinds = [json.loads(line)["event_type"] for line in path.read_text().splitlines()]
    assert kinds == ["agent_initialized", "tool_call_received", "tool_call_denied"]
    assert isinstance(security.permission_policy, DefaultPermissionPolicy)
    assert isinstance(security.tool_result_validator, DefaultResultValidator)


def test_research_logger_failure_still_blocks(tmp_path):
    class ResearchLogger(SecurityLogger):
        def __init__(self, path):
            self.path = path

        def log(self, event):
            with self.path.open("a", encoding="utf-8") as f:
                f.write("x\n")

    path = tmp_path / "events.jsonl"
    security = SAW(
        config=AgentConfig(agent_id="research-agent", allowed_tools=["web_search"]),
        logger=ResearchLogger(path),
    )
    ran = []
    security.register_tool("web_search", schema=TextArgs, function=lambda text: ran.append(text))
    path.unlink()
    path.mkdir()  # writing now fails: the "file" is a directory

    from saw import AuditError

    with pytest.raises(AuditError):
        security.execute_tool("web_search", {"text": "q"})
    assert ran == []


def test_coding_developer_custom_validator():
    class CodingValidator(DefaultResultValidator):
        def validate(self, call, result, config):
            verdict = super().validate(call, result, config)
            if verdict.valid and "PRIVATE KEY" in str(result):
                return ValidationResult.reject("result appears to contain a credential")
            return verdict

    security = SAW(
        config=AgentConfig(agent_id="coding-agent", allowed_tools=["run_tests", "read_file"]),
        tool_result_validator=CodingValidator(),
        logger=RecordingLogger(),
    )
    outputs = iter([{"passed": True, "failures": 0}, {"passed": "yes", "failures": -1}])
    security.register_tool("run_tests", schema=TextArgs, function=lambda text: next(outputs), result_schema=Counts)
    security.register_tool("read_file", schema=TextArgs, function=lambda text: text)

    assert security.execute_tool("run_tests", {"text": "."}).ok
    assert security.execute_tool("run_tests", {"text": "."}).status is CallStatus.RESULT_REJECTED
    assert security.execute_tool("read_file", {"text": "-----BEGIN PRIVATE KEY-----"}).status is CallStatus.RESULT_REJECTED
    assert security.execute_tool("read_file", {"text": "x" * 100_001}).status is CallStatus.RESULT_REJECTED
    assert security.execute_tool("read_file", {"text": b"x"}).status is CallStatus.INVALID_ARGUMENTS
    assert security.execute_tool("delete", {"text": "x"}).status is CallStatus.UNKNOWN_TOOL


def test_personal_developer_custom_policy():
    class PersonalPolicy(PermissionPolicy):
        def evaluate(self, call, config):
            if call.tool_name == "send_email" and not call.arguments["text"].endswith("@example.com"):
                return PolicyDecision.deny("email only within the user's domain")
            return PolicyDecision.allow("no personal rule applies")

    security = SAW(
        config=AgentConfig(
            agent_id="personal-agent",
            allowed_tools=["read_calendar", "send_email"],
            approval_required=["send_email"],
        ),
        permission_policy=PersonalPolicy(),
        logger=RecordingLogger(),
        approval_desk=ApprovalDesk(),
    )
    sent = []
    security.register_tool("read_calendar", schema=TextArgs, function=lambda text: ["09:00"])
    security.register_tool("send_email", schema=TextArgs, function=lambda text: sent.append(text))
    security.register_tool("delete_account", schema=TextArgs, function=lambda text: "gone")

    assert security.execute_tool("read_calendar", {"text": "mon"}).ok
    assert security.execute_tool("send_email", {"text": "x@other.org"}).status is CallStatus.DENIED
    assert security.execute_tool("send_email", {"text": "a@example.com"}).status is CallStatus.APPROVAL_REQUIRED
    assert security.execute_tool("delete_account", {"text": "x"}).status is CallStatus.DENIED
    assert sent == []


def test_example_script_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "custom_components.py")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "send_email to outsider (policy denies)           denied" in out
    assert "approval_required" in out
    assert "result appears to contain a credential" in out
    assert "over the limit of 100000" in out
    assert result.stderr == ""


# ------------------------------- mandatory checks survive any replacement


def build(policy, validator, logger, **config):
    config.setdefault("agent_id", "agent")
    security = SAW(
        config=AgentConfig(**config),
        permission_policy=policy,
        tool_result_validator=validator,
        logger=logger,
        approval_desk=ApprovalDesk(),
    )
    calls = []

    def record(text):
        calls.append(text)
        return text

    security.register_tool("echo", schema=TextArgs, function=record)
    security.register_tool("counts", schema=TextArgs, function=lambda text: {"passed": text}, result_schema=Counts)
    security.register_tool("blob", schema=TextArgs, function=lambda text: b"raw")
    return security, calls


COMBOS = list(itertools.product([None, AllowEverything], [None, AcceptEverything], [None, FileLikeLogger]))


@pytest.mark.parametrize("policy_cls, validator_cls, logger_cls", COMBOS)
def test_mandatory_checks_hold_for_every_combination(policy_cls, validator_cls, logger_cls):
    policy = policy_cls() if policy_cls else None
    validator = validator_cls() if validator_cls else None
    logger = logger_cls() if logger_cls else RecordingLogger()

    # Not allowed: denied, even with AllowEverything.
    security, calls = build(policy, validator, logger, allowed_tools=[])
    assert security.execute_tool("echo", {"text": "a"}).status is CallStatus.DENIED

    # Approval required: still required, even with AllowEverything.
    security, calls = build(policy, validator, logger, allowed_tools=["echo"], approval_required=["echo"])
    assert security.execute_tool("echo", {"text": "a"}).status is CallStatus.APPROVAL_REQUIRED

    security, calls = build(policy, validator, logger, allowed_tools=["echo", "counts"], max_result_chars=50)
    # Unknown tools and argument schemas: unchanged by any component.
    assert security.execute_tool("nope", {}).status is CallStatus.UNKNOWN_TOOL
    assert security.execute_tool("echo", {"text": 1}).status is CallStatus.INVALID_ARGUMENTS
    # Size limit: enforced by the runtime, even with AcceptEverything.
    assert security.execute_tool("echo", {"text": "x" * 51}).status is CallStatus.RESULT_REJECTED
    # Result schema: enforced, even with AcceptEverything.
    assert security.execute_tool("counts", {"text": "yes"}).status is CallStatus.RESULT_REJECTED
    # A valid call still works.
    assert security.execute_tool("echo", {"text": "ok"}).ok
    # Blocked calls never ran the tool (only the size-limited call and the valid call did).
    assert calls == ["x" * 51, "ok"]


@pytest.mark.parametrize("policy_cls, validator_cls", list(itertools.product([None, AllowEverything], [None, AcceptEverything])))
def test_every_decision_reaches_a_custom_logger(policy_cls, validator_cls):
    logger = FileLikeLogger()
    security, _ = build(
        policy_cls() if policy_cls else None,
        validator_cls() if validator_cls else None,
        logger,
        allowed_tools=["echo"],
    )
    security.execute_tool("echo", {"text": "ok"})
    security.execute_tool("blob", {"text": "x"})
    kinds = [json.loads(line)["event_type"] for line in logger.lines]
    assert kinds[0] == "agent_initialized"
    assert kinds.count("tool_call_received") == 2
    assert "tool_executed" in kinds and "tool_call_denied" in kinds


@pytest.mark.parametrize("validator_cls", [None, AcceptEverything])
def test_shape_checks_survive_any_validator(validator_cls):
    # Since Sprint 5 the JSON shape checks are mandatory runtime checks, so a
    # validator written from scratch cannot remove them.
    logger = RecordingLogger()
    security, _ = build(None, validator_cls() if validator_cls else None, logger, allowed_tools=["blob"])
    outcome = security.execute_tool("blob", {"text": "x"})
    assert outcome.status is CallStatus.RESULT_REJECTED
    assert logger.events[-1].details["check"] == "result_shape"


@pytest.mark.parametrize("policy_cls, validator_cls, logger_cls", COMBOS)
def test_components_are_independent(policy_cls, validator_cls, logger_cls):
    security = SAW(
        agent_id="a",
        permission_policy=policy_cls() if policy_cls else None,
        tool_result_validator=validator_cls() if validator_cls else None,
        logger=logger_cls() if logger_cls else None,
    )
    assert type(security.permission_policy) is (policy_cls or DefaultPermissionPolicy)
    assert type(security.tool_result_validator) is (validator_cls or DefaultResultValidator)
    assert type(security.logger) is (logger_cls or DefaultSecurityLogger)

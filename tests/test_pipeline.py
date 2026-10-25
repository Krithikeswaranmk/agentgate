"""Integration tests: complete allowed and blocked paths through SAW.execute_tool."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from saw import (
    SAW,
    AgentConfig,
    ApprovalDesk,
    ApprovalRequiredError,
    AuditError,
    CallStatus,
    EventType,
    PermissionDeniedError,
    PermissionPolicy,
    PolicyDecision,
    SecurityLogger,
    ToolRegistry,
)
from conftest import RecordingLogger


@dataclass
class ReadDocumentArgs:
    path: str


@dataclass
class SendEmailArgs:
    to: str
    body: str


@dataclass
class NoArgs:
    pass


class Spy:
    """A tool function that records every call."""

    def __init__(self, result="ok", raises: Exception | None = None):
        self.calls: list[dict] = []
        self.result = result
        self.raises = raises

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return self.result


def make_saw(logger=None, policy=None, **config):
    config.setdefault("agent_id", "research-agent")
    return SAW(
        config=AgentConfig(**config), logger=logger or RecordingLogger(),
        permission_policy=policy, approval_desk=ApprovalDesk(),
    )


def call_events(logger: RecordingLogger, call_id: str):
    return [e for e in logger.events if e.call_id == call_id]


def types_of(events):
    return [e.event_type for e in events]


# ---------------------------------------------------------------- allowed path


def test_valid_authorised_call_executes():
    logger = RecordingLogger()
    security = make_saw(logger, allowed_tools=["read_document"])
    spy = Spy(result="report contents")
    security.register_tool(name="read_document", schema=ReadDocumentArgs, function=spy)

    outcome = security.execute_tool("read_document", {"path": "report.txt"})

    assert outcome.ok
    assert outcome.status is CallStatus.EXECUTED
    assert outcome.value == "report contents"
    assert spy.calls == [{"path": "report.txt"}]
    assert types_of(call_events(logger, outcome.call_id)) == [
        EventType.TOOL_CALL_RECEIVED,
        EventType.TOOL_CALL_ALLOWED,
        EventType.TOOL_EXECUTED,
    ]


def test_tool_receives_validated_values():
    @dataclass
    class Ratio:
        value: float

    security = make_saw(allowed_tools=["scale"])
    spy = Spy()
    security.register_tool("scale", schema=Ratio, function=spy)
    security.execute_tool("scale", {"value": 2})
    assert spy.calls == [{"value": 2.0}]
    assert isinstance(spy.calls[0]["value"], float)


def test_no_arguments_tool_accepts_none_and_empty():
    security = make_saw(allowed_tools=["now"])
    spy = Spy()
    security.register_tool("now", schema=NoArgs, function=spy)
    assert security.execute_tool("now").ok
    assert security.execute_tool("now", {}).ok
    assert len(spy.calls) == 2


def test_caller_mutation_after_call_does_not_affect_tool_input():
    security = make_saw(allowed_tools=["read_document"])
    seen = []
    security.register_tool("read_document", schema=ReadDocumentArgs, function=lambda path: seen.append(path))
    args = {"path": "a.txt"}
    security.execute_tool("read_document", args)
    args["path"] = "changed"
    assert seen == ["a.txt"]


# ---------------------------------------------------------------- blocked paths


BLOCKED_CASES = [
    # (tool_name, arguments, expected status, allowed_tools)
    ("delete_files", {"path": "x"}, CallStatus.UNKNOWN_TOOL, ["read_document"]),
    (None, {}, CallStatus.UNKNOWN_TOOL, ["read_document"]),
    (["read_document"], {}, CallStatus.UNKNOWN_TOOL, ["read_document"]),
    ("read_document", {"path": 1}, CallStatus.INVALID_ARGUMENTS, ["read_document"]),
    ("read_document", {}, CallStatus.INVALID_ARGUMENTS, ["read_document"]),
    ("read_document", {"path": "a", "mode": "w"}, CallStatus.INVALID_ARGUMENTS, ["read_document"]),
    ("read_document", "path=a", CallStatus.INVALID_ARGUMENTS, ["read_document"]),
    ("read_document", ["a"], CallStatus.INVALID_ARGUMENTS, ["read_document"]),
    ("read_document", {1: "a"}, CallStatus.INVALID_ARGUMENTS, ["read_document"]),
    ("read_document", {"path": "a"}, CallStatus.DENIED, []),
    ("read_document", {"path": "a"}, CallStatus.DENIED, ["other_tool"]),
]


@pytest.mark.parametrize("tool_name, arguments, status, allowed", BLOCKED_CASES)
def test_blocked_call_never_invokes_function(tool_name, arguments, status, allowed):
    logger = RecordingLogger()
    security = make_saw(logger, allowed_tools=allowed)
    spy = Spy()
    security.register_tool("read_document", schema=ReadDocumentArgs, function=spy)

    outcome = security.execute_tool(tool_name, arguments)

    assert outcome.status is status
    assert not outcome.ok
    assert spy.calls == []
    events = call_events(logger, outcome.call_id)
    assert types_of(events) == [EventType.TOOL_CALL_RECEIVED, EventType.TOOL_CALL_DENIED]
    denied = events[-1]
    assert denied.decision == "deny"
    assert denied.details["status"] == status.value
    assert denied.reason == outcome.reason


def test_blocked_call_value_raises():
    security = make_saw(allowed_tools=[])
    security.register_tool("read_document", schema=ReadDocumentArgs, function=Spy())
    outcome = security.execute_tool("read_document", {"path": "a"})
    with pytest.raises(PermissionDeniedError) as info:
        outcome.value
    assert info.value.outcome is outcome


def test_approval_required_blocks_and_records_event():
    logger = RecordingLogger()
    security = make_saw(
        logger,
        agent_id="personal-assistant",
        allowed_tools=["send_email"],
        approval_required=["send_email"],
    )
    spy = Spy()
    security.register_tool("send_email", schema=SendEmailArgs, function=spy)

    outcome = security.execute_tool("send_email", {"to": "a@b.c", "body": "hi"})

    assert outcome.status is CallStatus.APPROVAL_REQUIRED
    assert spy.calls == []
    events = call_events(logger, outcome.call_id)
    assert types_of(events) == [EventType.TOOL_CALL_RECEIVED, EventType.APPROVAL_REQUIRED]
    assert events[-1].decision == "require_approval"
    with pytest.raises(ApprovalRequiredError):
        outcome.value


def test_unknown_tool_checked_before_permission():
    # An agent with no permissions asking for an unregistered tool gets
    # UNKNOWN_TOOL, matching the agreed step order.
    security = make_saw(allowed_tools=[])
    assert security.execute_tool("ghost").status is CallStatus.UNKNOWN_TOOL


def test_arguments_validated_before_permission():
    security = make_saw(allowed_tools=[])
    security.register_tool("read_document", schema=ReadDocumentArgs, function=Spy())
    assert security.execute_tool("read_document", {"path": 1}).status is CallStatus.INVALID_ARGUMENTS


# ---------------------------------------------------------- configuration effect


def test_changing_configuration_changes_authorisation():
    registry = ToolRegistry()
    spy = Spy()
    registry.register("read_document", schema=ReadDocumentArgs, function=spy)
    args = {"path": "a.txt"}

    statuses = []
    for config in [
        AgentConfig(agent_id="a", allowed_tools=[]),
        AgentConfig(agent_id="a", allowed_tools=["read_document"]),
        AgentConfig(agent_id="a", allowed_tools=["read_document"], approval_required=["read_document"]),
    ]:
        security = SAW(
            config=config, registry=registry, logger=RecordingLogger(), approval_desk=ApprovalDesk()
        )
        statuses.append(security.execute_tool("read_document", args).status)

    assert statuses == [CallStatus.DENIED, CallStatus.EXECUTED, CallStatus.APPROVAL_REQUIRED]
    assert len(spy.calls) == 1


def test_agents_sharing_a_registry_keep_separate_permissions():
    registry = ToolRegistry()
    registry.register("web_search", schema=ReadDocumentArgs, function=Spy())
    registry.register("run_tests", schema=NoArgs, function=Spy())
    research = SAW(
        config=AgentConfig(agent_id="research-agent", allowed_tools=["web_search"]),
        registry=registry, logger=RecordingLogger(),
    )
    coding = SAW(
        config=AgentConfig(agent_id="coding-agent", allowed_tools=["run_tests"]),
        registry=registry, logger=RecordingLogger(),
    )
    assert research.execute_tool("web_search", {"path": "q"}).ok
    assert research.execute_tool("run_tests").status is CallStatus.DENIED
    assert coding.execute_tool("run_tests").ok
    assert coding.execute_tool("web_search", {"path": "q"}).status is CallStatus.DENIED


def test_default_registries_are_separate():
    a = make_saw(agent_id="a", allowed_tools=["t"])
    b = make_saw(agent_id="b", allowed_tools=["t"])
    a.register_tool("t", schema=NoArgs, function=Spy())
    assert "t" in a.registered_tools
    assert b.execute_tool("t").status is CallStatus.UNKNOWN_TOOL


def test_events_carry_the_right_agent_id():
    logger = RecordingLogger()
    registry = ToolRegistry()
    registry.register("t", schema=NoArgs, function=Spy())
    SAW(config=AgentConfig(agent_id="one", allowed_tools=["t"]), registry=registry, logger=logger).execute_tool("t")
    SAW(config=AgentConfig(agent_id="two"), registry=registry, logger=logger).execute_tool("t")
    by_agent = {e.agent_id for e in logger.events if e.event_type is EventType.TOOL_EXECUTED}
    denied = {e.agent_id for e in logger.events if e.event_type is EventType.TOOL_CALL_DENIED}
    assert by_agent == {"one"} and denied == {"two"}


# ------------------------------------------------------- custom policy invariant


class AllowEverything(PermissionPolicy):
    def evaluate(self, call, config):
        return PolicyDecision.allow("anything goes")


class DenyLargeReads(PermissionPolicy):
    def evaluate(self, call, config):
        if call.arguments.get("path", "").startswith("/secret"):
            return PolicyDecision.deny("secret paths are off limits")
        return PolicyDecision.allow("ok")


def test_custom_policy_cannot_grant_unlisted_tool():
    security = make_saw(policy=AllowEverything(), allowed_tools=[])
    spy = Spy()
    security.register_tool("read_document", schema=ReadDocumentArgs, function=spy)
    assert security.execute_tool("read_document", {"path": "a"}).status is CallStatus.DENIED
    assert spy.calls == []


def test_custom_policy_cannot_skip_approval():
    security = make_saw(policy=AllowEverything(), allowed_tools=["send_email"], approval_required=["send_email"])
    spy = Spy()
    security.register_tool("send_email", schema=SendEmailArgs, function=spy)
    outcome = security.execute_tool("send_email", {"to": "x", "body": "y"})
    assert outcome.status is CallStatus.APPROVAL_REQUIRED
    assert spy.calls == []


def test_custom_policy_sees_validated_arguments_and_can_tighten():
    logger = RecordingLogger()
    security = make_saw(logger, policy=DenyLargeReads(), allowed_tools=["read_document"])
    spy = Spy()
    security.register_tool("read_document", schema=ReadDocumentArgs, function=spy)

    assert security.execute_tool("read_document", {"path": "/public/a"}).ok
    blocked = security.execute_tool("read_document", {"path": "/secret/key"})

    assert blocked.status is CallStatus.DENIED
    assert blocked.reason == "secret paths are off limits"
    assert spy.calls == [{"path": "/public/a"}]
    denied = call_events(logger, blocked.call_id)[-1]
    assert denied.details["config_decision"] == "allow"
    assert denied.details["policy_decision"] == "deny"


def test_crashing_policy_denies_and_keeps_error_for_debugging():
    class Crash(PermissionPolicy):
        def evaluate(self, call, config):
            raise RuntimeError("boom")

    security = make_saw(policy=Crash(), allowed_tools=["read_document"])
    spy = Spy()
    security.register_tool("read_document", schema=ReadDocumentArgs, function=spy)
    outcome = security.execute_tool("read_document", {"path": "a"})
    assert outcome.status is CallStatus.DENIED
    assert isinstance(outcome.error, RuntimeError)
    assert spy.calls == []


# --------------------------------------------------------------- tool failures


def test_tool_exception_becomes_failed_outcome():
    logger = RecordingLogger()
    security = make_saw(logger, allowed_tools=["read_document"])
    security.register_tool("read_document", schema=ReadDocumentArgs, function=Spy(raises=FileNotFoundError("/home/me/secret.txt")))

    outcome = security.execute_tool("read_document", {"path": "a"})

    assert outcome.status is CallStatus.FAILED
    assert "FileNotFoundError" in outcome.reason
    assert "secret.txt" not in outcome.reason
    assert isinstance(outcome.error, FileNotFoundError)
    events = call_events(logger, outcome.call_id)
    assert types_of(events) == [
        EventType.TOOL_CALL_RECEIVED,
        EventType.TOOL_CALL_ALLOWED,
        EventType.TOOL_FAILED,
    ]
    assert events[-1].details["exception_type"] == "FileNotFoundError"


# ------------------------------------------------------------ audit failures


class FailOn(SecurityLogger):
    """Logger that fails on one event type and records the rest."""

    def __init__(self, event_type):
        self.event_type = event_type
        self.events = []

    def log(self, event):
        if event.event_type is self.event_type:
            raise OSError("log disk full")
        self.events.append(event)


@pytest.mark.parametrize("failing", [EventType.TOOL_CALL_RECEIVED, EventType.TOOL_CALL_ALLOWED])
def test_logging_failure_before_execution_prevents_execution(failing):
    security = make_saw(FailOn(failing), allowed_tools=["read_document"])
    spy = Spy()
    security.register_tool("read_document", schema=ReadDocumentArgs, function=spy)

    with pytest.raises(AuditError) as info:
        security.execute_tool("read_document", {"path": "a"})

    assert info.value.tool_executed is False
    assert "was not run" in str(info.value)
    assert isinstance(info.value.__cause__, OSError)
    assert spy.calls == []


def test_logging_failure_on_denial_still_blocks():
    security = make_saw(FailOn(EventType.TOOL_CALL_DENIED), allowed_tools=[])
    spy = Spy()
    security.register_tool("read_document", schema=ReadDocumentArgs, function=spy)

    with pytest.raises(AuditError) as info:
        security.execute_tool("read_document", {"path": "a"})

    assert info.value.tool_executed is False
    assert info.value.outcome.status is CallStatus.DENIED
    assert spy.calls == []


def test_logging_failure_on_approval_still_blocks():
    security = make_saw(FailOn(EventType.APPROVAL_REQUIRED), allowed_tools=["t"], approval_required=["t"])
    spy = Spy()
    security.register_tool("t", schema=NoArgs, function=spy)
    with pytest.raises(AuditError) as info:
        security.execute_tool("t")
    assert info.value.outcome.status is CallStatus.APPROVAL_REQUIRED
    assert spy.calls == []


def test_logging_failure_after_execution_withholds_result():
    security = make_saw(FailOn(EventType.TOOL_EXECUTED), allowed_tools=["read_document"])
    spy = Spy(result="sensitive result")
    security.register_tool("read_document", schema=ReadDocumentArgs, function=spy)

    with pytest.raises(AuditError) as info:
        security.execute_tool("read_document", {"path": "a"})

    assert info.value.tool_executed is True
    assert info.value.outcome is None
    assert "result has been withheld" in str(info.value)
    assert len(spy.calls) == 1  # documented: the tool did run


def test_logging_failure_after_tool_failure():
    security = make_saw(FailOn(EventType.TOOL_FAILED), allowed_tools=["t"])
    security.register_tool("t", schema=NoArgs, function=Spy(raises=ValueError()))
    with pytest.raises(AuditError) as info:
        security.execute_tool("t")
    assert info.value.tool_executed is True
    assert info.value.outcome.status is CallStatus.FAILED


# ------------------------------------------------------------ event contents


def test_argument_values_are_never_logged():
    logger = RecordingLogger()
    security = make_saw(logger, allowed_tools=["send_email"])
    security.register_tool("send_email", schema=SendEmailArgs, function=Spy())
    secret = "my password is hunter2"

    security.execute_tool("send_email", {"to": "a@b.c", "body": secret})
    security.execute_tool("send_email", {"to": "a@b.c", "body": 12345678})

    for event in logger.events:
        assert secret not in repr(event.to_dict())
        assert "12345678" not in repr(event.to_dict())
    received = [e for e in logger.events if e.event_type is EventType.TOOL_CALL_RECEIVED]
    assert received[0].details["argument_names"] == ["body", "to"]


def test_very_long_tool_name_is_shortened_in_events_and_outcome():
    logger = RecordingLogger()
    security = make_saw(logger)
    outcome = security.execute_tool("x" * 100_000)
    assert len(outcome.tool_name) < 200
    assert all(len(e.tool_name or "") < 200 for e in logger.events)


def test_each_call_gets_its_own_call_id():
    logger = RecordingLogger()
    security = make_saw(logger, allowed_tools=["t"])
    security.register_tool("t", schema=NoArgs, function=Spy())
    ids = {security.execute_tool("t").call_id for _ in range(5)}
    assert len(ids) == 5


def test_registration_does_not_grant_permission():
    security = make_saw()  # no allowed tools
    security.register_tool("t", schema=NoArgs, function=Spy())
    assert "t" in security.registered_tools
    assert security.execute_tool("t").status is CallStatus.DENIED

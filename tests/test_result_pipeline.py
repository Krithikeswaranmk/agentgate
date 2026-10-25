"""Integration tests: results through SAW.execute_tool, and the failure table."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from saw import (
    SAW,
    AgentConfig,
    AuditError,
    CallStatus,
    EventType,
    ResultValidationError,
    ResultValidator,
    SecurityLogger,
    ValidationResult,
)
from conftest import RecordingLogger


@dataclass
class RunTestsArgs:
    path: str = "tests"


@dataclass
class RunTestsResult:
    passed: bool
    failures: int

    def __post_init__(self):
        if self.failures < 0:
            raise ValueError("failures must be >= 0")


def coding_agent(logger=None, validator=None, **config):
    config.setdefault("agent_id", "coding-agent")
    config.setdefault("allowed_tools", ["run_tests"])
    return SAW(
        config=AgentConfig(**config),
        logger=logger or RecordingLogger(),
        tool_result_validator=validator,
    )


def register_run_tests(security, returns, *, result_schema=RunTestsResult):
    calls = []

    def run_tests(path):
        calls.append(path)
        if isinstance(returns, BaseException):
            raise returns
        return returns

    security.register_tool(
        "run_tests", schema=RunTestsArgs, function=run_tests, result_schema=result_schema
    )
    return calls


def events_for(logger, outcome):
    return [e for e in logger.events if e.call_id == outcome.call_id]


SECRET = "SECRET-RESULT-CONTENT-123"


# --------------------------------------------------------------- spec example


def test_valid_result_reaches_agent():
    logger = RecordingLogger()
    security = coding_agent(logger)
    register_run_tests(security, {"passed": True, "failures": 0})

    outcome = security.execute_tool("run_tests", {})

    assert outcome.ok and outcome.value == {"passed": True, "failures": 0}
    executed = events_for(logger, outcome)[-1]
    assert executed.event_type is EventType.TOOL_EXECUTED
    assert executed.details["result_chars"] == len("passed") + len("failures")
    assert executed.details["duration_ms"] >= 0


def test_spec_invalid_result_never_reaches_agent():
    logger = RecordingLogger()
    security = coding_agent(logger)
    calls = register_run_tests(security, {"passed": "yes", "failures": -1})

    outcome = security.execute_tool("run_tests", {})

    assert calls == ["tests"]  # the tool did run
    assert outcome.status is CallStatus.RESULT_REJECTED
    assert not outcome.ok
    with pytest.raises(ResultValidationError) as info:
        outcome.value
    assert info.value.outcome is outcome
    events = events_for(logger, outcome)
    assert [e.event_type for e in events] == [
        EventType.TOOL_CALL_RECEIVED,
        EventType.TOOL_CALL_ALLOWED,
        EventType.RESULT_REJECTED,
    ]
    assert events[-1].details["check"] == "result_schema"
    assert events[-1].details["result_type"] == "dict"


def test_returning_schema_instance_is_converted():
    security = coding_agent()
    register_run_tests(security, RunTestsResult(passed=True, failures=0))
    outcome = security.execute_tool("run_tests", {})
    assert outcome.ok and outcome.value == {"passed": True, "failures": 0}


def test_badly_typed_instance_rejected():
    security = coding_agent()
    register_run_tests(security, RunTestsResult(passed="yes", failures=0))  # type: ignore[arg-type]
    assert security.execute_tool("run_tests", {}).status is CallStatus.RESULT_REJECTED


# ------------------------------------------- rejected content goes nowhere


@pytest.mark.parametrize(
    "returns, config",
    [
        ({"passed": SECRET, "failures": 0}, {}),                     # schema
        ({"passed": True, "failures": 0, SECRET: 1}, {}),            # unknown field
        (SECRET * 10_000, {}),                                       # size (and schema)
        ({"blob": SECRET.encode()}, {}),                             # validator
    ],
    # Explicit short IDs: pytest puts the test ID in an environment variable,
    # and Windows limits those to 32,767 characters.
    ids=["schema", "unknown-field", "size", "shape"],
)
def test_rejected_result_content_is_never_exposed(returns, config):
    logger = RecordingLogger()
    security = coding_agent(logger, **config)
    register_run_tests(security, returns)

    outcome = security.execute_tool("run_tests", {})

    assert outcome.status is CallStatus.RESULT_REJECTED
    with pytest.raises(ResultValidationError) as info:
        outcome.value
    exposed = [
        repr(outcome), outcome.reason, repr(outcome.error), str(info.value),
        *(repr(e.to_dict()) for e in logger.events),
    ]
    assert not any(SECRET in text for text in exposed)


# ----------------------------------------------------------- size handling


def test_oversized_result_rejected_without_schema():
    logger = RecordingLogger()
    security = coding_agent(logger, max_result_chars=50)
    register_run_tests(security, "log line\n" * 100, result_schema=None)
    outcome = security.execute_tool("run_tests", {})
    assert outcome.status is CallStatus.RESULT_REJECTED
    assert events_for(logger, outcome)[-1].details["check"] == "result_size"


def test_truncate_mode_constrains_text_and_records_it():
    logger = RecordingLogger()
    security = coding_agent(logger, max_result_chars=200, oversized_results="truncate")
    register_run_tests(security, "line\n" * 1000, result_schema=None)

    outcome = security.execute_tool("run_tests", {})

    assert outcome.ok and outcome.truncated
    assert len(outcome.value) <= 200
    assert "[SAW: result truncated to" in outcome.value
    executed = events_for(logger, outcome)[-1]
    assert executed.details["truncated"] is True
    assert executed.details["original_chars"] == 5000


def test_untruncated_result_reports_not_truncated():
    security = coding_agent(oversized_results="truncate")
    register_run_tests(security, "ok", result_schema=None)
    outcome = security.execute_tool("run_tests", {})
    assert outcome.ok and not outcome.truncated


def test_size_limits_are_per_agent():
    small = coding_agent(agent_id="small", max_result_chars=10)
    large = coding_agent(agent_id="large", max_result_chars=10_000)
    for security in (small, large):
        register_run_tests(security, "x" * 500, result_schema=None)
    assert small.execute_tool("run_tests", {}).status is CallStatus.RESULT_REJECTED
    assert large.execute_tool("run_tests", {}).ok


# ------------------------------------------------- exceptions and interrupts


def test_tool_exception_is_failure_with_duration():
    logger = RecordingLogger()
    security = coding_agent(logger)
    register_run_tests(security, RuntimeError("pytest crashed"))
    outcome = security.execute_tool("run_tests", {})
    assert outcome.status is CallStatus.FAILED
    failed = events_for(logger, outcome)[-1]
    assert failed.event_type is EventType.TOOL_FAILED
    assert failed.details["duration_ms"] >= 0


def test_returned_exception_is_not_a_success():
    security = coding_agent()
    register_run_tests(security, None, result_schema=None)
    security2 = coding_agent()
    security2.register_tool(
        "run_tests", schema=RunTestsArgs, function=lambda path: ValueError("oops")
    )
    outcome = security2.execute_tool("run_tests", {})
    assert outcome.status is CallStatus.RESULT_REJECTED


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_interrupt_is_recorded_then_reraised(exc_type):
    logger = RecordingLogger()
    security = coding_agent(logger)
    register_run_tests(security, exc_type())

    with pytest.raises(exc_type):
        security.execute_tool("run_tests", {})

    last = logger.events[-1]
    assert last.event_type is EventType.TOOL_FAILED
    assert last.details["status"] == "interrupted"
    assert last.details["exception_type"] == exc_type.__name__


def test_interrupt_wins_over_logging_failure():
    class FailOnFailed(SecurityLogger):
        def log(self, event):
            if event.event_type is EventType.TOOL_FAILED:
                raise OSError("disk full")

    security = coding_agent(FailOnFailed())
    register_run_tests(security, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt) as info:
        security.execute_tool("run_tests", {})
    assert isinstance(info.value.__context__, AuditError)


# ------------------------------------------------------------ audit failures


class FailOn(SecurityLogger):
    def __init__(self, event_type):
        self.event_type = event_type

    def log(self, event):
        if event.event_type is self.event_type:
            raise OSError("disk full")


def test_logging_failure_on_rejection_raises_audit_error():
    security = coding_agent(FailOn(EventType.RESULT_REJECTED))
    register_run_tests(security, {"passed": "yes", "failures": 0})
    with pytest.raises(AuditError) as info:
        security.execute_tool("run_tests", {})
    assert info.value.tool_executed is True
    assert info.value.outcome.status is CallStatus.RESULT_REJECTED


def test_logging_failure_on_accepted_result_withholds_it():
    security = coding_agent(FailOn(EventType.TOOL_EXECUTED))
    register_run_tests(security, {"passed": True, "failures": 0})
    with pytest.raises(AuditError) as info:
        security.execute_tool("run_tests", {})
    assert info.value.tool_executed is True and info.value.outcome is None


# ---------------------------------------------------------- failure table


class RaisingValidator(ResultValidator):
    def validate(self, call, result, config):
        raise RuntimeError("bug")


class WrongTypeValidator(ResultValidator):
    def validate(self, call, result, config):
        return "ok"


class AcceptAll(ResultValidator):
    def validate(self, call, result, config):
        return ValidationResult.accept()


@pytest.mark.parametrize(
    "returns, validator, schema, expected_status, check",
    [
        ({"passed": True, "failures": 0}, None, RunTestsResult, CallStatus.EXECUTED, None),
        ({"passed": "yes", "failures": 0}, None, RunTestsResult, CallStatus.RESULT_REJECTED, "result_schema"),
        ({"passed": True, "failures": 0}, RaisingValidator(), RunTestsResult, CallStatus.RESULT_REJECTED, "result_validator"),
        ({"passed": True, "failures": 0}, WrongTypeValidator(), RunTestsResult, CallStatus.RESULT_REJECTED, "result_validator"),
        (object(), None, None, CallStatus.RESULT_REJECTED, "result_shape"),
        (object(), AcceptAll(), RunTestsResult, CallStatus.RESULT_REJECTED, "result_shape"),
        ({"passed": True, "failures": 0, "b": b"x"}, AcceptAll(), None, CallStatus.RESULT_REJECTED, "result_shape"),
        ("x" * 100_001, AcceptAll(), None, CallStatus.RESULT_REJECTED, "result_size"),
    ],
    ids=[
        "accepted", "schema-rejects", "validator-raises", "validator-wrong-type",
        "object-no-schema", "object-with-schema", "bytes-inside", "oversized",
    ],
)
def test_failure_table(returns, validator, schema, expected_status, check):
    logger = RecordingLogger()
    security = coding_agent(logger, validator=validator)
    register_run_tests(security, returns, result_schema=schema)

    outcome = security.execute_tool("run_tests", {})

    assert outcome.status is expected_status
    last = events_for(logger, outcome)[-1]
    if check is None:
        assert last.event_type is EventType.TOOL_EXECUTED
    else:
        assert last.event_type is EventType.RESULT_REJECTED
        assert last.details["check"] == check
        with pytest.raises(ResultValidationError):
            outcome.value


def test_result_schema_registration_errors():
    from saw import ConfigurationError

    security = coding_agent()
    with pytest.raises(ConfigurationError):
        security.register_tool(
            "run_tests", schema=RunTestsArgs, function=lambda path: 1, result_schema=dict
        )
    assert "run_tests" not in security.registered_tools

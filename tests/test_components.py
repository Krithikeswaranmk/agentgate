"""Incorrect custom components fail clearly instead of silently weakening security."""

from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass

import pytest

from saw import (
    SAW,
    AgentConfig,
    AuditError,
    CallStatus,
    ConfigurationError,
    DefaultPermissionPolicy,
    DefaultResultValidator,
    DefaultSecurityLogger,
    PermissionPolicy,
    PolicyDecision,
    ResultValidator,
    SecurityLogger,
    ValidationResult,
)
from conftest import RecordingLogger


@dataclass
class NoArgs:
    pass


# ------------------------------------------------- setup-time: wrong methods


class PolicyMissingConfig(PermissionPolicy):
    def evaluate(self, call):  # type: ignore[override]
        return PolicyDecision.allow("x")


class PolicyExtraRequired(PermissionPolicy):
    def evaluate(self, call, config, user):  # type: ignore[override]
        return PolicyDecision.allow("x")


class AsyncPolicy(PermissionPolicy):
    async def evaluate(self, call, config):  # type: ignore[override]
        return PolicyDecision.allow("x")


class GeneratorPolicy(PermissionPolicy):
    def evaluate(self, call, config):  # type: ignore[override]
        yield PolicyDecision.allow("x")


class ValidatorMissingResult(ResultValidator):
    def validate(self, call, config):  # type: ignore[override]
        return ValidationResult.accept()


class AsyncValidator(ResultValidator):
    async def validate(self, call, result, config):  # type: ignore[override]
        return ValidationResult.accept()


class LoggerNoEvent(SecurityLogger):
    def log(self):  # type: ignore[override]
        pass


class AsyncLogger(SecurityLogger):
    async def log(self, event):  # type: ignore[override]
        pass


class AsyncGenLogger(SecurityLogger):
    async def log(self, event):  # type: ignore[override]
        yield


class NonCallableMethod(PermissionPolicy):
    def __init__(self):
        self.evaluate = "not a method"  # type: ignore[assignment]

    def evaluate(self, call, config):  # pragma: no cover - shadowed
        return PolicyDecision.allow("x")


@pytest.mark.parametrize(
    "kwarg, component, message",
    [
        ("permission_policy", PolicyMissingConfig(), r"PolicyMissingConfig\.evaluate\(\) must accept 2 positional argument\(s\) \(call, config\)"),
        ("permission_policy", PolicyExtraRequired(), r"must accept 2 positional"),
        ("permission_policy", AsyncPolicy(), r"AsyncPolicy\.evaluate\(\) is async"),
        ("permission_policy", GeneratorPolicy(), r"generator function"),
        ("permission_policy", NonCallableMethod(), r"is not callable"),
        ("tool_result_validator", ValidatorMissingResult(), r"must accept 3 positional argument\(s\) \(call, result, config\)"),
        ("tool_result_validator", AsyncValidator(), r"is async"),
        ("logger", LoggerNoEvent(), r"must accept 1 positional argument\(s\) \(event\)"),
        ("logger", AsyncLogger(), r"AsyncLogger\.log\(\) is async"),
        ("logger", AsyncGenLogger(), r"is async"),
    ],
)
def test_wrong_method_rejected_when_saw_is_created(kwarg, component, message):
    with pytest.raises(ConfigurationError, match=message) as info:
        SAW(agent_id="a", **{kwarg: component})
    assert kwarg in str(info.value)


class FlexiblePolicy(PermissionPolicy):
    def evaluate(self, call, config=None, *extra, **options):  # type: ignore[override]
        return PolicyDecision.allow("ok")


class StaticPolicy(PermissionPolicy):
    @staticmethod
    def evaluate(call, config):  # type: ignore[override]
        return PolicyDecision.allow("ok")


@pytest.mark.parametrize("policy", [FlexiblePolicy(), StaticPolicy()])
def test_compatible_signatures_accepted(policy):
    SAW(agent_id="a", permission_policy=policy, logger=RecordingLogger())


def test_built_in_components_pass_their_own_checks():
    security = SAW(
        agent_id="a",
        permission_policy=DefaultPermissionPolicy(),
        tool_result_validator=DefaultResultValidator(),
        logger=DefaultSecurityLogger(),
    )
    assert isinstance(security.logger, DefaultSecurityLogger)


def test_stdlib_logger_rejected_with_hint():
    with pytest.raises(ConfigurationError, match=r"DefaultSecurityLogger\(your_logger\)"):
        SAW(agent_id="a", logger=logging.getLogger("app"))  # type: ignore[arg-type]


@pytest.mark.parametrize("kwarg", ["permission_policy", "tool_result_validator", "logger"])
def test_old_argument_names_are_rejected(kwarg):
    old = {"permission_policy": "policy", "tool_result_validator": "result_validator", "logger": "security_logger"}[kwarg]
    with pytest.raises(TypeError, match=old):
        SAW(agent_id="a", **{old: None})


def test_explicit_none_uses_defaults():
    security = SAW(agent_id="a", permission_policy=None, tool_result_validator=None, logger=None)
    assert isinstance(security.permission_policy, DefaultPermissionPolicy)
    assert isinstance(security.tool_result_validator, DefaultResultValidator)
    assert isinstance(security.logger, DefaultSecurityLogger)


def test_invalid_registry_rejected():
    with pytest.raises(ConfigurationError, match="registry"):
        SAW(agent_id="a", registry={})  # type: ignore[arg-type]


# --------------------------------------- call-time: loggers cannot lose events


class ReturnsFalseLater(SecurityLogger):
    """Fine during setup, then signals failure by returning False."""

    def __init__(self):
        self.calls = 0

    def log(self, event):
        self.calls += 1
        return None if self.calls == 1 else False


async def _write(event):
    pass


class ReturnsCoroutine(SecurityLogger):
    """Not declared async, but returns a coroutine: the event is never written."""

    def __init__(self):
        self.returned = []

    def log(self, event):
        coroutine = _write(event)
        self.returned.append(coroutine)
        return coroutine


def assert_all_closed(coroutines):
    assert coroutines, "expected the component to have returned a coroutine"
    for coroutine in coroutines:
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


def test_logger_returning_false_stops_the_call():
    security = SAW(config=AgentConfig(agent_id="a", allowed_tools=["t"]), logger=ReturnsFalseLater())
    ran = []
    security.register_tool("t", schema=NoArgs, function=lambda: ran.append(1))
    with pytest.raises(AuditError, match=r"log\(\) returned bool") as info:
        security.execute_tool("t")
    assert info.value.tool_executed is False
    assert ran == []


def test_logger_returning_value_at_setup_fails_setup():
    class ReturnsTrue(SecurityLogger):
        def log(self, event):
            return True

    with pytest.raises(ConfigurationError, match="must return None"):
        SAW(agent_id="a", logger=ReturnsTrue())


def test_logger_returning_coroutine_fails_without_leaking_a_coroutine():
    logger = ReturnsCoroutine()
    with pytest.raises(ConfigurationError, match="returned coroutine"):
        SAW(agent_id="a", logger=logger)
    assert_all_closed(logger.returned)


# ------------------------- call-time: policies and validators that misbehave


async def _decision():
    return PolicyDecision.allow("x")


async def _verdict():
    return ValidationResult.accept()


class PolicyReturnsCoroutine(PermissionPolicy):
    def __init__(self):
        self.returned = []

    def evaluate(self, call, config):
        self.returned.append(_decision())
        return self.returned[-1]


class ValidatorReturnsCoroutine(ResultValidator):
    def __init__(self):
        self.returned = []

    def validate(self, call, result, config):
        self.returned.append(_verdict())
        return self.returned[-1]


def test_policy_returning_coroutine_denies_cleanly():
    policy = PolicyReturnsCoroutine()
    security = SAW(
        config=AgentConfig(agent_id="a", allowed_tools=["t"]),
        permission_policy=policy, logger=RecordingLogger(),
    )
    ran = []
    security.register_tool("t", schema=NoArgs, function=lambda: ran.append(1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outcome = security.execute_tool("t")
    assert outcome.status is CallStatus.DENIED
    assert "returned coroutine, not a PolicyDecision" in outcome.reason
    assert ran == []
    assert_all_closed(policy.returned)


def test_validator_returning_coroutine_rejects_cleanly():
    validator = ValidatorReturnsCoroutine()
    security = SAW(
        config=AgentConfig(agent_id="a", allowed_tools=["t"]),
        tool_result_validator=validator, logger=RecordingLogger(),
    )
    security.register_tool("t", schema=NoArgs, function=lambda: "ok")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outcome = security.execute_tool("t")
    assert outcome.status is CallStatus.RESULT_REJECTED
    assert "returned coroutine, not a ValidationResult" in outcome.reason
    assert_all_closed(validator.returned)

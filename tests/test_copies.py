"""Policies, validators and result schemas only ever see copies."""

from __future__ import annotations

from dataclasses import dataclass, field

from saw import (
    SAW,
    AgentConfig,
    CallStatus,
    PermissionPolicy,
    PolicyDecision,
    ResultValidator,
    ValidationResult,
)
from conftest import RecordingLogger


@dataclass
class TagArgs:
    tags: list[str]


@dataclass
class Inner:
    value: str

    def __deepcopy__(self, memo):
        raise RuntimeError("cannot copy")


@dataclass
class UncopyableArgs:
    inner: Inner


@dataclass
class ListResult:
    items: list[str]

    def __post_init__(self):
        self.items.append("added by schema")  # misbehaving schema


class MutatingPolicy(PermissionPolicy):
    def evaluate(self, call, config):
        call.arguments["tags"].append("injected-by-policy")
        return PolicyDecision.allow("ok")


class MutatingValidator(ResultValidator):
    def validate(self, call, result, config):
        result["items"].append("injected-by-validator")
        result["extra"] = "added"
        return ValidationResult.accept()


def make(**kwargs):
    config = AgentConfig(agent_id="a", allowed_tools=["t"])
    return SAW(config=config, logger=RecordingLogger(), **kwargs)


def test_policy_changes_do_not_reach_the_tool():
    security = make(permission_policy=MutatingPolicy())
    received = []
    security.register_tool("t", schema=TagArgs, function=lambda tags: received.append(list(tags)))
    assert security.execute_tool("t", {"tags": ["a"]}).ok
    assert received == [["a"]]


def test_validator_changes_do_not_reach_the_agent():
    security = make(tool_result_validator=MutatingValidator())
    security.register_tool("t", schema=TagArgs, function=lambda tags: {"items": ["x"]})
    outcome = security.execute_tool("t", {"tags": []})
    assert outcome.ok and outcome.value == {"items": ["x"]}


def test_result_schema_changes_do_not_reach_the_agent():
    security = make()
    security.register_tool(
        "t", schema=TagArgs, function=lambda tags: {"items": ["x"]}, result_schema=ListResult
    )
    outcome = security.execute_tool("t", {"tags": []})
    assert outcome.ok and outcome.value == {"items": ["x"]}


def test_validator_cannot_mutate_then_bypass_schema():
    # The validator changes its copy into something the schema would reject;
    # the schema checks its own copy of the original, so the call is fine and
    # the agent receives the original.
    @dataclass
    class Items:
        items: list[str]

    security = make(tool_result_validator=MutatingValidator())
    security.register_tool("t", schema=TagArgs, function=lambda tags: {"items": ["x"]}, result_schema=Items)
    outcome = security.execute_tool("t", {"tags": []})
    assert outcome.ok and outcome.value == {"items": ["x"]}


def test_uncopyable_arguments_fail_closed():
    security = make()
    ran = []
    security.register_tool("t", schema=UncopyableArgs, function=lambda inner: ran.append(1))
    outcome = security.execute_tool("t", {"inner": {"value": "x"}})
    assert outcome.status is CallStatus.DENIED
    assert "could not be copied" in outcome.reason
    assert ran == []

"""The mandatory invariant: a custom policy can only make decisions stricter."""

import pytest

from saw import AgentConfig, Decision, PermissionPolicy, PolicyDecision, ToolCall
from saw.policies.enforcement import config_decision, enforce

CFG = AgentConfig(
    agent_id="pa",
    allowed_tools=["read_calendar", "send_email"],
    approval_required=["send_email"],
)


class Fixed(PermissionPolicy):
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def evaluate(self, call, config):
        self.calls += 1
        return self.decision


ALLOW = PolicyDecision.allow("policy allows")
APPROVE = PolicyDecision.require_approval("policy wants approval")
DENY = PolicyDecision.deny("policy denies")


@pytest.mark.parametrize(
    "tool, policy_decision, expected",
    [
        # Tool allowed by config: policy can keep or tighten.
        ("read_calendar", ALLOW, Decision.ALLOW),
        ("read_calendar", APPROVE, Decision.REQUIRE_APPROVAL),
        ("read_calendar", DENY, Decision.DENY),
        # Tool needs approval by config: policy cannot loosen to ALLOW.
        ("send_email", ALLOW, Decision.REQUIRE_APPROVAL),
        ("send_email", APPROVE, Decision.REQUIRE_APPROVAL),
        ("send_email", DENY, Decision.DENY),
        # Tool not allowed by config: denied whatever the policy says.
        ("delete_files", ALLOW, Decision.DENY),
        ("delete_files", APPROVE, Decision.DENY),
        ("delete_files", DENY, Decision.DENY),
    ],
)
def test_final_decision_is_the_stricter(tool, policy_decision, expected):
    result = enforce(Fixed(policy_decision), ToolCall(tool), CFG)
    assert result.final.decision is expected


def test_policy_not_consulted_for_tools_config_denies():
    policy = Fixed(ALLOW)
    result = enforce(policy, ToolCall("delete_files"), CFG)
    assert policy.calls == 0
    assert result.policy is None
    assert result.details() == {"config_decision": "deny"}


def test_policy_reason_used_when_policy_is_stricter():
    result = enforce(Fixed(DENY), ToolCall("read_calendar"), CFG)
    assert result.final.reason == "policy denies"
    assert result.details() == {"config_decision": "allow", "policy_decision": "deny"}


def test_config_reason_used_when_config_is_stricter():
    result = enforce(Fixed(ALLOW), ToolCall("send_email"), CFG)
    assert "requires approval" in result.final.reason
    assert result.details() == {
        "config_decision": "require_approval",
        "policy_decision": "allow",
    }


def test_policy_exception_denies():
    class Broken(PermissionPolicy):
        def evaluate(self, call, config):
            raise RuntimeError("database down: password=hunter2")

    result = enforce(Broken(), ToolCall("read_calendar"), CFG)
    assert result.final.decision is Decision.DENY
    assert "RuntimeError" in result.final.reason
    assert "hunter2" not in result.final.reason  # message kept out of reason
    assert isinstance(result.policy_error, RuntimeError)
    assert result.details()["policy_problem"] == "permission policy raised RuntimeError"


@pytest.mark.parametrize("bad", [None, True, "allow", Decision.ALLOW, {"decision": "allow"}])
def test_policy_returning_wrong_type_denies(bad):
    result = enforce(Fixed(bad), ToolCall("read_calendar"), CFG)
    assert result.final.decision is Decision.DENY
    assert "not a PolicyDecision" in result.final.reason


def test_config_decision_rules():
    assert config_decision(ToolCall("read_calendar"), CFG).decision is Decision.ALLOW
    assert config_decision(ToolCall("send_email"), CFG).decision is Decision.REQUIRE_APPROVAL
    assert config_decision(ToolCall("other"), CFG).decision is Decision.DENY

import pytest

from saw import AgentConfig, Decision, DefaultPermissionPolicy, PolicyDecision, ToolCall


@pytest.fixture
def policy():
    return DefaultPermissionPolicy()


@pytest.fixture
def assistant_config():
    return AgentConfig(
        agent_id="personal-assistant",
        allowed_tools=["read_calendar", "send_email"],
        approval_required=["send_email"],
    )


def test_allowed_tool_is_allowed(policy, assistant_config):
    decision = policy.evaluate(ToolCall("read_calendar"), assistant_config)
    assert decision.decision is Decision.ALLOW
    assert decision.is_allowed


def test_unlisted_tool_is_denied(policy, assistant_config):
    decision = policy.evaluate(ToolCall("delete_files"), assistant_config)
    assert decision.decision is Decision.DENY
    assert not decision.is_allowed
    assert "delete_files" in decision.reason


def test_approval_tool_requires_approval(policy, assistant_config):
    decision = policy.evaluate(ToolCall("send_email"), assistant_config)
    assert decision.decision is Decision.REQUIRE_APPROVAL
    assert not decision.is_allowed


def test_agent_with_no_allowed_tools_is_denied_everything(policy):
    cfg = AgentConfig(agent_id="empty")
    assert policy.evaluate(ToolCall("anything"), cfg).decision is Decision.DENY


def test_tool_name_matching_is_exact(policy):
    cfg = AgentConfig(agent_id="a", allowed_tools=["read_file"])
    for name in ["Read_File", "read_file ", "read_files", "read"]:
        assert policy.evaluate(ToolCall(name), cfg).decision is Decision.DENY


def test_same_policy_gives_per_agent_answers(policy):
    research = AgentConfig(agent_id="research", allowed_tools=["web_search"])
    coding = AgentConfig(agent_id="coding", allowed_tools=["run_tests"])
    call = ToolCall("web_search")
    assert policy.evaluate(call, research).decision is Decision.ALLOW
    assert policy.evaluate(call, coding).decision is Decision.DENY


def test_policy_decision_constructors():
    assert PolicyDecision.allow("ok").decision is Decision.ALLOW
    assert PolicyDecision.deny("no").decision is Decision.DENY
    assert PolicyDecision.require_approval("ask").decision is Decision.REQUIRE_APPROVAL


@pytest.mark.parametrize("decision", ["allow", True, None])
def test_policy_decision_rejects_non_enum_decision(decision):
    with pytest.raises(TypeError, match="Decision"):
        PolicyDecision(decision, "reason")  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", ["", "   ", None])
def test_policy_decision_requires_reason(reason):
    with pytest.raises(TypeError, match="reason"):
        PolicyDecision(Decision.DENY, reason)  # type: ignore[arg-type]

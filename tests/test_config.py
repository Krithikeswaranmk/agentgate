import dataclasses
import math

import pytest

from saw import AgentConfig, ConfigurationError
from saw.config import DEFAULT_MAX_RESULT_CHARS


def test_minimal_config_defaults():
    cfg = AgentConfig(agent_id="research-agent")
    assert cfg.agent_id == "research-agent"
    assert cfg.allowed_tools == frozenset()
    assert cfg.approval_required == frozenset()
    assert cfg.max_result_chars == DEFAULT_MAX_RESULT_CHARS == 100_000
    assert cfg.oversized_results == "reject"


def test_collections_are_normalised_to_frozensets():
    cfg = AgentConfig(
        agent_id="a",
        allowed_tools=["read_file", "run_tests", "read_file"],
        approval_required=("run_tests",),
    )
    assert cfg.allowed_tools == frozenset({"read_file", "run_tests"})
    assert isinstance(cfg.allowed_tools, frozenset)
    assert isinstance(cfg.approval_required, frozenset)


def test_caller_collection_changes_do_not_leak_into_config():
    tools = {"web_search"}
    cfg = AgentConfig(agent_id="a", allowed_tools=tools)
    tools.add("send_email")
    assert cfg.allowed_tools == frozenset({"web_search"})


def test_result_settings_accepted():
    cfg = AgentConfig(agent_id="a", max_result_chars=500, oversized_results="truncate")
    assert cfg.max_result_chars == 500
    assert cfg.oversized_results == "truncate"


def test_timeout_setting_no_longer_exists():
    with pytest.raises(TypeError):
        AgentConfig(agent_id="a", tool_timeout_seconds=5)  # type: ignore[call-arg]


def test_config_is_immutable():
    cfg = AgentConfig(agent_id="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent_id = "b"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        cfg.allowed_tools.add("x")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "agent_id",
    ["", " ", "-leading-dash", "has space", "slash/name", "x" * 129, "ünicode", "a\n"],
)
def test_invalid_agent_id_rejected(agent_id):
    with pytest.raises(ConfigurationError, match="agent_id"):
        AgentConfig(agent_id=agent_id)


@pytest.mark.parametrize("agent_id", [None, 123, b"bytes"])
def test_non_string_agent_id_rejected(agent_id):
    with pytest.raises(ConfigurationError, match="must be a string"):
        AgentConfig(agent_id=agent_id)


@pytest.mark.parametrize("agent_id", ["a", "A1", "coding-agent", "agent_2.v1", "x" * 128])
def test_valid_agent_ids_accepted(agent_id):
    assert AgentConfig(agent_id=agent_id).agent_id == agent_id


def test_single_string_for_allowed_tools_rejected_with_hint():
    with pytest.raises(ConfigurationError, match=r"did you mean \['web_search'\]"):
        AgentConfig(agent_id="a", allowed_tools="web_search")


@pytest.mark.parametrize("value", [42, None, {"web_search": True}])
def test_non_collection_allowed_tools_rejected(value):
    with pytest.raises(ConfigurationError, match="allowed_tools"):
        AgentConfig(agent_id="a", allowed_tools=value)


def test_invalid_tool_name_rejected():
    with pytest.raises(ConfigurationError, match="allowed_tools entry"):
        AgentConfig(agent_id="a", allowed_tools=["ok", "bad name"])


def test_approval_required_must_be_subset_of_allowed_tools():
    with pytest.raises(ConfigurationError, match=r"not in allowed_tools: \['send_email'\]"):
        AgentConfig(agent_id="a", allowed_tools=["read_calendar"], approval_required=["send_email"])


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "100", None, math.inf])
def test_invalid_max_result_chars_rejected(limit):
    with pytest.raises(ConfigurationError, match="max_result_chars"):
        AgentConfig(agent_id="a", max_result_chars=limit)


@pytest.mark.parametrize("policy", ["truncated", "REJECT", "", None, True])
def test_invalid_oversized_results_rejected(policy):
    with pytest.raises(ConfigurationError, match="oversized_results"):
        AgentConfig(agent_id="a", oversized_results=policy)


def test_configs_with_same_values_are_equal_but_independent():
    a = AgentConfig(agent_id="a", allowed_tools=["t"])
    b = AgentConfig(agent_id="a", allowed_tools=["t"])
    assert a == b
    assert a is not b

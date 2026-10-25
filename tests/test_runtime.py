import pytest

import saw
from saw import (
    SAW,
    AgentConfig,
    ConfigurationError,
    DefaultPermissionPolicy,
    DefaultResultValidator,
    DefaultSecurityLogger,
    EventType,
)


def test_package_imports_and_exposes_public_api():
    for name in saw.__all__:
        assert hasattr(saw, name), name
    assert saw.__version__


def test_init_with_agent_id_uses_defaults():
    security = SAW(agent_id="research-agent")
    assert security.agent_id == "research-agent"
    assert security.config == AgentConfig(agent_id="research-agent")
    assert isinstance(security.permission_policy, DefaultPermissionPolicy)
    assert isinstance(security.tool_result_validator, DefaultResultValidator)
    assert isinstance(security.logger, DefaultSecurityLogger)


def test_agent_id_can_be_positional():
    assert SAW("research-agent").agent_id == "research-agent"


def test_init_with_config():
    cfg = AgentConfig(agent_id="coding-agent", allowed_tools=["read_file"])
    security = SAW(config=cfg)
    assert security.config is cfg
    assert security.agent_id == "coding-agent"


def test_matching_agent_id_and_config_accepted():
    cfg = AgentConfig(agent_id="a")
    assert SAW(agent_id="a", config=cfg).config is cfg


def test_mismatched_agent_id_and_config_rejected():
    with pytest.raises(ConfigurationError, match="does not match"):
        SAW(agent_id="a", config=AgentConfig(agent_id="b"))


def test_missing_agent_id_and_config_rejected():
    with pytest.raises(ConfigurationError, match="agent_id or config"):
        SAW()


def test_invalid_agent_id_rejected():
    with pytest.raises(ConfigurationError):
        SAW(agent_id="bad id")


def test_wrong_config_type_rejected():
    with pytest.raises(ConfigurationError, match="must be an AgentConfig"):
        SAW(config={"agent_id": "a"})  # type: ignore[arg-type]


def test_config_and_components_are_read_only_properties():
    security = SAW(agent_id="a")
    for attr in ("config", "agent_id", "permission_policy", "tool_result_validator", "logger"):
        with pytest.raises(AttributeError):
            setattr(security, attr, None)


def test_two_agents_have_independent_configuration():
    research = SAW(
        config=AgentConfig(agent_id="research-agent", allowed_tools=["web_search"])
    )
    coding = SAW(
        config=AgentConfig(
            agent_id="coding-agent",
            allowed_tools=["read_file", "run_tests"],
            max_result_chars=5_000,
        )
    )
    assert research.config.allowed_tools == frozenset({"web_search"})
    assert coding.config.allowed_tools == frozenset({"read_file", "run_tests"})
    assert research.config.max_result_chars == 100_000
    assert coding.config.max_result_chars == 5_000


def test_creating_second_agent_does_not_change_first():
    research = SAW(config=AgentConfig(agent_id="research-agent", allowed_tools=["web_search"]))
    before = research.config
    SAW(config=AgentConfig(agent_id="coding-agent", allowed_tools=["run_tests"]))
    assert research.config is before
    assert research.config.allowed_tools == frozenset({"web_search"})


def test_default_components_are_not_shared_between_instances():
    a = SAW(agent_id="a")
    b = SAW(agent_id="b")
    assert a.permission_policy is not b.permission_policy
    assert a.tool_result_validator is not b.tool_result_validator
    assert a.logger is not b.logger


def test_same_agent_id_twice_gives_separate_instances():
    # SAW does not keep a global registry of agent IDs (see README).
    a1 = SAW(config=AgentConfig(agent_id="a", allowed_tools=["x"]))
    a2 = SAW(config=AgentConfig(agent_id="a", allowed_tools=["y"]))
    assert a1.config.allowed_tools == frozenset({"x"})
    assert a2.config.allowed_tools == frozenset({"y"})


def test_initialisation_is_logged(recording_logger):
    SAW(
        config=AgentConfig(agent_id="pa", allowed_tools=["send_email"], approval_required=["send_email"]),
        logger=recording_logger,
    )
    assert len(recording_logger.events) == 1
    event = recording_logger.events[0]
    assert event.event_type is EventType.AGENT_INITIALIZED
    assert event.agent_id == "pa"
    assert event.details["allowed_tools"] == ["send_email"]
    assert event.details["approval_required"] == ["send_email"]
    assert event.details["permission_policy"] == "DefaultPermissionPolicy"


def test_logger_failure_during_init_fails_closed():
    class BrokenLogger(saw.SecurityLogger):
        def log(self, event):
            raise OSError("disk full")

    with pytest.raises(ConfigurationError, match="disk full") as info:
        SAW(agent_id="a", logger=BrokenLogger())
    assert isinstance(info.value.__cause__, OSError)


def test_repr():
    assert repr(SAW(agent_id="a")) == "SAW(agent_id='a')"

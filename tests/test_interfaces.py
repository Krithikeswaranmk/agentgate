"""Custom components plug in through the base interfaces; anything else is rejected."""

import pytest

from saw import (
    SAW,
    ConfigurationError,
    PermissionPolicy,
    PolicyDecision,
    ResultValidator,
    SecurityLogger,
    ValidationResult,
)


class DenyAllPolicy(PermissionPolicy):
    def evaluate(self, call, config):
        return PolicyDecision.deny("deny all")


class AcceptAllValidator(ResultValidator):
    def validate(self, call, result, config):
        return ValidationResult.accept()


class NullLogger(SecurityLogger):
    def log(self, event):
        pass


def test_custom_components_are_used():
    policy, validator, logger = DenyAllPolicy(), AcceptAllValidator(), NullLogger()
    security = SAW(
        agent_id="a", permission_policy=policy, tool_result_validator=validator, logger=logger
    )
    assert security.permission_policy is policy
    assert security.tool_result_validator is validator
    assert security.logger is logger


def test_components_can_be_replaced_independently():
    security = SAW(agent_id="a", permission_policy=DenyAllPolicy())
    assert isinstance(security.permission_policy, DenyAllPolicy)
    assert type(security.tool_result_validator).__name__ == "DefaultResultValidator"
    assert type(security.logger).__name__ == "DefaultSecurityLogger"


@pytest.mark.parametrize(
    "kwarg",
    ["permission_policy", "tool_result_validator", "logger"],
)
def test_duck_typed_component_rejected(kwarg):
    class LooksRight:
        def evaluate(self, call, config): ...
        def validate(self, call, result, config): ...
        def log(self, event): ...

    with pytest.raises(ConfigurationError, match=kwarg):
        SAW(agent_id="a", **{kwarg: LooksRight()})


@pytest.mark.parametrize(
    "kwarg, wrong",
    [
        ("permission_policy", AcceptAllValidator()),
        ("tool_result_validator", DenyAllPolicy()),
        ("logger", DenyAllPolicy()),
    ],
)
def test_component_in_wrong_slot_rejected(kwarg, wrong):
    with pytest.raises(ConfigurationError, match=kwarg):
        SAW(agent_id="a", **{kwarg: wrong})


def test_component_class_instead_of_instance_rejected():
    with pytest.raises(ConfigurationError, match=r"permission_policy .*the class DenyAllPolicy itself\. Did you mean DenyAllPolicy\(\)\?"):
        SAW(agent_id="a", permission_policy=DenyAllPolicy)  # type: ignore[arg-type]


@pytest.mark.parametrize("base", [PermissionPolicy, ResultValidator, SecurityLogger])
def test_interfaces_cannot_be_instantiated(base):
    with pytest.raises(TypeError):
        base()


@pytest.mark.parametrize("base", [PermissionPolicy, ResultValidator, SecurityLogger])
def test_subclass_missing_method_cannot_be_instantiated(base):
    Incomplete = type("Incomplete", (base,), {})
    with pytest.raises(TypeError):
        Incomplete()


def test_shared_custom_component_is_allowed():
    # Stateless components may be shared; the README warns about stateful ones.
    policy = DenyAllPolicy()
    assert SAW(agent_id="a", permission_policy=policy).permission_policy is SAW(agent_id="b", permission_policy=policy).permission_policy

"""Mandatory permission enforcement that custom policies cannot weaken.

The final decision for a call is the *stricter* of two decisions:

1. The **configuration decision**: deny tools not in ``allowed_tools``;
   require approval for tools in ``approval_required``; otherwise allow.
2. The **policy decision** from the agent's ``PermissionPolicy``.

Strictness order: DENY > REQUIRE_APPROVAL > ALLOW. So a custom policy can
make a call stricter, but never less strict than the configuration.

The policy is not consulted for tools the configuration denies. If the policy
raises or returns something other than a ``PolicyDecision``, the call is
denied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from saw.calls import ToolCall
from saw.config import AgentConfig
from saw.execution import discard_awaitable
from saw.policies.base import Decision, PermissionPolicy, PolicyDecision

_STRICTNESS = {Decision.ALLOW: 0, Decision.REQUIRE_APPROVAL: 1, Decision.DENY: 2}


def config_decision(call: ToolCall, config: AgentConfig) -> PolicyDecision:
    """Decision required by the agent's configuration alone."""
    name = call.tool_name
    if name not in config.allowed_tools:
        return PolicyDecision.deny(
            f"tool {name!r} is not in allowed_tools for agent {config.agent_id!r}"
        )
    if name in config.approval_required:
        return PolicyDecision.require_approval(
            f"tool {name!r} requires approval for agent {config.agent_id!r}"
        )
    return PolicyDecision.allow(
        f"tool {name!r} is allowed for agent {config.agent_id!r}"
    )


@dataclass(frozen=True)
class EnforcedDecision:
    """The final decision and how it was reached.

    Attributes:
        final: The decision the runtime acts on.
        config: The configuration decision.
        policy: The policy's own decision, or ``None`` if the policy was not
            consulted or failed.
        policy_error: Exception raised by the policy, if any.
        policy_problem: Short description of a policy failure, if any.
    """

    final: PolicyDecision
    config: PolicyDecision
    policy: PolicyDecision | None = None
    policy_error: BaseException | None = None
    policy_problem: str | None = None

    def details(self) -> dict[str, Any]:
        """Event details describing how the decision was reached."""
        out: dict[str, Any] = {"config_decision": self.config.decision.value}
        if self.policy is not None:
            out["policy_decision"] = self.policy.decision.value
        if self.policy_problem is not None:
            out["policy_problem"] = self.policy_problem
        return out


def enforce(
    policy: PermissionPolicy, call: ToolCall, config: AgentConfig
) -> EnforcedDecision:
    """Return the final decision for ``call``. Never raises for policy errors."""
    floor = config_decision(call, config)
    if floor.decision is Decision.DENY:
        return EnforcedDecision(final=floor, config=floor)

    try:
        result = policy.evaluate(call, config)
    except Exception as exc:
        problem = f"permission policy raised {type(exc).__name__}"
        return EnforcedDecision(
            final=PolicyDecision.deny(f"{problem}; call denied"),
            config=floor,
            policy_error=exc,
            policy_problem=problem,
        )

    if not isinstance(result, PolicyDecision):
        discard_awaitable(result)
        problem = (
            f"permission policy returned {type(result).__name__}, "
            "not a PolicyDecision"
        )
        return EnforcedDecision(
            final=PolicyDecision.deny(f"{problem}; call denied"),
            config=floor,
            policy_problem=problem,
        )

    # A PolicyDecision subclass could carry a decision that is not a real
    # Decision member (for example the plain string "allow"); deny it.
    if type(result.decision) is not Decision or not isinstance(result.reason, str):
        problem = "permission policy returned a malformed PolicyDecision"
        return EnforcedDecision(
            final=PolicyDecision.deny(f"{problem}; call denied"),
            config=floor,
            policy_problem=problem,
        )

    stricter = (
        result
        if _STRICTNESS[result.decision] >= _STRICTNESS[floor.decision]
        else floor
    )
    return EnforcedDecision(final=stricter, config=floor, policy=result)


__all__ = ["config_decision", "enforce", "EnforcedDecision"]

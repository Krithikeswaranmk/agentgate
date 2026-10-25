"""Default deny-by-default permission policy."""

from __future__ import annotations

from saw.calls import ToolCall
from saw.config import AgentConfig
from saw.policies.base import PermissionPolicy, PolicyDecision
from saw.policies.enforcement import config_decision


class DefaultPermissionPolicy(PermissionPolicy):
    """Allows only tools listed in the agent's configuration.

    Rules, in order:

    1. Tool not in ``config.allowed_tools`` -> DENY.
    2. Tool in ``config.approval_required`` -> REQUIRE_APPROVAL.
    3. Otherwise -> ALLOW.

    These are the same rules the runtime always enforces as a minimum (see
    ``saw.policies.enforcement``), so this policy adds nothing beyond them.
    Write a custom policy to add stricter rules. The policy is stateless, so
    one instance is safe to share.
    """

    def evaluate(self, call: ToolCall, config: AgentConfig) -> PolicyDecision:
        return config_decision(call, config)


__all__ = ["DefaultPermissionPolicy"]

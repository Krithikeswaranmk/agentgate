"""Permission-policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from saw.calls import ToolCall
from saw.config import AgentConfig


class Decision(str, Enum):
    """Outcome of a permission check."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class PolicyDecision:
    """A permission decision and the reason for it.

    Use the ``allow``, ``deny`` and ``require_approval`` constructors rather
    than building one by hand.

    Raises:
        TypeError: If ``decision`` is not a :class:`Decision` or ``reason``
            is not a non-empty string.
    """

    decision: Decision
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision):
            raise TypeError(
                f"decision must be a Decision, got {type(self.decision).__name__}"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise TypeError("reason must be a non-empty string")

    @classmethod
    def allow(cls, reason: str) -> PolicyDecision:
        return cls(Decision.ALLOW, reason)

    @classmethod
    def deny(cls, reason: str) -> PolicyDecision:
        return cls(Decision.DENY, reason)

    @classmethod
    def require_approval(cls, reason: str) -> PolicyDecision:
        return cls(Decision.REQUIRE_APPROVAL, reason)

    @property
    def is_allowed(self) -> bool:
        """True only for an unconditional ALLOW."""
        return self.decision is Decision.ALLOW


class PermissionPolicy(ABC):
    """Decides whether an agent may make a tool call.
    Pass an instance as ``SAW(permission_policy=...)``.

    **Implement:** ``evaluate(self, call, config)`` as a regular (not
    ``async``) method.

    **Inputs:**

    * ``call``: a :class:`saw.ToolCall`. ``call.tool_name`` is a registered
      tool, ``call.arguments`` holds the arguments **after** schema
      validation (read-only mapping), and ``call.call_id`` matches the
      call's security events.
    * ``config``: the agent's immutable :class:`saw.AgentConfig`.

    **Return:** a :class:`PolicyDecision`, made with
    ``PolicyDecision.allow(reason)``, ``.deny(reason)`` or
    ``.require_approval(reason)``. The reason is shown to the model and
    logged, so do not put secrets in it.

    **What SAW does with your answer:** the final decision is the *stricter*
    of yours and the configuration's (deny > require approval > allow).

    * You can deny, or require approval for, a call the configuration allows.
    * You **cannot** allow a tool missing from ``allowed_tools``, or skip an
      approval listed in ``approval_required``. You are not even called for
      tools the configuration denies.
    * Raising, or returning anything other than a ``PolicyDecision``, denies
      the call. The exception is kept on ``outcome.error`` for debugging.

    So a custom policy only needs to express its *extra* rules; the
    configuration's limits are applied regardless.

    **Notes:**

    * Do not modify ``call.arguments`` or anything inside it; the tool
      receives the same values.
    * Keep no per-agent mutable state if one instance is shared between
      several ``SAW`` instances.
    * Checked when ``SAW`` is created: the method must accept two positional
      arguments and must not be ``async`` or a generator.
    """

    @abstractmethod
    def evaluate(self, call: ToolCall, config: AgentConfig) -> PolicyDecision:
        """Return the permission decision for ``call`` under ``config``."""


__all__ = ["Decision", "PolicyDecision", "PermissionPolicy"]

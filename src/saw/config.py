"""Per-agent configuration.

An :class:`AgentConfig` is immutable once created. Each :class:`saw.SAW`
instance holds exactly one, so two agents' configurations cannot overwrite
each other through SAW's API.

Limitation: Python cannot prevent code running in the same process from
deliberately bypassing immutability (for example with ``object.__setattr__``).
Immutability here protects against accidental changes, not hostile in-process
code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from saw.exceptions import ConfigurationError

#: Allowed format for agent IDs and tool names: starts with a letter or digit,
#: then letters, digits, ``.``, ``_`` or ``-``; at most 128 characters.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Default maximum number of characters of text in a tool result.
DEFAULT_MAX_RESULT_CHARS = 100_000

#: Allowed values for ``AgentConfig.oversized_results``.
OVERSIZED_RESULT_POLICIES = frozenset({"reject", "truncate"})

#: Default maximum nesting depth and number of values in a tool result.
DEFAULT_MAX_RESULT_DEPTH = 32
DEFAULT_MAX_RESULT_NODES = 100_000

#: Default lifetime of a pending approval request, in seconds (15 minutes).
DEFAULT_APPROVAL_TTL_SECONDS = 900.0

#: Default maximum number of approval requests pending at once per agent.
DEFAULT_MAX_PENDING_APPROVALS = 100


def _check_name(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(
            f"{field_name} must be a string, got {type(value).__name__}"
        )
    if not NAME_PATTERN.fullmatch(value):
        raise ConfigurationError(
            f"{field_name} {value!r} is invalid: use 1-128 characters, starting "
            "with a letter or digit, containing only letters, digits, '.', '_' or '-'"
        )
    return value


def _to_name_set(value: object, field_name: str) -> frozenset[str]:
    # A bare string is iterable, so allowed_tools="web_search" would otherwise
    # silently become {"w", "e", "b", ...}. Reject it explicitly.
    if isinstance(value, (str, bytes)):
        raise ConfigurationError(
            f"{field_name} must be a collection of tool names, not a single string; "
            f"did you mean [{value!r}]?"
        )
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise ConfigurationError(
            f"{field_name} must be a set, list or tuple of tool names, "
            f"got {type(value).__name__}"
        )
    return frozenset(_check_name(name, f"{field_name} entry") for name in value)


@dataclass(frozen=True)
class AgentConfig:
    """Security configuration for a single agent.

    Attributes:
        agent_id: Label for the agent. Used in security events and errors.
            It is *not* an authenticated identity; see the README.
        allowed_tools: Names of tools this agent may call. Empty by default,
            which means every tool call will be denied (deny-by-default).
        approval_required: Tools that need explicit approval before running.
            Must be a subset of ``allowed_tools``.
        max_result_chars: Maximum characters of text in a tool result:
            the total length of all strings and dict keys in it. A positive
            integer, default 100,000. Enforced by the runtime for every
            result, whatever result validator is used.
        oversized_results: What to do with a result over the limit:
            ``"reject"`` (default) blocks it. ``"truncate"`` shortens a
            plain-text result to fit, with a visible marker; structured
            results over the limit are still rejected.

        max_result_depth: Maximum nesting depth of a tool result. Positive
            integer, default 32.
        max_result_nodes: Maximum number of values in a tool result.
            Positive integer, default 100,000.
        approval_ttl_seconds: How long an approval request stays valid.
            Finite number greater than 0, default 900 (15 minutes).
        max_pending_approvals: Maximum approval requests waiting at once for
            this agent. Further calls that need approval are denied until
            some are decided or expire. Positive integer, default 100.

    The result limits are enforced by the runtime for every result, whatever
    result validator is used. SAW does not limit how long a tool runs; see
    the README.

    Raises:
        ConfigurationError: If any field is invalid.
    """

    agent_id: str
    allowed_tools: frozenset[str] = frozenset()
    approval_required: frozenset[str] = frozenset()
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    oversized_results: str = "reject"
    max_result_depth: int = DEFAULT_MAX_RESULT_DEPTH
    max_result_nodes: int = DEFAULT_MAX_RESULT_NODES
    approval_ttl_seconds: float = DEFAULT_APPROVAL_TTL_SECONDS
    max_pending_approvals: int = DEFAULT_MAX_PENDING_APPROVALS

    def __post_init__(self) -> None:
        _check_name(self.agent_id, "agent_id")
        allowed = _to_name_set(self.allowed_tools, "allowed_tools")
        approval = _to_name_set(self.approval_required, "approval_required")

        not_allowed = approval - allowed
        if not_allowed:
            raise ConfigurationError(
                "approval_required lists tools that are not in allowed_tools: "
                f"{sorted(not_allowed)}. A tool must be allowed before it can "
                "require approval."
            )

        for field_name in (
            "max_result_chars", "max_result_depth", "max_result_nodes",
            "max_pending_approvals",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(
                    f"{field_name} must be a positive integer, got {value!r}"
                )
        ttl = self.approval_ttl_seconds
        if (
            isinstance(ttl, bool)
            or not isinstance(ttl, (int, float))
            or not math.isfinite(ttl)
            or ttl <= 0
        ):
            raise ConfigurationError(
                f"approval_ttl_seconds must be a finite number greater than 0, got {ttl!r}"
            )
        if self.oversized_results not in OVERSIZED_RESULT_POLICIES:
            raise ConfigurationError(
                "oversized_results must be 'reject' or 'truncate', "
                f"got {self.oversized_results!r}"
            )

        # Normalise to immutable types. The dataclass is frozen, so assignment
        # inside __post_init__ has to go through object.__setattr__.
        object.__setattr__(self, "allowed_tools", allowed)
        object.__setattr__(self, "approval_required", approval)
        object.__setattr__(self, "approval_ttl_seconds", float(ttl))


__all__ = [
    "AgentConfig",
    "NAME_PATTERN",
    "DEFAULT_MAX_RESULT_CHARS",
    "DEFAULT_MAX_RESULT_DEPTH",
    "DEFAULT_MAX_RESULT_NODES",
    "DEFAULT_APPROVAL_TTL_SECONDS",
    "DEFAULT_MAX_PENDING_APPROVALS",
    "OVERSIZED_RESULT_POLICIES",
]

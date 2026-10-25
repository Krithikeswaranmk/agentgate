"""Security-event structures passed to security loggers."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


class EventType(str, Enum):
    """Kinds of security event.

    Emitted since Sprint 2: ``AGENT_INITIALIZED``, ``TOOL_CALL_RECEIVED``,
    ``TOOL_CALL_ALLOWED``, ``TOOL_CALL_DENIED`` (unknown tool, invalid
    arguments or permission denied; ``details["status"]`` says which),
    ``APPROVAL_REQUIRED``, ``TOOL_EXECUTED``, ``TOOL_FAILED`` (including
    interrupted calls; ``details["status"]`` is ``"interrupted"``) and
    ``RESULT_REJECTED`` (``details["check"]`` says which check).
    Approvals add ``APPROVAL_GRANTED`` and ``APPROVAL_DENIED`` (with
    ``details["approver"]``) and ``APPROVAL_EXPIRED``; they share the
    original call's ``call_id``.
    """

    AGENT_INITIALIZED = "agent_initialized"
    TOOL_CALL_RECEIVED = "tool_call_received"
    TOOL_CALL_ALLOWED = "tool_call_allowed"
    TOOL_CALL_DENIED = "tool_call_denied"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_EXPIRED = "approval_expired"
    TOOL_EXECUTED = "tool_executed"
    TOOL_FAILED = "tool_failed"
    RESULT_REJECTED = "result_rejected"


#: Event types that indicate something was blocked or went wrong.
#: The default logger records these at WARNING level.
WARNING_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_DENIED,
        EventType.TOOL_FAILED,
        EventType.RESULT_REJECTED,
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class SecurityEvent:
    """An immutable record of a security-relevant action.

    Attributes:
        event_type: What happened.
        agent_id: The agent the event belongs to.
        tool_name: The tool involved, if any.
        call_id: The :class:`saw.ToolCall` ID, if any, to correlate events.
        decision: The policy decision value, if any (for example ``"deny"``).
        reason: Human-readable explanation, if any.
        details: Extra context. Stored as a read-only shallow copy.
            SAW does not redact this; do not put secrets in it.
        timestamp: Timezone-aware UTC time the event was created.
        event_id: Unique ID for this event.
    """

    event_type: EventType
    agent_id: str
    tool_name: str | None = None
    call_id: str | None = None
    decision: str | None = None
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)
    event_id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise TypeError(
                f"event_type must be an EventType, got {type(self.event_type).__name__}"
            )
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise TypeError("agent_id must be a non-empty string")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view suitable for structured logging.

        The timestamp is an ISO 8601 string and the event type is its string
        value. ``details`` is copied as-is, so it is JSON-serialisable only if
        its values are.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "decision": self.decision,
            "reason": self.reason,
            "details": dict(self.details),
        }


__all__ = ["EventType", "SecurityEvent", "WARNING_EVENT_TYPES"]

from datetime import datetime, timezone

import pytest

from saw import EventType, SecurityEvent


def test_event_defaults():
    event = SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a")
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset().total_seconds() == 0
    assert len(event.event_id) == 32
    assert event.tool_name is None and event.decision is None
    assert dict(event.details) == {}


def test_event_ids_are_unique():
    ids = {SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a").event_id for _ in range(100)}
    assert len(ids) == 100


def test_event_is_immutable():
    event = SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a", details={"k": 1})
    with pytest.raises(AttributeError):
        event.agent_id = "b"  # type: ignore[misc]
    with pytest.raises(TypeError):
        event.details["k"] = 2  # type: ignore[index]


def test_details_are_copied():
    details = {"k": 1}
    event = SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a", details=details)
    details["k"] = 2
    assert event.details["k"] == 1


def test_to_dict():
    ts = datetime(2026, 10, 24, 8, 0, tzinfo=timezone.utc)
    event = SecurityEvent(
        EventType.TOOL_CALL_DENIED,
        agent_id="a",
        tool_name="send_email",
        call_id="c1",
        decision="deny",
        reason="not allowed",
        details={"k": 1},
        timestamp=ts,
        event_id="e1",
    )
    assert event.to_dict() == {
        "event_id": "e1",
        "event_type": "tool_call_denied",
        "timestamp": "2026-10-24T08:00:00+00:00",
        "agent_id": "a",
        "tool_name": "send_email",
        "call_id": "c1",
        "decision": "deny",
        "reason": "not allowed",
        "details": {"k": 1},
    }


def test_event_type_must_be_enum():
    with pytest.raises(TypeError, match="EventType"):
        SecurityEvent("agent_initialized", agent_id="a")  # type: ignore[arg-type]


@pytest.mark.parametrize("agent_id", ["", None])
def test_event_requires_agent_id(agent_id):
    with pytest.raises(TypeError, match="agent_id"):
        SecurityEvent(EventType.AGENT_INITIALIZED, agent_id=agent_id)  # type: ignore[arg-type]


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a", timestamp=datetime(2026, 1, 1))


def test_details_must_be_mapping():
    with pytest.raises(TypeError, match="details"):
        SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a", details=["x"])  # type: ignore[arg-type]

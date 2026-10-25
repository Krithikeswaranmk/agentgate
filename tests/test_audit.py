import json
import logging

import pytest

from saw import DefaultSecurityLogger, EventType, SecurityEvent
from saw.audit import DEFAULT_LOGGER_NAME


def test_default_logger_name():
    assert DefaultSecurityLogger().logger.name == DEFAULT_LOGGER_NAME


def test_info_event_written_as_json(caplog):
    caplog.set_level(logging.INFO, logger=DEFAULT_LOGGER_NAME)
    event = SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a", details={"k": "v"})
    DefaultSecurityLogger().log(event)

    (record,) = caplog.records
    assert record.levelno == logging.INFO
    data = json.loads(record.getMessage())
    assert data["event_type"] == "agent_initialized"
    assert data["agent_id"] == "a"
    assert data["details"] == {"k": "v"}
    assert record.saw_event["event_id"] == event.event_id


@pytest.mark.parametrize(
    "event_type", [EventType.TOOL_CALL_DENIED, EventType.TOOL_FAILED, EventType.RESULT_REJECTED]
)
def test_problem_events_logged_at_warning(caplog, event_type):
    caplog.set_level(logging.INFO, logger=DEFAULT_LOGGER_NAME)
    DefaultSecurityLogger().log(SecurityEvent(event_type, agent_id="a"))
    assert caplog.records[0].levelno == logging.WARNING


def test_non_json_details_do_not_break_logging(caplog):
    caplog.set_level(logging.INFO, logger=DEFAULT_LOGGER_NAME)
    DefaultSecurityLogger().log(
        SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a", details={"obj": object()})
    )
    data = json.loads(caplog.records[0].getMessage())
    assert data["details"]["obj"].startswith("<object object")


def test_custom_stdlib_logger_used(caplog):
    custom = logging.getLogger("my.app.audit")
    caplog.set_level(logging.INFO, logger="my.app.audit")
    DefaultSecurityLogger(custom).log(SecurityEvent(EventType.AGENT_INITIALIZED, agent_id="a"))
    assert caplog.records[0].name == "my.app.audit"


def test_non_logger_rejected():
    with pytest.raises(TypeError, match="logging.Logger"):
        DefaultSecurityLogger("saw.security")  # type: ignore[arg-type]


def test_saw_package_does_not_shadow_stdlib_logging():
    # Regression guard for renaming saw/logging -> saw/audit.
    import saw.audit.default as mod

    assert mod.logging is logging

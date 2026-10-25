"""Shared test helpers."""

from __future__ import annotations

import pytest

from saw import SecurityEvent, SecurityLogger


class RecordingLogger(SecurityLogger):
    """Keeps events in memory so tests can inspect them."""

    def __init__(self) -> None:
        self.events: list[SecurityEvent] = []

    def log(self, event: SecurityEvent) -> None:
        self.events.append(event)


@pytest.fixture
def recording_logger() -> RecordingLogger:
    return RecordingLogger()


def pytest_configure(config):
    # Also registered in pyproject.toml; repeated here so the tests run
    # cleanly when copied next to an installed wheel without the project file.
    config.addinivalue_line(
        "markers", "live: calls a real LLM provider; skipped unless configured"
    )

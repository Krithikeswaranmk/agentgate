"""Default security logger built on Python's standard ``logging`` module."""

from __future__ import annotations

import json
import logging

from saw.audit.base import SecurityLogger
from saw.events import WARNING_EVENT_TYPES, SecurityEvent

#: Name of the standard-library logger used by default.
DEFAULT_LOGGER_NAME = "saw.security"


class DefaultSecurityLogger(SecurityLogger):
    """Writes each event as one JSON line to a standard-library logger.

    Denials, failures and rejected results are logged at WARNING; everything
    else at INFO. The event dict is also attached to the log record as
    ``record.saw_event`` for structured handlers.

    Nothing is shown unless your application configures logging (for example
    ``logging.basicConfig(level=logging.INFO)``). Python's fallback handler
    prints WARNING and above to stderr.

    Args:
        logger: Standard-library logger to write to. Defaults to
            ``logging.getLogger("saw.security")``.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger")
        self._logger = logger or logging.getLogger(DEFAULT_LOGGER_NAME)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def log(self, event: SecurityEvent) -> None:
        data = event.to_dict()
        level = (
            logging.WARNING if event.event_type in WARNING_EVENT_TYPES else logging.INFO
        )
        # default=str keeps logging from failing on non-JSON values in details.
        message = json.dumps(data, default=str, sort_keys=True)
        self._logger.log(level, "%s", message, extra={"saw_event": data})


__all__ = ["DefaultSecurityLogger", "DEFAULT_LOGGER_NAME"]

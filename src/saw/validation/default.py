"""Default result validator: adds no rules beyond SAW's mandatory checks."""

from __future__ import annotations

from typing import Any

from saw.calls import ToolCall
from saw.config import AgentConfig
from saw.validation.base import ResultValidator, ValidationResult


class DefaultResultValidator(ResultValidator):
    """Accepts every result that has passed SAW's mandatory result checks.

    The runtime always checks, before any validator runs, that a result is
    JSON-compatible, within the agent's ``max_result_depth``,
    ``max_result_nodes`` and ``max_result_chars``, and (later) that it
    matches the tool's ``result_schema``. Those checks cannot be removed by
    replacing the validator. This default therefore adds nothing; subclass
    :class:`ResultValidator` (or this class) to add application rules.
    """

    def validate(
        self, call: ToolCall, result: Any, config: AgentConfig
    ) -> ValidationResult:
        return ValidationResult.accept()


__all__ = ["DefaultResultValidator"]

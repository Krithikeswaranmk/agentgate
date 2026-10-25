"""Tool-result validator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from saw.calls import ToolCall
from saw.config import AgentConfig


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a tool result.

    Use :meth:`accept` and :meth:`reject` rather than building one by hand.
    A rejection must carry a reason.
    """

    valid: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if not self.valid and not self.reason.strip():
            raise TypeError("a rejected result must include a reason")

    @classmethod
    def accept(cls) -> ValidationResult:
        return cls(True)

    @classmethod
    def reject(cls, reason: str) -> ValidationResult:
        return cls(False, reason)


class ResultValidator(ABC):
    """Checks a tool's result before it reaches the agent.
    Pass an instance as ``SAW(tool_result_validator=...)``.

    **Implement:** ``validate(self, call, result, config)`` as a regular (not
    ``async``) method.

    **Inputs:**

    * ``call``: the :class:`saw.ToolCall`, with validated arguments.
    * ``result``: what the tool returned. If the tool has a
      ``result_schema`` and returned an instance of it, this is the
      instance converted to a plain dict.
    * ``config``: the agent's immutable :class:`saw.AgentConfig`.

    **Return:** a :class:`ValidationResult`: ``ValidationResult.accept()`` or
    ``ValidationResult.reject(reason)``. The reason is shown to the model and
    logged, so it must not repeat content from the result.

    **What SAW does with your answer:**

    * Accept: the result goes on to the remaining checks.
    * Reject: the result is withheld and the call ends with status
      ``result_rejected``.
    * Raising, or returning anything other than a ``ValidationResult``,
      rejects the result.

    **Cannot change:** the agent's ``max_result_chars`` size limit and the
    tool's ``result_schema``. Both still run after an accept, so a
    validator can only make result checking stricter.

    **Notes:**

    * Validators accept or reject; they must **not modify** the result. The
      agent receives the object the tool returned, so any change you make
      in place would reach the agent unchecked. SAW cannot detect this.
    * Checked when ``SAW`` is created: the method must accept three
      positional arguments and must not be ``async`` or a generator.
    """

    @abstractmethod
    def validate(
        self, call: ToolCall, result: Any, config: AgentConfig
    ) -> ValidationResult:
        """Return whether ``result`` from ``call`` may reach the agent."""


__all__ = ["ValidationResult", "ResultValidator"]

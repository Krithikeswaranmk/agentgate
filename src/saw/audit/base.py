"""Security-logger interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from saw.events import SecurityEvent


class SecurityLogger(ABC):
    """Records security events. Pass an instance as ``SAW(logger=...)``.

    **Implement:** ``log(self, event)`` as a regular (not ``async``) method.

    **Input:** one immutable :class:`saw.SecurityEvent`. ``event.to_dict()``
    gives a plain dict. Events never contain argument values or rejected
    results; ``details`` may contain names, types, counts and reasons.

    **Return:** ``None``. Signal failure by **raising**.

    **What SAW does with your answer:**

    * Returns ``None``: the event counts as recorded.
    * Raises, or returns anything other than ``None`` (for example ``False``,
      or the coroutine an ``async def log`` would return): the event counts
      as **not** recorded. During ``SAW(...)`` this raises
      ``ConfigurationError``. During a call it raises ``AuditError``: if the
      tool has not run yet it does not run, and if it has run its result is
      withheld. There is no fallback logger, so SAW never runs a tool
      without an audit trail.

    **Cannot change:** which events are produced or when, the decisions SAW
    makes, or the no-unaudited-execution rule.

    **Notes:**

    * Called synchronously on the calling thread. If several threads use one
      ``SAW`` instance, ``log`` must be safe to call concurrently.
    * Keep it fast; it runs up to four times per tool call.
    * Checked when ``SAW`` is created: the method must accept one positional
      argument and must not be ``async`` or a generator.
    """

    @abstractmethod
    def log(self, event: SecurityEvent) -> None:
        """Record ``event``, or raise if it cannot be recorded."""


__all__ = ["SecurityLogger"]

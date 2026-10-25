"""Standard exceptions raised by SAW.

Every SAW exception derives from :class:`SAWError`, so callers can catch the
whole family with one ``except`` clause. None derive from ``ValueError`` or
``TypeError``, so a broad ``except ValueError`` cannot swallow them.

Expected decisions (a denied call, a call needing approval, bad arguments)
are *not* reported by raising. ``SAW.execute_tool`` returns a
:class:`saw.ToolCallOutcome`; the matching exception is raised only when code
reads ``outcome.value`` for a call that did not run. In that case the
exception's ``outcome`` attribute holds the outcome.

Raised directly by SAW:

* :class:`ConfigurationError` - invalid configuration or tool registration.
* :class:`AuditError` - the security logger failed during a tool call.
* :class:`ApprovalError` - an approval decision could not be applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from saw.calls import ToolCallOutcome


class SAWError(Exception):
    """Base class for all SAW errors."""

    #: The tool-call outcome this error describes, when there is one.
    outcome: ToolCallOutcome | None = None


class ConfigurationError(SAWError):
    """Configuration, component wiring or tool registration is invalid.

    Raised at setup time, before any tool can run.
    """


class ToolNotRegisteredError(SAWError):
    """The call named a tool that is not registered."""


class ArgumentValidationError(SAWError):
    """The call's arguments are malformed or fail schema validation.

    Attributes:
        errors: One message per problem found, such as
            ``"path: expected string, got int"``. Messages name types and
            fields but never include argument values.
    """

    errors: tuple[str, ...] = ()


class PermissionDeniedError(SAWError):
    """The agent is not permitted to make this call."""


class ApprovalRequiredError(SAWError):
    """The call needs approval and has not been approved.

    This is deliberately *not* a subclass of :class:`PermissionDeniedError`:
    a pending approval is a different state from a denial, and callers may
    want to handle it differently. In both cases the tool does not run.
    """


class ApprovalError(SAWError):
    """An approval decision could not be applied.

    Raised directly by ``SAW.approve`` and ``SAW.deny`` when the request ID
    is unknown or already decided, the approver is invalid, or the call is
    made from inside a running tool. The tool does not run.
    """


class ResultValidationError(SAWError):
    """The tool ran, but its result was rejected and withheld."""


class ToolExecutionError(SAWError):
    """The tool raised an exception while running.

    The original exception is chained as ``__cause__``.
    """


class AuditError(SAWError):
    """The security logger failed while recording a tool call.

    SAW does not run tools without a working audit trail (see the README).

    Attributes:
        tool_executed: ``False`` if the failure happened before the tool ran,
            so the tool did not run. ``True`` if the tool had already run: its
            side effects happened, but SAW has withheld the result.
        outcome: For a blocked or failed call, the outcome that could not be
            recorded. ``None`` when a successful result was withheld.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_executed: bool,
        outcome: ToolCallOutcome | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_executed = tool_executed
        self.outcome = outcome


__all__ = [
    "SAWError",
    "ConfigurationError",
    "ToolNotRegisteredError",
    "ArgumentValidationError",
    "PermissionDeniedError",
    "ApprovalRequiredError",
    "ApprovalError",
    "ResultValidationError",
    "ToolExecutionError",
    "AuditError",
]

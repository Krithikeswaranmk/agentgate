"""Tool-call request and outcome structures."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from saw.exceptions import (
    ApprovalRequiredError,
    ArgumentValidationError,
    PermissionDeniedError,
    ResultValidationError,
    SAWError,
    ToolExecutionError,
    ToolNotRegisteredError,
)


@dataclass(frozen=True)
class ToolCall:
    """A request from an agent to call a tool.

    The runtime passes policies a ``ToolCall`` whose ``arguments`` have
    already passed schema validation.

    Attributes:
        tool_name: Name of the requested tool.
        arguments: Keyword arguments for the tool. Stored as a read-only
            shallow copy, so later changes to the caller's dict do not affect
            the call. Nested values (lists, dicts) are not deep-copied.
        call_id: Unique ID used to correlate security events for this call.

    Raises:
        ArgumentValidationError: If ``tool_name`` is not a non-empty string or
            ``arguments`` is not a mapping with string keys.
    """

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ArgumentValidationError("tool_name must be a non-empty string")
        if not isinstance(self.arguments, Mapping):
            raise ArgumentValidationError(
                f"arguments must be a mapping, got {type(self.arguments).__name__}"
            )
        bad_keys = [k for k in self.arguments if not isinstance(k, str)]
        if bad_keys:
            raise ArgumentValidationError(
                f"argument names must be strings, got {bad_keys!r}"
            )
        object.__setattr__(
            self, "arguments", MappingProxyType(dict(self.arguments))
        )


class CallStatus(str, Enum):
    """What happened to a tool call."""

    #: All checks passed, the tool ran, and its result passed every result check.
    EXECUTED = "executed"
    #: The agent is not permitted to make this call. The tool did not run.
    DENIED = "denied"
    #: The call needs approval. The tool did not run.
    APPROVAL_REQUIRED = "approval_required"
    #: No tool with this name is registered. Nothing ran.
    UNKNOWN_TOOL = "unknown_tool"
    #: The arguments failed validation. The tool did not run.
    INVALID_ARGUMENTS = "invalid_arguments"
    #: The tool ran and raised an exception, or could not be run.
    FAILED = "failed"
    #: The tool ran, but its result failed a result check and was withheld.
    RESULT_REJECTED = "result_rejected"


_STATUS_ERRORS: dict[CallStatus, type[SAWError]] = {
    CallStatus.DENIED: PermissionDeniedError,
    CallStatus.APPROVAL_REQUIRED: ApprovalRequiredError,
    CallStatus.UNKNOWN_TOOL: ToolNotRegisteredError,
    CallStatus.INVALID_ARGUMENTS: ArgumentValidationError,
    CallStatus.FAILED: ToolExecutionError,
    CallStatus.RESULT_REJECTED: ResultValidationError,
}


class ToolCallOutcome:
    """The result of ``SAW.execute_tool``.

    Check ``ok`` (or ``status``) to handle each case, or read ``value`` to
    get the tool's return value and let SAW raise if the tool did not run.

    Attributes are read-only.
    """

    __slots__ = (
        "_status", "_tool_name", "_call_id", "_reason", "_value", "_error",
        "_truncated", "_approval_request",
    )

    def __init__(
        self,
        status: CallStatus,
        tool_name: str,
        call_id: str,
        reason: str,
        *,
        value: Any = None,
        error: BaseException | None = None,
        truncated: bool = False,
        approval_request: Any = None,
    ) -> None:
        if not isinstance(status, CallStatus):
            raise TypeError("status must be a CallStatus")
        if status is not CallStatus.EXECUTED and value is not None:
            raise TypeError("only an executed call can carry a value")
        if (approval_request is None) != (status is not CallStatus.APPROVAL_REQUIRED):
            raise TypeError(
                "an approval_request is required for, and only allowed with, "
                "status APPROVAL_REQUIRED"
            )
        self._status = status
        self._tool_name = tool_name
        self._call_id = call_id
        self._reason = reason
        self._value = value
        self._error = error
        self._truncated = truncated
        self._approval_request = approval_request

    @property
    def status(self) -> CallStatus:
        return self._status

    @property
    def ok(self) -> bool:
        """True only if the tool ran and its result was accepted."""
        return self._status is CallStatus.EXECUTED

    @property
    def truncated(self) -> bool:
        """True if a plain-text result was shortened to fit the size limit.

        Only possible when the agent's ``oversized_results`` is
        ``"truncate"``. The value then ends with a ``[SAW: result
        truncated ...]`` marker.
        """
        return self._truncated

    @property
    def approval_request(self):
        """The :class:`saw.ApprovalRequest` when status is ``APPROVAL_REQUIRED``.

        ``None`` otherwise. It describes the call for the person deciding;
        it cannot run it. Use ``SAW.approve`` or ``SAW.deny``.
        """
        return self._approval_request

    def for_model(self) -> dict[str, Any]:
        """A message describing this outcome, safe to send back to the model.

        Executed calls give ``{"status": "executed", "result": ...}``.
        Everything else gives the status and the reason, which never contain
        argument values, rejected results or exception messages. A call
        waiting for approval states plainly that it has not been performed.
        """
        if self.ok:
            message: dict[str, Any] = {"status": self._status.value, "result": self._value}
            if self._truncated:
                message["truncated"] = True
            return message
        message = {"status": self._status.value, "error": self._reason}
        if self._status is CallStatus.APPROVAL_REQUIRED:
            message["note"] = (
                "This action has NOT been performed. It is waiting for the "
                "user's approval."
            )
        else:
            message["note"] = "This action was not completed."
        return message

    @property
    def tool_name(self) -> str:
        """The requested tool name (shortened if very long)."""
        return self._tool_name

    @property
    def call_id(self) -> str:
        """Matches the ``call_id`` on this call's security events."""
        return self._call_id

    @property
    def reason(self) -> str:
        """Why the call was blocked or failed. Safe to show to the model."""
        return self._reason

    @property
    def error(self) -> BaseException | None:
        """The underlying exception, if any, for debugging.

        Set when the tool raised, or when a custom policy, result validator or
        result schema raised. Never holds a rejected result. Its message may contain sensitive details, so do not send
        it to the model.
        """
        return self._error

    @property
    def value(self) -> Any:
        """The tool's return value.

        Raises:
            PermissionDeniedError, ApprovalRequiredError,
            ToolNotRegisteredError, ArgumentValidationError,
            ToolExecutionError, ResultValidationError: If the tool did not
                run, or its result was rejected. The
                exception's ``outcome`` attribute is this outcome.
        """
        if self._status is CallStatus.EXECUTED:
            return self._value
        exc = _STATUS_ERRORS[self._status](self._reason)
        exc.outcome = self
        if isinstance(exc, ArgumentValidationError) and isinstance(
            self._error, ArgumentValidationError
        ):
            exc.errors = self._error.errors
        if self._status is CallStatus.FAILED and self._error is not None:
            raise exc from self._error
        raise exc

    def __repr__(self) -> str:
        return (
            f"ToolCallOutcome(status={self._status.value!r}, "
            f"tool_name={self._tool_name!r}, call_id={self._call_id!r})"
        )


__all__ = ["ToolCall", "CallStatus", "ToolCallOutcome"]

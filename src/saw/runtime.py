"""The SAW runtime: one instance per agent.

Every tool call made through :meth:`SAW.execute_tool` goes through these
steps, in order, and stops at the first one that fails:

1. Record that the call was received.
2. Check the tool is registered.
3. Validate the arguments against the tool's schema.
4. Decide permission: the stricter of the agent's configuration and its
   policy (see ``saw.policies.enforcement``).
5. Record that the call was allowed.
6. Run the tool, timing it.
7. Check the result (see ``saw.results``): agent validator, size limit and
   the tool's result schema. A rejected result is withheld.
8. Record the outcome (executed, failed or result rejected) and return it.

A blocked call never reaches step 6, and a result that fails step 7 never
reaches the agent. If the security logger fails at any step, the call stops
and :class:`AuditError` is raised. If the tool is interrupted
(``KeyboardInterrupt``, ``SystemExit``, cancellation), the interruption is
recorded and then re-raised.

If the decision in step 4 is "require approval", the call is stored as a
pending request instead of running (see ``saw.approvals``); it runs only
through :meth:`SAW.approve`, which repeats step 4 and continues from step 5.

SAW does not limit how long a tool runs (see ``saw.execution``).
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

from saw import approvals as approvals_module
from saw.approvals import ApprovalDesk, ApprovalRequest, PendingCall
from saw.audit.base import SecurityLogger
from saw.calls import CallStatus, ToolCall, ToolCallOutcome
from saw.components import (
    LOGGER,
    PERMISSION_POLICY,
    TOOL_RESULT_VALIDATOR,
    resolve_component,
)
from saw.config import AgentConfig
from saw.events import EventType, SecurityEvent
from saw.exceptions import (
    ArgumentValidationError,
    AuditError,
    ConfigurationError,
)
from saw.execution import discard_awaitable, invoke
from saw.policies.base import Decision, PermissionPolicy
from saw.policies.enforcement import enforce
from saw.registry import RegisteredTool, ToolRegistry
from saw.results import check_result
from saw.validation.base import ResultValidator

_MAX_NAME_DISPLAY = 128

_MAX_ARGUMENT_NAMES_LOGGED = 50


def _display_name(tool_name: object) -> str:
    """A safe, bounded string for an agent-supplied tool name."""
    if not isinstance(tool_name, str):
        return f"<{type(tool_name).__name__}>"
    if len(tool_name) > _MAX_NAME_DISPLAY:
        return tool_name[:_MAX_NAME_DISPLAY] + "..."
    return tool_name


def _argument_summary(arguments: object, declared: frozenset[str]) -> dict[str, Any]:
    """What may be logged about the arguments.

    Values are never logged. Names are logged only if the tool's schema
    declares them; any other name is supplied by the model and could carry
    data, so only how many there were is recorded.
    """
    if not isinstance(arguments, Mapping):
        return {"argument_count": 0}
    names = sorted(k for k in arguments if isinstance(k, str) and k in declared)
    summary: dict[str, Any] = {
        "argument_count": len(arguments),
        "argument_names": names[:_MAX_ARGUMENT_NAMES_LOGGED],
    }
    undeclared = len(arguments) - len(names)
    if undeclared:
        summary["undeclared_argument_count"] = undeclared
    return summary


class SAW:
    """Security runtime for a single agent.

    Pass either ``agent_id`` (defaults for everything else) or a full
    :class:`AgentConfig`. If both are given, their agent IDs must match.

    Args:
        agent_id: Agent label. Creates ``AgentConfig(agent_id=agent_id)``,
            which allows no tools.
        config: Full agent configuration.
        permission_policy: A :class:`PermissionPolicy`. Defaults to
            :class:`DefaultPermissionPolicy`. Whatever it decides, the
            configuration's limits still apply (see
            ``saw.policies.enforcement``).
        tool_result_validator: A :class:`ResultValidator`, applied to every
            tool result. Defaults to :class:`DefaultResultValidator`. The size
            limit and any tool result schema still apply.
        logger: A :class:`SecurityLogger`. Defaults to
            :class:`DefaultSecurityLogger`. A failing logger stops tool use
            (see the README).

    Custom components are checked here (see ``saw.components``), so a
    mistake such as a missing method parameter or an ``async`` method is
    reported now rather than on the first call.
        approval_desk: The :class:`ApprovalDesk` that decides calls needing
            approval. Without one, such calls are denied. The desk is not
            reachable through this object, so code holding the ``SAW``
            cannot approve anything.
        registry: Tool registry. Defaults to a new, empty registry for this
            instance. Pass a shared registry only if several agents should see
            the same tools; each agent's configuration still decides what it
            may call.

    Raises:
        ConfigurationError: If the configuration or any component is invalid,
            or the security logger fails while recording initialisation.

    Example::

        from dataclasses import dataclass
        from saw import SAW, AgentConfig

        @dataclass
        class ReadDocumentArgs:
            path: str

        def read_document(path: str) -> str:
            ...

        security = SAW(config=AgentConfig(
            agent_id="research-agent",
            allowed_tools={"read_document"},
        ))
        security.register_tool(
            name="read_document", schema=ReadDocumentArgs, function=read_document
        )
        outcome = security.execute_tool("read_document", {"path": "report.txt"})
        if outcome.ok:
            text = outcome.value
    """

    def __init__(
        self,
        agent_id: str | None = None,
        *,
        config: AgentConfig | None = None,
        permission_policy: PermissionPolicy | None = None,
        tool_result_validator: ResultValidator | None = None,
        logger: SecurityLogger | None = None,
        registry: ToolRegistry | None = None,
        approval_desk: ApprovalDesk | None = None,
    ) -> None:
        self._config = self._resolve_config(agent_id, config)
        self._permission_policy = resolve_component(permission_policy, PERMISSION_POLICY)
        self._tool_result_validator = resolve_component(
            tool_result_validator, TOOL_RESULT_VALIDATOR
        )
        self._logger = resolve_component(logger, LOGGER)
        if registry is None:
            registry = ToolRegistry()
        elif not isinstance(registry, ToolRegistry):
            raise ConfigurationError(
                f"registry must be a ToolRegistry, got {type(registry).__name__}"
            )
        self._registry = registry
        if approval_desk is not None and not isinstance(approval_desk, ApprovalDesk):
            raise ConfigurationError(
                f"approval_desk must be an ApprovalDesk, got {type(approval_desk).__name__}"
            )
        # Deliberately not exposed as a property: code holding this SAW
        # object (such as a tool making nested calls) cannot reach approvals.
        self._approval_desk = approval_desk
        self._record_initialization()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _resolve_config(
        agent_id: str | None, config: AgentConfig | None
    ) -> AgentConfig:
        if config is None:
            if agent_id is None:
                raise ConfigurationError("provide either agent_id or config")
            return AgentConfig(agent_id=agent_id)
        if not isinstance(config, AgentConfig):
            raise ConfigurationError(
                f"config must be an AgentConfig, got {type(config).__name__}"
            )
        if agent_id is not None and agent_id != config.agent_id:
            raise ConfigurationError(
                f"agent_id {agent_id!r} does not match config.agent_id "
                f"{config.agent_id!r}"
            )
        return config

    def _record_initialization(self) -> None:
        event = SecurityEvent(
            event_type=EventType.AGENT_INITIALIZED,
            agent_id=self._config.agent_id,
            details={
                "allowed_tools": sorted(self._config.allowed_tools),
                "approval_required": sorted(self._config.approval_required),
                "max_result_chars": self._config.max_result_chars,
                "oversized_results": self._config.oversized_results,
                "max_result_depth": self._config.max_result_depth,
                "max_result_nodes": self._config.max_result_nodes,
                "approval_ttl_seconds": self._config.approval_ttl_seconds,
                "max_pending_approvals": self._config.max_pending_approvals,
                "approval_desk": self._approval_desk is not None,
                "permission_policy": type(self._permission_policy).__name__,
                "tool_result_validator": type(self._tool_result_validator).__name__,
                "logger": type(self._logger).__name__,
            },
        )
        try:
            self._log_event(event)
        except Exception as exc:  # fail closed: no SAW without working audit
            raise ConfigurationError(
                f"logger failed during initialisation: {exc}"
            ) from exc

    def register_tool(
        self,
        name: str,
        *,
        schema: type,
        function: Callable[..., Any],
        result_schema: type | None = None,
        description: str | None = None,
    ) -> None:
        """Register a tool in this runtime's registry.

        Registering does not grant permission: the tool must also be in the
        agent's ``allowed_tools``.

        Args:
            name: Tool name.
            schema: A dataclass, or a Pydantic v2 model with
                ``extra="forbid"``, describing the arguments.
            function: Regular callable taking one keyword argument per schema
                field. Async and generator functions are rejected.
            result_schema: Optional dataclass or Pydantic v2 model (with
                ``extra="forbid"``) the result must match. The tool may
                return a plain dict or an instance of this class.
            description: What the tool does, for the model. Defaults to the
                first paragraph of the function's docstring.

        Raises:
            ConfigurationError: See :meth:`ToolRegistry.register`.
        """
        self._registry.register(
            name, schema=schema, function=function, result_schema=result_schema,
            description=description,
        )

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Describe the tools this agent may call, for an LLM.

        Only tools that are both registered **and** in ``allowed_tools`` are
        listed, so the model is never told about tools it cannot use.
        Tools that need approval are included (the model may request them).
        Each entry is provider-neutral::

            {"name": ..., "description": ..., "parameters": <JSON Schema>}

        The adapters in ``saw.adapters`` convert these to each provider's
        format. Listing a tool does not grant anything: every call is still
        checked by :meth:`execute_tool`.
        """
        definitions = []
        for name in sorted(self._registry.names & self._config.allowed_tools):
            tool = self._registry.get(name)
            if tool is None:
                continue
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema.json_schema(),
            })
        return definitions

    # -------------------------------------------------------------- execution

    def execute_tool(
        self, tool_name: str, arguments: Mapping[str, Any] | None = None
    ) -> ToolCallOutcome:
        """Check and, if every check passes, run a tool.

        Blocked calls are reported through the returned outcome, not by
        raising. Read ``outcome.value`` to get the result or raise.

        Args:
            tool_name: Name of the tool, usually as chosen by the model.
            arguments: Arguments as an object of name/value pairs, usually as
                produced by the model. ``None`` means no arguments.

        Returns:
            A :class:`ToolCallOutcome`.

        Raises:
            AuditError: If the security logger fails. Check
                ``tool_executed`` to see whether the tool had already run.
        """
        call_id = uuid.uuid4().hex
        display = _display_name(tool_name)
        tool = self._registry.get(tool_name)
        declared = frozenset(tool.schema.field_names) if tool is not None else frozenset()

        self._record(
            EventType.TOOL_CALL_RECEIVED,
            tool_name=display,
            call_id=call_id,
            details=_argument_summary(arguments, declared),
            tool_executed=False,
        )

        # Step 2: registration.
        if tool is None:
            return self._block(
                CallStatus.UNKNOWN_TOOL, display, call_id,
                f"tool {display!r} is not registered",
            )

        # Step 3: arguments.
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            return self._block(
                CallStatus.INVALID_ARGUMENTS, tool.name, call_id,
                f"arguments for tool {tool.name!r} must be an object, "
                f"got {type(arguments).__name__}",
            )
        non_string = [type(k).__name__ for k in arguments if not isinstance(k, str)]
        if non_string:
            return self._block(
                CallStatus.INVALID_ARGUMENTS, tool.name, call_id,
                f"argument names for tool {tool.name!r} must be strings, "
                f"got {sorted(set(non_string))}",
            )
        try:
            # redact_keys: unknown argument names come from the model and
            # could carry data, so they are not repeated in the reason.
            kwargs = tool.schema.validate(arguments, redact_keys=True)
        except ArgumentValidationError as exc:
            return self._block(
                CallStatus.INVALID_ARGUMENTS, tool.name, call_id,
                f"invalid arguments for tool {tool.name!r}: {exc}",
                error=exc,
            )
        except Exception as exc:  # defensive: validator bug must not allow the call
            return self._block(
                CallStatus.INVALID_ARGUMENTS, tool.name, call_id,
                f"argument validation for tool {tool.name!r} failed with "
                f"{type(exc).__name__}",
                error=exc,
            )

        # Step 4: permission. The policy gets a copy of the arguments, so
        # nothing it changes in place can alter what the tool receives.
        policy_call = self._call_copy(tool.name, kwargs, call_id)
        if policy_call is None:
            return self._block(
                CallStatus.DENIED, tool.name, call_id,
                f"arguments for tool {tool.name!r} could not be copied for the "
                "permission check; call denied",
            )
        enforced = enforce(self._permission_policy, policy_call, self._config)
        decision = enforced.final.decision
        # Only an exact ALLOW runs; anything else is approval or denial.
        if decision is Decision.ALLOW:
            return self._run_allowed(tool, call_id, kwargs, enforced)
        if decision is Decision.REQUIRE_APPROVAL:
            return self._request_approval(tool, call_id, kwargs, enforced)
        return self._block(
            CallStatus.DENIED, tool.name, call_id, enforced.final.reason,
            details=enforced.details(), error=enforced.policy_error,
        )

    # -------------------------------------------------------------- approvals

    def _request_approval(
        self, tool: RegisteredTool, call_id: str, kwargs: dict[str, Any], enforced: Any
    ) -> ToolCallOutcome:
        """Put the call on the approval desk. The tool does not run."""
        desk = self._approval_desk
        if desk is None:
            return self._block(
                CallStatus.DENIED, tool.name, call_id,
                f"tool {tool.name!r} requires approval, but no approval desk is "
                "configured for this agent; call denied",
                details={**enforced.details(), "approval_unavailable": True},
            )
        desk._expire(self)
        try:
            stored_kwargs = copy.deepcopy(kwargs)
            shown_arguments = MappingProxyType(copy.deepcopy(kwargs))
        except Exception as exc:
            return self._block(
                CallStatus.DENIED, tool.name, call_id,
                f"arguments for tool {tool.name!r} could not be copied for an "
                "approval request; call denied",
                error=exc,
            )
        now = datetime.now(timezone.utc)
        ttl = self._config.approval_ttl_seconds
        request = ApprovalRequest(
            request_id=call_id,
            agent_id=self._config.agent_id,
            tool_name=tool.name,
            arguments=shown_arguments,
            reason=enforced.final.reason,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
        )
        pending = PendingCall(
            request=request, tool=tool, kwargs=stored_kwargs,
            deadline=approvals_module._monotonic() + ttl,
        )
        limit = self._config.max_pending_approvals
        status = desk._submit(self, pending, limit)
        if status == "closed":
            return self._block(
                CallStatus.DENIED, tool.name, call_id,
                f"tool {tool.name!r} requires approval, but the approval desk is "
                "closed; call denied",
                details={**enforced.details(), "approval_unavailable": True},
            )
        if status != "ok":
            return self._block(
                CallStatus.DENIED, tool.name, call_id,
                f"tool {tool.name!r} needs approval, but {limit} approval "
                "requests are already pending for this agent; call denied",
                details={**enforced.details(), "pending_limit": limit},
            )
        try:
            return self._block(
                CallStatus.APPROVAL_REQUIRED, tool.name, call_id, enforced.final.reason,
                details={**enforced.details(), "expires_at": request.expires_at.isoformat()},
                approval_request=request,
            )
        except AuditError:
            desk._withdraw(call_id)  # never leave an unaudited request pending
            raise

    def _approve_pending(
        self, pending: PendingCall, approver: str, note: str | None
    ) -> ToolCallOutcome:
        """Run an approved call. Called by ApprovalDesk.approve only."""
        request = pending.request
        details: dict[str, Any] = {"approver": approver}
        if note:
            details["note"] = str(note)[:200]
        self._record(
            EventType.APPROVAL_GRANTED,
            tool_name=request.tool_name,
            call_id=request.request_id,
            decision=Decision.ALLOW.value,
            reason=f"approved by {approver}",
            details=details,
            tool_executed=False,
        )

        # Re-check before running: same tool, and permission not now denied.
        if self._registry.get(request.tool_name) is not pending.tool:
            return self._block(
                CallStatus.DENIED, request.tool_name, request.request_id,
                f"tool {request.tool_name!r} is no longer the tool that was "
                "approved; call denied",
            )
        policy_call = self._call_copy(request.tool_name, pending.kwargs, request.request_id)
        if policy_call is None:
            return self._block(
                CallStatus.DENIED, request.tool_name, request.request_id,
                "approved arguments could not be copied for the permission "
                "check; call denied",
            )
        enforced = enforce(self._permission_policy, policy_call, self._config)
        # This approval satisfies REQUIRE_APPROVAL; anything else but ALLOW denies.
        if enforced.final.decision not in (Decision.ALLOW, Decision.REQUIRE_APPROVAL):
            return self._block(
                CallStatus.DENIED, request.tool_name, request.request_id,
                f"approved by {approver}, but permission is now denied: "
                f"{enforced.final.reason}",
                details={**enforced.details(), "after_approval": True},
                error=enforced.policy_error,
            )
        return self._run_allowed(
            pending.tool, request.request_id, pending.kwargs, enforced, approved_by=approver
        )

    def _deny_pending(self, pending: PendingCall, approver: str, reason: str) -> ToolCallOutcome:
        """Record a denial. Called by ApprovalDesk.deny only."""
        request = pending.request
        message = f"approval denied by {approver}: {reason[:500]}"
        outcome = ToolCallOutcome(
            CallStatus.DENIED, request.tool_name, request.request_id, message
        )
        self._record(
            EventType.APPROVAL_DENIED,
            tool_name=request.tool_name,
            call_id=request.request_id,
            decision=Decision.DENY.value,
            reason=message,
            details={"status": CallStatus.DENIED.value, "approver": approver},
            tool_executed=False,
            outcome=outcome,
        )
        return outcome

    def _record_expiry(self, pending: PendingCall, *, discarded: bool = False) -> ToolCallOutcome:
        request = pending.request
        if discarded:
            message = (
                f"approval request for tool {request.tool_name!r} was discarded "
                "when the approval desk closed; call not performed"
            )
        else:
            message = (
                f"approval request for tool {request.tool_name!r} expired at "
                f"{request.expires_at.isoformat()}; call not performed"
            )
        outcome = ToolCallOutcome(CallStatus.DENIED, request.tool_name, request.request_id, message)
        details: dict[str, Any] = {"status": CallStatus.DENIED.value}
        if discarded:
            details["discarded"] = True
        self._record(
            EventType.APPROVAL_EXPIRED,
            tool_name=request.tool_name,
            call_id=request.request_id,
            decision=Decision.DENY.value,
            reason=message,
            details=details,
            tool_executed=False,
            outcome=outcome,
        )
        return outcome

    # ------------------------------------------------------------ running

    @staticmethod
    def _call_copy(tool_name: str, kwargs: dict[str, Any], call_id: str) -> ToolCall | None:
        try:
            return ToolCall(tool_name, copy.deepcopy(kwargs), call_id)
        except Exception:
            return None

    def _run_allowed(
        self,
        tool: RegisteredTool,
        call_id: str,
        kwargs: dict[str, Any],
        enforced: Any,
        *,
        approved_by: str | None = None,
    ) -> ToolCallOutcome:
        """Steps 5-8: record the allow, run the tool, check and return the result."""
        allowed_details = enforced.details()
        if approved_by is not None:
            allowed_details["approved_by"] = approved_by
        self._record(
            EventType.TOOL_CALL_ALLOWED,
            tool_name=tool.name,
            call_id=call_id,
            decision=Decision.ALLOW.value,
            reason=(
                f"approved by {approved_by}" if approved_by is not None
                else enforced.final.reason
            ),
            details=allowed_details,
            tool_executed=False,
        )

        # Step 6: run. invoke() marks the thread as inside a tool, so the
        # approval desk refuses decisions made from inside it.
        invocation = invoke(tool.function, kwargs)
        timing = {"duration_ms": invocation.duration_ms}

        if invocation.interrupted:
            reason = f"tool {tool.name!r} {invocation.reason}"
            try:
                self._record(
                    EventType.TOOL_FAILED,
                    tool_name=tool.name,
                    call_id=call_id,
                    reason=reason,
                    details={
                        "status": "interrupted",
                        "exception_type": type(invocation.error).__name__,
                        **timing,
                    },
                    tool_executed=True,
                )
            except AuditError:
                # The interruption wins; the AuditError is kept as __context__.
                raise invocation.error  # noqa: B904
            raise invocation.error

        if not invocation.succeeded:
            reason = f"tool {tool.name!r} failed: {invocation.reason}"
            outcome = ToolCallOutcome(
                CallStatus.FAILED, tool.name, call_id, reason, error=invocation.error
            )
            details: dict[str, Any] = {"status": CallStatus.FAILED.value, **timing}
            if invocation.error is not None:
                details["exception_type"] = type(invocation.error).__name__
            self._record(
                EventType.TOOL_FAILED,
                tool_name=tool.name,
                call_id=call_id,
                reason=reason,
                details=details,
                tool_executed=True,
                outcome=outcome,
            )
            return outcome

        # Step 7: check the result. Validators get their own copy of the call.
        checked = check_result(
            invocation.value,
            call=self._call_copy(tool.name, kwargs, call_id) or ToolCall(tool.name, {}, call_id),
            config=self._config,
            validator=self._tool_result_validator,
            result_schema=tool.result_schema,
        )
        if not checked.accepted:
            outcome = ToolCallOutcome(
                CallStatus.RESULT_REJECTED, tool.name, call_id, checked.reason,
                error=checked.error,
            )
            self._record(
                EventType.RESULT_REJECTED,
                tool_name=tool.name,
                call_id=call_id,
                reason=checked.reason,
                details={
                    "status": CallStatus.RESULT_REJECTED.value,
                    "check": checked.check,
                    "result_type": type(invocation.value).__name__,
                    **timing,
                },
                tool_executed=True,
                outcome=outcome,
            )
            return outcome

        # Step 8: record and return the accepted result.
        executed: dict[str, Any] = {
            "status": CallStatus.EXECUTED.value,
            "result_type": type(checked.value).__name__,
            "result_chars": checked.result_chars,
            **timing,
        }
        if checked.converted:
            executed["converted_from"] = type(invocation.value).__name__
        if checked.truncated:
            executed["truncated"] = True
            executed["original_chars"] = checked.original_chars
        self._record(
            EventType.TOOL_EXECUTED,
            tool_name=tool.name,
            call_id=call_id,
            details=executed,
            tool_executed=True,
        )
        return ToolCallOutcome(
            CallStatus.EXECUTED, tool.name, call_id, "executed",
            value=checked.value, truncated=checked.truncated,
        )

    def _block(
        self,
        status: CallStatus,
        tool_name: str,
        call_id: str,
        reason: str,
        *,
        details: dict[str, Any] | None = None,
        error: BaseException | None = None,
        approval_request: ApprovalRequest | None = None,
    ) -> ToolCallOutcome:
        outcome = ToolCallOutcome(
            status, tool_name, call_id, reason, error=error, approval_request=approval_request
        )
        if status is CallStatus.APPROVAL_REQUIRED:
            event_type, decision = EventType.APPROVAL_REQUIRED, Decision.REQUIRE_APPROVAL
        else:
            event_type, decision = EventType.TOOL_CALL_DENIED, Decision.DENY
        self._record(
            event_type,
            tool_name=tool_name,
            call_id=call_id,
            decision=decision.value,
            reason=reason,
            details={"status": status.value, **(details or {})},
            tool_executed=False,
            outcome=outcome,
        )
        return outcome

    def _record(
        self,
        event_type: EventType,
        *,
        tool_name: str,
        call_id: str,
        tool_executed: bool,
        decision: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
        outcome: ToolCallOutcome | None = None,
    ) -> None:
        event = SecurityEvent(
            event_type=event_type,
            agent_id=self._config.agent_id,
            tool_name=tool_name,
            call_id=call_id,
            decision=decision,
            reason=reason,
            details=details or {},
        )
        try:
            self._log_event(event)
        except Exception as exc:
            consequence = (
                "the tool ran but its result has been withheld"
                if tool_executed
                else "the tool was not run"
            )
            raise AuditError(
                f"security logger failed to record a {event_type.value} event "
                f"for tool {tool_name!r} (call {call_id}); {consequence}. "
                f"Cause: {type(exc).__name__}: {exc}",
                tool_executed=tool_executed,
                outcome=outcome,
            ) from exc

    def _log_event(self, event: SecurityEvent) -> None:
        """Send ``event`` to the logger, enforcing its contract.

        A logger signals failure by raising. Returning anything but ``None``
        (for example a coroutine from an ``async`` method, or ``False``) is
        treated as a failure too, so events can never be lost silently.
        """
        returned = self._logger.log(event)
        if returned is not None:
            discard_awaitable(returned)
            raise TypeError(
                f"{type(self._logger).__name__}.log() returned "
                f"{type(returned).__name__}; it must return None and raise "
                "an exception if the event could not be recorded"
            )

    # ------------------------------------------------------------- properties

    @property
    def agent_id(self) -> str:
        return self._config.agent_id

    @property
    def config(self) -> AgentConfig:
        """The agent's immutable configuration."""
        return self._config

    @property
    def permission_policy(self) -> PermissionPolicy:
        return self._permission_policy

    @property
    def tool_result_validator(self) -> ResultValidator:
        return self._tool_result_validator

    @property
    def logger(self) -> SecurityLogger:
        return self._logger

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def registered_tools(self) -> frozenset[str]:
        """Names of registered tools (not necessarily allowed)."""
        return self._registry.names

    def __repr__(self) -> str:
        return f"SAW(agent_id={self._config.agent_id!r})"


__all__ = ["SAW"]

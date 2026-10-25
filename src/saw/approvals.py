"""Approval requests, and the desk where an application decides them.

When the final permission decision for a call is "require approval", the
runtime does **not** run the tool. If the agent's ``SAW`` was given an
:class:`ApprovalDesk`, the call is stored on that desk and ``execute_tool``
returns status ``approval_required`` with an :class:`ApprovalRequest`. If no
desk was given, the call is **denied** (fail closed).

The request only describes the call; it cannot run anything. The call runs
only through ``desk.approve(...)``. Approval decisions live on the desk, not
on the ``SAW`` object, so code that holds the agent's ``SAW`` (for example a
tool that makes nested calls) cannot approve anything. Keep the desk in
application code only.

Guarantees:

* **Bound arguments.** An approved call runs with exactly the validated
  arguments in the request. SAW keeps its own deep copy, so changing
  ``request.arguments`` or the caller's original dict has no effect.
* **Single use.** Approving or denying removes the request atomically, so
  even concurrent ``approve`` calls run the tool at most once.
* **Expiry.** A request expires ``approval_ttl_seconds`` after it is made
  (default 15 minutes) and can then no longer be approved.
* **Re-check.** Permission is evaluated again at approval time; if it is
  now denied, the call does not run.
* **No self-approval.** The approver may not be the requesting agent's ID.
* **Optional allowlist.** If the desk has ``approvers``, only those names
  may decide.
* **Bounded.** At most ``max_pending_approvals`` requests wait per agent.
* **In memory only, by design.** Requests are never written to disk, where
  they could be altered. A restart drops them (they never run);
  ``desk.close()`` records each one as discarded first.

Limits: approver names are labels supplied by your application; SAW does
not authenticate people. Decisions are refused while a tool is running in
the same thread, but in-process Python cannot stop deliberately malicious
tool code from reaching the desk object (see the README's security model).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from saw.exceptions import ApprovalError, ConfigurationError
from saw.execution import inside_tool
from saw.registry import RegisteredTool

# Module-level so tests can replace the clock. Monotonic, so changing the
# system clock cannot extend or shorten a request's lifetime.
_monotonic = time.monotonic

_MAX_APPROVER_LENGTH = 128


@dataclass(frozen=True)
class ApprovalRequest:
    """A tool call waiting for a person to approve or deny it.

    Show this to the person deciding. It describes the call; it cannot run it.

    Attributes:
        request_id: Pass to ``ApprovalDesk.approve`` or ``.deny``. Equal to
            the ``call_id`` on the call's security events.
        agent_id: The agent that made the call.
        tool_name: The tool that would run.
        arguments: The validated arguments the tool would receive, as a
            read-only copy. These may contain sensitive values (for example
            an email body), which is why they are not in ``repr`` or in
            security events.
        reason: Why approval is needed.
        created_at: When the request was made (UTC).
        expires_at: When it stops being approvable (UTC).
    """

    request_id: str
    agent_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(repr=False)
    reason: str
    created_at: datetime
    expires_at: datetime


@dataclass
class PendingCall:
    """A stored call. Internal: never handed to application code."""

    request: ApprovalRequest
    tool: RegisteredTool
    kwargs: dict[str, Any]
    deadline: float

    @property
    def expired(self) -> bool:
        return _monotonic() >= self.deadline


def _check_approver_label(approver: object) -> str:
    if (
        not isinstance(approver, str)
        or not approver.strip()
        or len(approver) > _MAX_APPROVER_LENGTH
        or not approver.isprintable()
    ):
        raise ApprovalError(
            "approver must be a non-empty, printable string of at most "
            f"{_MAX_APPROVER_LENGTH} characters"
        )
    return approver


class ApprovalDesk:
    """Where the application approves or denies pending tool calls.

    Create one and pass it to each ``SAW`` whose calls may need approval::

        desk = ApprovalDesk(approvers={"alice", "bob"})
        security = SAW(config=..., approval_desk=desk)
        decision = security.execute_tool("send_email", {...})
        desk.approve(decision.approval_request.request_id, approver="alice")

    One desk may serve several agents; each request records its agent, and
    each agent's ``max_pending_approvals`` applies to its own requests.

    Hold the desk in application code only. Never give it, or anything that
    can reach it, to a tool or to the model.

    Args:
        approvers: Optional allowlist of approver names. If given, only these
            names can approve or deny. Names your application has
            authenticated belong here; SAW does not authenticate people.

    Raises:
        ConfigurationError: If ``approvers`` is not a collection of valid
            names.
    """

    def __init__(self, *, approvers: Iterable[str] | None = None) -> None:
        if approvers is None:
            self._approvers: frozenset[str] | None = None
        else:
            if isinstance(approvers, (str, bytes)) or not isinstance(approvers, Iterable):
                raise ConfigurationError(
                    "approvers must be a collection of names, not a single string"
                )
            names = frozenset(approvers)
            try:
                for name in names:
                    _check_approver_label(name)
            except ApprovalError as exc:
                raise ConfigurationError(f"approvers: {exc}") from None
            if not names:
                raise ConfigurationError("approvers must not be empty; pass None to allow any name")
            self._approvers = names
        self._entries: dict[str, tuple[Any, PendingCall]] = {}
        self._lock = threading.Lock()
        self._closed = False

    # ---------------------------------------------------------- public API

    @property
    def approvers(self) -> frozenset[str] | None:
        """The approver allowlist, or None if any valid name may decide."""
        return self._approvers

    @property
    def closed(self) -> bool:
        return self._closed

    def pending(self, agent_id: str | None = None) -> tuple[ApprovalRequest, ...]:
        """Requests still waiting, oldest first; optionally for one agent.

        Expired requests are removed first, each recorded as an
        ``APPROVAL_EXPIRED`` event in its agent's log.
        """
        self._expire()
        with self._lock:
            requests = [p.request for _, p in self._entries.values()]
        if agent_id is not None:
            requests = [r for r in requests if r.agent_id == agent_id]
        return tuple(sorted(requests, key=lambda r: r.created_at))

    def approve(self, request_id: str, *, approver: str, note: str | None = None):
        """Approve a pending call and run it.

        The call runs with exactly the arguments in the request, after the
        agent's permission check is repeated. Its result goes through every
        result check, as for any call.

        Args:
            request_id: ``outcome.approval_request.request_id``.
            approver: Who approved. Recorded in the security log.
            note: Optional comment recorded in the log (first 200 characters).

        Returns:
            The call's ``ToolCallOutcome``: executed, failed, result rejected,
            or denied (expired, or permission now denied).

        Raises:
            ApprovalError: Unknown or already-decided request, invalid or
                unlisted approver, the requesting agent approving itself, or
                called while a tool is running.
            AuditError: The agent's logger failed. If ``tool_executed`` is
                False the tool did not run. The request is used up either way.
        """
        approver = self._check_decider("approve", approver)
        runtime, pending = self._take(request_id, approver)
        if pending.expired:
            return runtime._record_expiry(pending)
        return runtime._approve_pending(pending, approver, note)

    def deny(self, request_id: str, *, approver: str, reason: str = "denied by the user"):
        """Deny a pending call. The tool never runs.

        Args:
            request_id: ``outcome.approval_request.request_id``.
            approver: Who denied. Recorded in the security log.
            reason: Why. Included in the outcome's reason, which is shown to
                the model (first 500 characters).

        Returns:
            A ``ToolCallOutcome`` with status ``DENIED``.

        Raises:
            ApprovalError: As for :meth:`approve`, or an empty reason.
            AuditError: The agent's logger failed. The tool did not run.
        """
        approver = self._check_decider("deny", approver)
        if not isinstance(reason, str) or not reason.strip():
            raise ApprovalError("reason must be a non-empty string")
        runtime, pending = self._take(request_id, approver)
        if pending.expired:
            return runtime._record_expiry(pending)
        return runtime._deny_pending(pending, approver, reason)

    def close(self) -> None:
        """Discard every pending request and refuse new ones.

        Each discarded request is recorded as ``APPROVAL_EXPIRED`` with
        ``details["discarded"] = True``, so the audit log shows it never ran.
        Afterwards, calls that need approval are denied. Call this when the
        application shuts down. Safe to call more than once.
        """
        with self._lock:
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
        errors = []
        for runtime, pending in entries:
            try:
                runtime._record_expiry(pending, discarded=True)
            except Exception as exc:  # record every one we can, then report
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> ApprovalDesk:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._entries)
        return f"ApprovalDesk(pending={count}, closed={self._closed})"

    # ------------------------------------------------ used by the runtime

    def _submit(self, runtime: Any, pending: PendingCall, limit: int) -> str:
        """Store a call. Returns "ok", "full" or "closed"."""
        with self._lock:
            if self._closed:
                return "closed"
            mine = sum(1 for owner, _ in self._entries.values() if owner is runtime)
            if mine >= limit:
                return "full"
            self._entries[pending.request.request_id] = (runtime, pending)
            return "ok"

    def _withdraw(self, request_id: str) -> None:
        with self._lock:
            self._entries.pop(request_id, None)

    def _expire(self, runtime: Any = None) -> None:
        now = _monotonic()
        with self._lock:
            gone = [
                (owner, p) for owner, p in self._entries.values()
                if now >= p.deadline and (runtime is None or owner is runtime)
            ]
            for _, p in gone:
                del self._entries[p.request.request_id]
        for owner, p in gone:
            owner._record_expiry(p)

    # ------------------------------------------------------------ helpers

    def _check_decider(self, action: str, approver: object) -> str:
        if inside_tool.get():
            raise ApprovalError(
                f"{action}() cannot be called while a tool is running. Approval "
                "decisions must come from the application, never from a tool"
            )
        approver = _check_approver_label(approver)
        if self._approvers is not None and approver not in self._approvers:
            raise ApprovalError(f"{approver!r} is not an allowed approver on this desk")
        return approver

    def _take(self, request_id: object, approver: str) -> tuple[Any, PendingCall]:
        with self._lock:
            entry = self._entries.get(request_id) if isinstance(request_id, str) else None
            if entry is None:
                shown = request_id if isinstance(request_id, str) else type(request_id).__name__
                raise ApprovalError(
                    f"no pending approval request {str(shown)[:64]!r} on this desk; "
                    "it may already have been decided, expired or been discarded"
                )
            if approver == entry[1].request.agent_id:
                raise ApprovalError("an agent cannot decide its own approval request")
            del self._entries[request_id]
        return entry


__all__ = ["ApprovalRequest", "ApprovalDesk", "PendingCall"]

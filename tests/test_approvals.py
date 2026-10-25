"""The approval workflow: pending calls cannot run until a person approves."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

import saw.approvals
from saw import (
    SAW,
    AgentConfig,
    ApprovalDesk,
    ApprovalError,
    ApprovalRequest,
    ApprovalRequiredError,
    AuditError,
    CallStatus,
    EventType,
    PermissionDeniedError,
    PermissionPolicy,
    PolicyDecision,
    SecurityLogger,
)
from conftest import RecordingLogger


@dataclass
class EmailArgs:
    to: str
    body: str
    cc: list[str] | None = None


@dataclass
class EmailResult:
    sent: bool


class Outbox:
    def __init__(self):
        self.sent: list[dict] = []

    def __call__(self, to, body, cc=None):
        self.sent.append({"to": to, "body": body, "cc": cc})
        return {"sent": True}


EMAIL = {"to": "user@example.com", "body": "Meeting update"}


class Assistant:
    """A personal-assistant SAW plus its desk and outbox, for tests."""

    def __init__(self, logger=None, policy=None, desk=None, **config):
        config.setdefault("agent_id", "personal-assistant")
        config.setdefault("allowed_tools", ["send_email", "calendar"])
        config.setdefault("approval_required", ["send_email"])
        self.desk = desk or ApprovalDesk()
        self.security = SAW(
            config=AgentConfig(**config), logger=logger or RecordingLogger(),
            permission_policy=policy, approval_desk=self.desk,
        )
        self.outbox = Outbox()
        self.security.register_tool(
            "send_email", schema=EmailArgs, function=self.outbox, result_schema=EmailResult
        )

    def execute_tool(self, *args, **kwargs):
        return self.security.execute_tool(*args, **kwargs)

    def approve(self, *args, **kwargs):
        return self.desk.approve(*args, **kwargs)

    def deny(self, *args, **kwargs):
        return self.desk.deny(*args, **kwargs)

    def pending_approvals(self):
        return self.desk.pending()

    @property
    def registry(self):
        return self.security.registry

    def register_tool(self, *args, **kwargs):
        return self.security.register_tool(*args, **kwargs)


def assistant(logger=None, policy=None, **config):
    a = Assistant(logger, policy, **config)
    return a, a.outbox


def kinds(logger, call_id):
    return [e.event_type for e in logger.events if e.call_id == call_id]


@pytest.fixture
def clock(monkeypatch):
    """A controllable monotonic clock for expiry tests."""
    now = [1000.0]
    monkeypatch.setattr(saw.approvals, "_monotonic", lambda: now[0])
    return now


# ------------------------------------------------------------------ pending


def test_pending_call_does_not_run_and_cannot_be_read_as_success():
    logger = RecordingLogger()
    security, outbox = assistant(logger)

    decision = security.execute_tool("send_email", EMAIL)

    assert decision.status is CallStatus.APPROVAL_REQUIRED
    assert not decision.ok
    assert outbox.sent == []
    with pytest.raises(ApprovalRequiredError):
        decision.value
    assert "NOT been performed" in decision.for_model()["note"]
    assert kinds(logger, decision.call_id) == [EventType.TOOL_CALL_RECEIVED, EventType.APPROVAL_REQUIRED]


def test_approval_request_describes_the_call():
    security, _ = assistant(approval_ttl_seconds=60)
    request = security.execute_tool("send_email", EMAIL).approval_request

    assert isinstance(request, ApprovalRequest)
    assert request.agent_id == "personal-assistant"
    assert request.tool_name == "send_email"
    assert dict(request.arguments) == {**EMAIL, "cc": None}
    assert "requires approval" in request.reason
    assert (request.expires_at - request.created_at).total_seconds() == 60
    assert "Meeting update" not in repr(request)
    assert not hasattr(request, "approve") and not callable(getattr(request, "execute", None))
    assert security.pending_approvals() == (request,)


def test_request_arguments_are_read_only():
    security, _ = assistant()
    request = security.execute_tool("send_email", EMAIL).approval_request
    with pytest.raises(TypeError):
        request.arguments["to"] = "attacker@evil.test"  # type: ignore[index]


# ------------------------------------------------------------------ approve


def test_approve_runs_the_call_once_with_the_approved_arguments():
    logger = RecordingLogger()
    security, outbox = assistant(logger)
    decision = security.execute_tool("send_email", EMAIL)

    outcome = security.approve(decision.approval_request.request_id, approver="alice", note="ok")

    assert outcome.ok and outcome.value == {"sent": True}
    assert outcome.call_id == decision.call_id
    assert outbox.sent == [{**EMAIL, "cc": None}]
    assert security.pending_approvals() == ()
    assert kinds(logger, decision.call_id) == [
        EventType.TOOL_CALL_RECEIVED,
        EventType.APPROVAL_REQUIRED,
        EventType.APPROVAL_GRANTED,
        EventType.TOOL_CALL_ALLOWED,
        EventType.TOOL_EXECUTED,
    ]
    granted = [e for e in logger.events if e.event_type is EventType.APPROVAL_GRANTED][0]
    assert granted.details == {"approver": "alice", "note": "ok"}
    allowed = [e for e in logger.events if e.event_type is EventType.TOOL_CALL_ALLOWED][0]
    assert allowed.details["approved_by"] == "alice"


def test_approved_arguments_cannot_be_changed_after_the_request():
    security, outbox = assistant()
    arguments = {"to": "user@example.com", "body": "hi", "cc": ["boss@example.com"]}
    request = security.execute_tool("send_email", arguments).approval_request

    arguments["to"] = "attacker@evil.test"
    arguments["cc"].append("attacker@evil.test")
    request.arguments["cc"].append("also-evil@evil.test")  # nested list in the shown copy

    security.approve(request.request_id, approver="alice")
    assert outbox.sent == [{"to": "user@example.com", "body": "hi", "cc": ["boss@example.com"]}]


def test_request_can_only_be_used_once():
    security, outbox = assistant()
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id
    security.approve(request_id, approver="alice")
    with pytest.raises(ApprovalError, match="no pending approval request"):
        security.approve(request_id, approver="alice")
    with pytest.raises(ApprovalError):
        security.deny(request_id, approver="alice")
    assert len(outbox.sent) == 1


def test_concurrent_approvals_run_the_tool_at_most_once():
    security, outbox = assistant()
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id
    barrier = threading.Barrier(16)
    results = []

    def worker():
        barrier.wait()
        try:
            results.append(security.approve(request_id, approver="alice").status)
        except ApprovalError:
            results.append("refused")

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(CallStatus.EXECUTED) == 1
    assert results.count("refused") == 15
    assert len(outbox.sent) == 1


def test_result_checks_still_apply_after_approval():
    security, _ = assistant()
    security.register_tool(
        "bad_email", schema=EmailArgs, function=lambda to, body, cc=None: {"sent": "yes"},
        result_schema=EmailResult,
    )
    desk = ApprovalDesk()
    security2 = SAW(
        config=AgentConfig(agent_id="pa", allowed_tools=["bad_email"], approval_required=["bad_email"]),
        registry=security.registry, logger=RecordingLogger(), approval_desk=desk,
    )
    request_id = security2.execute_tool("bad_email", EMAIL).approval_request.request_id
    assert desk.approve(request_id, approver="alice").status is CallStatus.RESULT_REJECTED


# --------------------------------------------------------------------- deny


def test_deny_prevents_execution():
    logger = RecordingLogger()
    security, outbox = assistant(logger)
    decision = security.execute_tool("send_email", EMAIL)

    outcome = security.deny(decision.approval_request.request_id, approver="alice", reason="not now")

    assert outcome.status is CallStatus.DENIED
    assert outcome.reason == "approval denied by alice: not now"
    assert outbox.sent == []
    with pytest.raises(PermissionDeniedError):
        outcome.value
    assert kinds(logger, decision.call_id)[-1] is EventType.APPROVAL_DENIED
    assert security.pending_approvals() == ()
    with pytest.raises(ApprovalError):
        security.approve(decision.approval_request.request_id, approver="alice")
    assert outbox.sent == []


# ------------------------------------------------------------------- expiry


def test_expired_request_cannot_be_approved(clock):
    logger = RecordingLogger()
    security, outbox = assistant(logger, approval_ttl_seconds=60)
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id

    clock[0] += 60
    outcome = security.approve(request_id, approver="alice")

    assert outcome.status is CallStatus.DENIED
    assert "expired" in outcome.reason
    assert outbox.sent == []
    assert kinds(logger, request_id)[-1] is EventType.APPROVAL_EXPIRED
    with pytest.raises(ApprovalError):
        security.approve(request_id, approver="alice")


def test_request_just_before_expiry_can_be_approved(clock):
    security, outbox = assistant(approval_ttl_seconds=60)
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id
    clock[0] += 59.9
    assert security.approve(request_id, approver="alice").ok


def test_listing_removes_and_records_expired_requests(clock):
    logger = RecordingLogger()
    security, _ = assistant(logger, approval_ttl_seconds=10)
    old = security.execute_tool("send_email", EMAIL).approval_request
    clock[0] += 5
    new = security.execute_tool("send_email", EMAIL).approval_request
    clock[0] += 6

    assert security.pending_approvals() == (new,)
    assert kinds(logger, old.request_id)[-1] is EventType.APPROVAL_EXPIRED


def test_denying_an_expired_request_records_expiry(clock):
    security, _ = assistant(approval_ttl_seconds=1)
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id
    clock[0] += 2
    assert "expired" in security.deny(request_id, approver="alice").reason


# --------------------------------------------------------------- pending cap


def test_pending_requests_are_capped(clock):
    security, _ = assistant(max_pending_approvals=2, approval_ttl_seconds=10)
    first = security.execute_tool("send_email", EMAIL)
    security.execute_tool("send_email", EMAIL)

    third = security.execute_tool("send_email", EMAIL)
    assert third.status is CallStatus.DENIED
    assert "2 approval requests are already pending" in third.reason

    security.deny(first.approval_request.request_id, approver="alice")
    assert security.execute_tool("send_email", EMAIL).status is CallStatus.APPROVAL_REQUIRED

    clock[0] += 11  # everything expires, freeing capacity
    assert security.execute_tool("send_email", EMAIL).status is CallStatus.APPROVAL_REQUIRED


# ------------------------------------------------------------ bad decisions


@pytest.mark.parametrize("request_id", ["nope", "", None, 42])
def test_unknown_request_ids_rejected(request_id):
    security, _ = assistant()
    with pytest.raises(ApprovalError):
        security.approve(request_id, approver="alice")  # type: ignore[arg-type]


def test_requests_belong_to_their_desk():
    first, outbox = assistant(agent_id="assistant-1")
    second, _ = assistant(agent_id="assistant-2")
    request_id = first.execute_tool("send_email", EMAIL).approval_request.request_id
    with pytest.raises(ApprovalError, match="no pending approval request"):
        second.approve(request_id, approver="alice")
    assert outbox.sent == []
    assert len(first.pending_approvals()) == 1


@pytest.mark.parametrize("approver", ["", "   ", None, 7, "x" * 129, "alice\nbob"])
def test_invalid_approver_rejected_and_request_kept(approver):
    security, outbox = assistant()
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id
    with pytest.raises(ApprovalError, match="approver"):
        security.approve(request_id, approver=approver)  # type: ignore[arg-type]
    assert outbox.sent == []
    assert len(security.pending_approvals()) == 1


def test_deny_requires_a_reason():
    security, _ = assistant()
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id
    with pytest.raises(ApprovalError, match="reason"):
        security.deny(request_id, approver="alice", reason="")


def test_tool_cannot_approve_requests():
    @dataclass
    class IdArgs:
        request_id: str

    security, outbox = assistant(allowed_tools=["send_email", "approve_pending"])
    desk = security.desk
    security.register_tool(
        "approve_pending", schema=IdArgs,
        function=lambda request_id: desk.approve(request_id, approver="model").status.value,
    )
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id

    sneaky = security.execute_tool("approve_pending", {"request_id": request_id})

    assert sneaky.status is CallStatus.FAILED
    assert isinstance(sneaky.error, ApprovalError)
    assert outbox.sent == []
    assert len(security.pending_approvals()) == 1


# ---------------------------------------------------------- re-check at approve


class ChangingPolicy(PermissionPolicy):
    def __init__(self):
        self.blocked = False

    def evaluate(self, call, config):
        if self.blocked:
            return PolicyDecision.deny("email is now blocked")
        return PolicyDecision.allow("ok")


def test_permission_is_rechecked_at_approval_time():
    policy = ChangingPolicy()
    security, outbox = assistant(policy=policy)
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id
    policy.blocked = True

    outcome = security.approve(request_id, approver="alice")

    assert outcome.status is CallStatus.DENIED
    assert "permission is now denied: email is now blocked" in outcome.reason
    assert outbox.sent == []


def test_policy_can_require_approval_and_approval_satisfies_it():
    class AskForExternal(PermissionPolicy):
        def evaluate(self, call, config):
            if call.arguments.get("to", "").endswith("@example.com"):
                return PolicyDecision.allow("internal")
            return PolicyDecision.require_approval("external recipient")

    security, outbox = assistant(policy=AskForExternal(), approval_required=[])
    assert security.execute_tool("send_email", EMAIL).ok
    external = security.execute_tool("send_email", {"to": "x@other.org", "body": "hi"})
    assert external.status is CallStatus.APPROVAL_REQUIRED
    assert external.approval_request.reason == "external recipient"
    assert security.approve(external.approval_request.request_id, approver="alice").ok
    assert len(outbox.sent) == 2


# -------------------------------------------------------------- audit failures


class FailOn(SecurityLogger):
    def __init__(self, event_type):
        self.event_type = event_type
        self.active = True

    def log(self, event):
        if self.active and event.event_type is self.event_type:
            raise OSError("disk full")


def test_unrecorded_request_is_not_left_pending():
    security, outbox = assistant(FailOn(EventType.APPROVAL_REQUIRED))
    with pytest.raises(AuditError):
        security.execute_tool("send_email", EMAIL)
    assert security.pending_approvals() == ()


def test_unrecorded_approval_does_not_run_the_tool():
    logger = FailOn(EventType.APPROVAL_GRANTED)
    security, outbox = assistant(logger)
    request_id = security.execute_tool("send_email", EMAIL).approval_request.request_id
    with pytest.raises(AuditError) as info:
        security.approve(request_id, approver="alice")
    assert info.value.tool_executed is False
    assert outbox.sent == []
    logger.active = False
    with pytest.raises(ApprovalError):  # used up: must be requested again
        security.approve(request_id, approver="alice")

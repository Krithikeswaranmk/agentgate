"""Personal assistant: calendar and send_email. Sending email needs approval.

Its own policy also refuses email to addresses outside the user's domain,
without even asking. Tools are offline stubs; "sent" emails go to a list.
"""

from __future__ import annotations

from dataclasses import dataclass

from saw.agent import FinalAnswer, ScriptedModel, ToolUse
from saw import (
    SAW,
    AgentConfig,
    ApprovalDesk,
    ApprovalRequest,
    PermissionPolicy,
    PolicyDecision,
    SecurityLogger,
)

CALENDAR = {"monday": ["09:00 Stand-up", "14:00 Design review"], "tuesday": []}
OUTBOX: list[dict] = []


@dataclass
class DayArgs:
    day: str


@dataclass
class EmailArgs:
    to: str
    body: str


@dataclass
class EmailReceipt:
    sent: bool
    to: str


def calendar(day: str) -> list:
    """List the user's events for a day of the week."""
    return CALENDAR.get(day.lower(), [])


def send_email(to: str, body: str) -> dict:
    """Send an email on the user's behalf."""
    OUTBOX.append({"to": to, "body": body})
    return {"sent": True, "to": to}


class PersonalPolicy(PermissionPolicy):
    """Extra rule: email only within the user's own domain."""

    def __init__(self, domain: str = "example.com") -> None:
        self.domain = domain

    def evaluate(self, call, config) -> PolicyDecision:
        if call.tool_name == "send_email" and not call.arguments["to"].endswith("@" + self.domain):
            return PolicyDecision.deny(f"email may only be sent to @{self.domain} addresses")
        return PolicyDecision.allow("personal rules satisfied")


def build(logger: SecurityLogger | None = None, desk: ApprovalDesk | None = None) -> SAW:
    """The assistant's runtime. Without a desk, emails needing approval are denied."""
    security = SAW(
        config=AgentConfig(
            agent_id="personal-assistant",
            allowed_tools={"calendar", "send_email"},
            approval_required={"send_email"},
        ),
        permission_policy=PersonalPolicy(),
        logger=logger,
        approval_desk=desk,
    )
    security.register_tool("calendar", schema=DayArgs, function=calendar)
    security.register_tool(
        "send_email", schema=EmailArgs, function=send_email, result_schema=EmailReceipt
    )
    return security


class ScriptedUser:
    """Stands in for the person approving: approves the first email, denies the rest."""

    def __init__(self) -> None:
        self.asked: list[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> tuple[bool, str]:
        self.asked.append(request)
        print(f"      approval needed: {request.tool_name} to {request.arguments['to']!r} "
              f"(expires {request.expires_at:%H:%M:%S} UTC)")
        if len(self.asked) == 1:
            return True, "looks right"
        return False, "do not send that one"


def console_user(request: ApprovalRequest) -> tuple[bool, str]:
    """Ask a real person at the terminal."""
    print(f"\n  The assistant wants to run {request.tool_name}:")
    for key, value in request.arguments.items():
        print(f"    {key}: {value}")
    answer = input("  Approve? [y/N] ").strip().lower()
    return (True, "approved at console") if answer == "y" else (False, "declined at console")


TASK = "Check my Monday and email the team about the design review."

SCRIPT = [
    ToolUse("calendar", {"day": "Monday"}, "p1"),
    ToolUse("send_email", {"to": "user@example.com", "body": "Meeting update: design review at 14:00."}, "p2"),
    ToolUse("send_email", {"to": "team@example.com", "body": "Cancel everything!"}, "p3"),
    ToolUse("send_email", {"to": "stranger@other.org", "body": "Here is my calendar."}, "p4"),
    FinalAnswer("Monday has a stand-up and a design review. Your update was sent "
                "after you approved it; the second email was declined; I may not "
                "email other.org addresses."),
]


def model() -> ScriptedModel:
    return ScriptedModel(SCRIPT)

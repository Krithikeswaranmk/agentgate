"""Run the three demo agents, each with its own SAW configuration and policy.

    python examples/agents/run_demos.py               # scripted user approves/denies
    python examples/agents/run_demos.py --interactive # you approve emails yourself

All three use the same SAW package; nothing is shared between them except
the code. Everything runs offline.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter

import coding_agent
import personal_assistant
import research_agent
from saw import ApprovalDesk, CallStatus, SecurityLogger
from saw.agent import describe_turn, run_agent


class EventCounter(SecurityLogger):
    """Counts security events per type (each agent gets its own)."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def log(self, event) -> None:
        self.counts[event.event_type.value] += 1


def run(name: str, module, approval_handler=None, desk=None):
    print(f"\n=== {name} ===")
    print(f"  Task: {module.TASK}")
    events = EventCounter()
    security = module.build(logger=events, desk=desk) if desk else module.build(logger=events)
    print(f"  Allowed tools: {', '.join(sorted(security.config.allowed_tools))}"
          + (f"; needs approval: {', '.join(sorted(security.config.approval_required))}"
             if security.config.approval_required else ""))
    result = run_agent(
        security, module.model(), module.TASK,
        approval_desk=desk, approval_handler=approval_handler,
        on_turn=lambda turn: print("    " + describe_turn(turn)),
    )
    print(f"  Answer: {result.answer}")
    statuses = Counter(t.outcome.status.value for t in result.turns)
    print(f"  Outcomes: {dict(statuses)}; security events: {sum(events.counts.values())}")
    return security, result


def main(argv: list[str]) -> int:
    logging.getLogger("saw.security").addHandler(logging.NullHandler())
    interactive = "--interactive" in argv

    run("Research agent", research_agent)
    run("Coding agent", coding_agent)
    user = personal_assistant.console_user if interactive else personal_assistant.ScriptedUser()
    with ApprovalDesk() as desk:  # closing it discards anything left pending
        security, result = run("Personal assistant", personal_assistant, approval_handler=user, desk=desk)
        print("\n  Emails actually sent:", personal_assistant.OUTBOX or "none")
        print(f"  Approval requests still pending: {len(desk.pending())}")

    executed_without_approval = [
        t for t in result.turns
        if t.tool_use.name == "send_email" and t.outcome.status is CallStatus.EXECUTED and t.approval != "approved"
    ]
    assert not executed_without_approval, "an email ran without approval"
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

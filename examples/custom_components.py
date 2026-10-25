"""Three developers, three custom components, one unchanged security core.

Run from the repository root after `pip install -e .`:

    python examples/custom_components.py

* Research developer: custom logger (where and how events are recorded).
* Coding developer: custom result validator (what results are acceptable).
* Personal-agent developer: custom permission policy (extra authorisation).

Each replaces one component. Everything else - configuration limits,
argument and result schemas, the result size limit and audit logging -
keeps working unchanged. The tools are stubs; nothing touches the network
or your files except one log file in a temporary folder.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from saw import (
    SAW,
    AgentConfig,
    ApprovalDesk,
    PermissionPolicy,
    PolicyDecision,
    ResultValidator,
    SecurityLogger,
    ValidationResult,
)


def show(label: str, outcome) -> None:
    detail = "" if outcome.ok else f" - {outcome.reason}"
    print(f"  {label:<48} {outcome.status.value}{detail}")


# ---------------------------------------------------------------------------
# 1. Research developer: custom logging
# ---------------------------------------------------------------------------


class ResearchLogger(SecurityLogger):
    """Appends each event as one JSON line to a file.

    Raising on failure is part of the contract: if the file cannot be
    written, SAW stops the call instead of running it unaudited.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, event) -> None:
        with self.path.open("a", encoding="utf-8") as f:  # raises on failure
            f.write(json.dumps(event.to_dict(), default=str) + "\n")


@dataclass
class SearchArgs:
    query: str


def web_search(query: str) -> list:
    return [{"title": f"Result for {query}", "url": "https://example.org"}]


def research_demo(log_path: Path) -> None:
    print("\n1. Research agent with ResearchLogger")
    security = SAW(
        config=AgentConfig(agent_id="research-agent", allowed_tools={"web_search"}),
        logger=ResearchLogger(log_path),
    )
    security.register_tool("web_search", schema=SearchArgs, function=web_search)
    security.register_tool("delete_file", schema=SearchArgs, function=lambda query: None)

    show("web_search (allowed)", security.execute_tool("web_search", {"query": "SAW"}))
    show("delete_file (not in allowed_tools)", security.execute_tool("delete_file", {"query": "x"}))
    show("web_search with a bad argument", security.execute_tool("web_search", {"query": 42}))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(line)["event_type"] for line in lines]
    print(f"  {len(lines)} events written to {log_path.name}: {', '.join(kinds)}")


# ---------------------------------------------------------------------------
# 2. Coding developer: custom result validation
# ---------------------------------------------------------------------------


class CodingValidator(ResultValidator):
    """One coding-specific rule on top of SAW's mandatory result checks.

    SAW has already checked that the result is JSON-compatible and within
    the size limits before this runs, and checks the tool's result schema
    afterwards, so the validator only needs its own rule. It receives a copy
    of the result. The marker scan is illustrative only, not real secret
    detection.
    """

    BLOCKED_MARKERS = ("-----BEGIN PRIVATE KEY-----", "AWS_SECRET_ACCESS_KEY")

    def validate(self, call, result, config) -> ValidationResult:
        if any(marker in text for text in _strings(result) for marker in self.BLOCKED_MARKERS):
            # The reason must not repeat the result's content.
            return ValidationResult.reject("result appears to contain a credential")
        return ValidationResult.accept()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


@dataclass
class RunTestsArgs:
    path: str = "tests"


@dataclass
class RunTestsResult:
    passed: bool
    failures: int

    def __post_init__(self):
        if self.failures < 0:
            raise ValueError("failures must be >= 0")


@dataclass
class ReadFileArgs:
    path: str


FILES = {
    "README.md": "# Project\nHow to build it.",
    ".env": "AWS_SECRET_ACCESS_KEY=abc123",
    "big.log": "x" * 200_000,
}


def coding_demo() -> None:
    print("\n2. Coding agent with CodingValidator")
    security = SAW(
        config=AgentConfig(agent_id="coding-agent", allowed_tools={"run_tests", "read_file"}),
        tool_result_validator=CodingValidator(),
    )
    outputs = iter([{"passed": True, "failures": 0}, {"passed": "yes", "failures": -1}])
    security.register_tool(
        "run_tests", schema=RunTestsArgs, function=lambda path: next(outputs),
        result_schema=RunTestsResult,
    )
    security.register_tool("read_file", schema=ReadFileArgs, function=lambda path: FILES[path])

    show("run_tests -> valid result", security.execute_tool("run_tests", {}))
    show("run_tests -> {'passed': 'yes', 'failures': -1}", security.execute_tool("run_tests", {}))
    show("read_file README.md", security.execute_tool("read_file", {"path": "README.md"}))
    show("read_file .env (custom rule)", security.execute_tool("read_file", {"path": ".env"}))
    show("read_file big.log (size limit still applies)", security.execute_tool("read_file", {"path": "big.log"}))


# ---------------------------------------------------------------------------
# 3. Personal-agent developer: custom permissions
# ---------------------------------------------------------------------------


class PersonalPolicy(PermissionPolicy):
    """Email may only go to the user's own domain.

    The policy only expresses this extra rule. SAW still applies the
    configuration: tools outside allowed_tools are denied and send_email
    still needs approval, whatever this policy returns.
    """

    def __init__(self, own_domain: str) -> None:
        self.own_domain = own_domain

    def evaluate(self, call, config) -> PolicyDecision:
        if call.tool_name == "send_email":
            recipient = call.arguments["to"]  # already validated as a string
            if not recipient.endswith("@" + self.own_domain):
                return PolicyDecision.deny("email may only be sent within the user's domain")
        return PolicyDecision.allow("no personal rule applies")


@dataclass
class EmailArgs:
    to: str
    body: str


@dataclass
class DayArgs:
    day: str


def personal_demo() -> None:
    print("\n3. Personal agent with PersonalPolicy")
    desk = ApprovalDesk()  # kept by the application; the agent never sees it
    security = SAW(
        config=AgentConfig(
            agent_id="personal-agent",
            allowed_tools={"read_calendar", "send_email"},
            approval_required={"send_email"},
        ),
        permission_policy=PersonalPolicy(own_domain="example.com"),
        approval_desk=desk,
    )
    security.register_tool("read_calendar", schema=DayArgs, function=lambda day: ["09:00 stand-up"])
    security.register_tool("send_email", schema=EmailArgs, function=lambda to, body: "sent")
    security.register_tool("delete_account", schema=DayArgs, function=lambda day: "deleted")

    show("read_calendar", security.execute_tool("read_calendar", {"day": "Monday"}))
    show("send_email to outsider (policy denies)", security.execute_tool("send_email", {"to": "x@other.org", "body": "hi"}))
    show("send_email to colleague (still needs approval)", security.execute_tool("send_email", {"to": "ana@example.com", "body": "hi"}))
    show("delete_account (not in allowed_tools)", security.execute_tool("delete_account", {"day": "now"}))
    desk.close()  # discard anything still pending


def main() -> None:
    # Agents 2 and 3 keep SAW's default logger, which writes to the standard
    # "saw.security" logger. Silence it here so the demo output stays readable;
    # a real application would configure a handler instead.
    logging.getLogger("saw.security").addHandler(logging.NullHandler())

    with tempfile.TemporaryDirectory() as folder:
        research_demo(Path(folder) / "research-events.jsonl")
    coding_demo()
    personal_demo()


if __name__ == "__main__":
    main()

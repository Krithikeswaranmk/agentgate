"""Coding agent: read_file and run_tests. Only registered tools can execute.

Its own policy keeps file reads inside the project's src/ and tests/
folders, and run_tests results must match a result schema. Tools are
offline stubs.
"""

from __future__ import annotations

from dataclasses import dataclass

from saw.agent import FinalAnswer, ScriptedModel, ToolUse
from saw import SAW, AgentConfig, PermissionPolicy, PolicyDecision, SecurityLogger

REPOSITORY = {
    "src/app.py": "def add(a, b):\n    return a + b\n",
    "tests/test_app.py": "from app import add\n\ndef test_add():\n    assert add(2, 2) == 4\n",
    ".env": "API_KEY=not-for-the-agent",
}


@dataclass
class FileArgs:
    path: str


@dataclass
class TestArgs:
    path: str = "tests"


@dataclass
class TestReport:
    passed: bool
    failures: int

    def __post_init__(self):
        if self.failures < 0:
            raise ValueError("failures must be >= 0")


def read_file(path: str) -> str:
    """Read a file from the project repository."""
    return REPOSITORY[path]


class FakeTestRunner:
    """Returns a good report, then a malformed one (as a buggy runner might)."""

    def __init__(self) -> None:
        self.reports = [{"passed": True, "failures": 0}, {"passed": "yes", "failures": -1}]

    def __call__(self, path: str = "tests") -> dict:
        """Run the project's tests and report whether they passed."""
        return self.reports.pop(0) if self.reports else {"passed": True, "failures": 0}


class CodingPolicy(PermissionPolicy):
    """Extra rule: only files under src/ or tests/ may be read."""

    def evaluate(self, call, config) -> PolicyDecision:
        if call.tool_name == "read_file":
            path = call.arguments["path"]
            if not path.startswith(("src/", "tests/")) or ".." in path:
                return PolicyDecision.deny("the coding agent may only read files in src/ or tests/")
        return PolicyDecision.allow("coding rules satisfied")


def build(logger: SecurityLogger | None = None) -> SAW:
    security = SAW(
        config=AgentConfig(agent_id="coding-agent", allowed_tools={"read_file", "run_tests"}),
        permission_policy=CodingPolicy(),
        logger=logger,
    )
    security.register_tool("read_file", schema=FileArgs, function=read_file)
    security.register_tool(
        "run_tests", schema=TestArgs, function=FakeTestRunner(), result_schema=TestReport
    )
    # Deliberately no "shell" or "delete_file" tool: unregistered tools cannot run.
    return security


TASK = "Check that the add function is tested and the tests pass."

SCRIPT = [
    ToolUse("read_file", {"path": "src/app.py"}, "c1"),
    ToolUse("run_tests", {"path": "tests"}, "c2"),
    ToolUse("shell", {"command": "rm -rf /"}, "c3"),
    ToolUse("read_file", {"path": ".env"}, "c4"),
    ToolUse("read_file", {"path": 42}, "c5"),
    ToolUse("run_tests", {}, "c6"),
    FinalAnswer("add() is tested and the first test run passed. The second run "
                "returned an invalid report, which was rejected."),
]


def model() -> ScriptedModel:
    return ScriptedModel(SCRIPT)

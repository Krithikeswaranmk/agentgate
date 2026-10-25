"""Research agent: web_search and read_document. Email sending is denied.

Its own policy also keeps document reads inside the ``docs/`` folder.
Tools are offline stubs.
"""

from __future__ import annotations

from dataclasses import dataclass

from saw.agent import FinalAnswer, ScriptedModel, ToolUse
from saw import SAW, AgentConfig, PermissionPolicy, PolicyDecision, SecurityLogger

DOCUMENTS = {
    "docs/saw-overview.md": "SAW checks tool calls before they run and results before they return.",
    "docs/limits.md": "SAW does not sandbox tools and does not limit execution time.",
    "private/salaries.csv": "name,salary\nana,100\n",
}


@dataclass
class SearchArgs:
    query: str
    max_results: int = 3


@dataclass
class DocumentArgs:
    path: str


@dataclass
class EmailArgs:
    to: str
    body: str


def web_search(query: str, max_results: int = 3) -> list:
    """Search the web and return result titles and URLs."""
    return [
        {"title": f"{query} - result {i}", "url": f"https://example.org/{i}"}
        for i in range(1, max_results + 1)
    ]


def read_document(path: str) -> str:
    """Read a document from the shared research folder."""
    return DOCUMENTS[path]


def send_email(to: str, body: str) -> str:  # registered, but never allowed
    raise AssertionError("the research agent must never send email")


class ResearchPolicy(PermissionPolicy):
    """Extra rule: documents may only be read from docs/."""

    def evaluate(self, call, config) -> PolicyDecision:
        if call.tool_name == "read_document":
            path = call.arguments["path"]
            if not path.startswith("docs/") or ".." in path:
                return PolicyDecision.deny("the research agent may only read documents in docs/")
        return PolicyDecision.allow("research rules satisfied")


def build(logger: SecurityLogger | None = None) -> SAW:
    security = SAW(
        config=AgentConfig(
            agent_id="research-agent",
            allowed_tools={"web_search", "read_document"},
        ),
        permission_policy=ResearchPolicy(),
        logger=logger,
    )
    security.register_tool("web_search", schema=SearchArgs, function=web_search)
    security.register_tool("read_document", schema=DocumentArgs, function=read_document)
    security.register_tool("send_email", schema=EmailArgs, function=send_email)
    return security


TASK = "Research how SAW works and email a summary to my manager."

SCRIPT = [
    ToolUse("web_search", {"query": "SAW agent security"}, "r1"),
    ToolUse("read_document", {"path": "docs/saw-overview.md"}, "r2"),
    ToolUse("read_document", {"path": "private/salaries.csv"}, "r3"),
    ToolUse("read_document", {"path": "docs/../private/salaries.csv"}, "r4"),
    ToolUse("send_email", {"to": "manager@example.com", "body": "Summary..."}, "r5"),
    FinalAnswer("SAW checks tool calls and results. I could not send the email: "
                "this agent is not allowed to send email."),
]


def model() -> ScriptedModel:
    return ScriptedModel(SCRIPT)

# Getting started

This guide takes you from installation to an agent whose tool calls are checked by SAW: registering tools, setting permissions, requiring approval, and plugging in your own logger, validator and policy. You don't need to read SAW's source code.

## 1. Install

SAW needs Python 3.10 or newer and has no required dependencies.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install saw_agent_security-0.1.0-py3-none-any.whl
```

Install from the wheel you were given, or from a source checkout with `pip install .`. Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `[pydantic]` | Pydantic v2 | Schemas written as Pydantic models |
| `[anthropic]` | Anthropic SDK | `saw.adapters.anthropic` |
| `[openai]` | OpenAI SDK | `saw.adapters.openai` |
| `[dev]` | pytest, Pydantic | Running the test suite |

Check it works:

```powershell
python -c "import saw; print(saw.__version__)"
```

## 2. Your first protected tool

SAW sits between your agent and its tools. You **register** each tool with a schema for its arguments, **allow** it in the agent's configuration, and then make every call through `execute_tool`.

```python
from dataclasses import dataclass
from saw import SAW, AgentConfig

@dataclass
class ReadDocumentArgs:
    path: str

def read_document(path: str) -> str:
    """Read a document from the research folder."""
    with open(path, encoding="utf-8") as f:
        return f.read()

security = SAW(config=AgentConfig(
    agent_id="research-agent",
    allowed_tools={"read_document"},
))
security.register_tool(name="read_document", schema=ReadDocumentArgs, function=read_document)

outcome = security.execute_tool("read_document", {"path": "report.txt"})
if outcome.ok:
    print(outcome.value)
else:
    print(outcome.status.value, "-", outcome.reason)
```

Things to know:

* **Deny by default.** An agent can call nothing that isn't in `allowed_tools`. Registering a tool does not allow it.
* **Arguments are checked strictly.** `{"path": 3}` gives status `invalid_arguments`; the function never runs.
* **Blocked calls don't raise.** Check `outcome.ok` or `outcome.status`. Reading `outcome.value` on a blocked call raises the matching exception (for example `PermissionDeniedError`), so you can't mistake a blocked call for a result.
* **Results are checked too.** They must be JSON-compatible and at most 100,000 characters by default.

## 3. Check the result's structure

Add a `result_schema` and SAW rejects results that don't match it:

```python
@dataclass
class RunTestsArgs:
    path: str = "tests"

@dataclass
class TestReport:
    passed: bool
    failures: int

    def __post_init__(self):
        if self.failures < 0:
            raise ValueError("failures must be >= 0")

def run_tests(path: str = "tests") -> dict:
    """Run the project's tests."""
    ...  # e.g. run pytest and summarise
    return {"passed": True, "failures": 0}

security.register_tool("run_tests", schema=RunTestsArgs, function=run_tests,
                       result_schema=TestReport)
```

If `run_tests` returns `{"passed": "yes", "failures": -1}`, the outcome is `result_rejected` and the result is withheld.

## 4. Require approval

List sensitive tools in `approval_required`, and give the agent an `ApprovalDesk`. Your application keeps the desk; the agent and its tools never see it.

```python
from saw import ApprovalDesk, CallStatus

desk = ApprovalDesk(approvers={"alice"})          # optional allowlist of people
assistant = SAW(
    config=AgentConfig(
        agent_id="personal-assistant",
        allowed_tools={"calendar", "send_email"},
        approval_required={"send_email"},
    ),
    approval_desk=desk,
)
# ... register calendar and send_email ...

decision = assistant.execute_tool("send_email", {"to": "user@example.com", "body": "Meeting update"})
if decision.status is CallStatus.APPROVAL_REQUIRED:
    request = decision.approval_request             # what would run, and why it needs approval
    # Show request.tool_name and request.arguments to Alice, then:
    outcome = desk.approve(request.request_id, approver="alice")
    # or:    desk.deny(request.request_id, approver="alice", reason="not now")
```

* Nothing runs until `desk.approve(...)`. The call then runs with exactly the arguments Alice saw.
* A request can be decided once, expires after 15 minutes by default, and cannot be approved by the agent itself.
* **Without a desk, calls that need approval are denied.**
* Call `desk.close()` when your application shuts down; pending requests are recorded as discarded.

## 5. Replace a component

Each of the three components can be replaced on its own. Subclass the interface, implement one method, and pass an instance. Everything you don't replace keeps its built-in default, and SAW's mandatory checks keep running.

### Log events your way

```python
import json
from saw import SecurityLogger

class FileLogger(SecurityLogger):
    def __init__(self, path):
        self.path = path

    def log(self, event):
        with open(self.path, "a", encoding="utf-8") as f:   # raise on failure
            f.write(json.dumps(event.to_dict()) + "\n")

security = SAW(agent_id="research-agent", logger=FileLogger("events.jsonl"))
```

`log` must return `None` and **raise** if it can't record the event. If it raises, SAW stops the call rather than run a tool without an audit record.

### Add rules for results

```python
from saw import ResultValidator, ValidationResult

class NoCredentials(ResultValidator):
    def validate(self, call, result, config):
        if "PRIVATE KEY" in str(result):
            return ValidationResult.reject("result appears to contain a credential")
        return ValidationResult.accept()

security = SAW(agent_id="coding-agent", tool_result_validator=NoCredentials())
```

The validator receives a copy of a result that has already passed SAW's shape and size checks. Don't repeat result content in the reason.

### Add authorisation rules

```python
from saw import PermissionPolicy, PolicyDecision

class OwnDomainOnly(PermissionPolicy):
    def evaluate(self, call, config):
        if call.tool_name == "send_email" and not call.arguments["to"].endswith("@example.com"):
            return PolicyDecision.deny("email may only be sent within the company")
        return PolicyDecision.allow("no extra rule applies")

security = SAW(config=..., permission_policy=OwnDomainOnly())
```

A policy only needs its **extra** rules. It can deny, or require approval for, calls the configuration allows. It can never allow a tool that isn't in `allowed_tools`, or skip a required approval.

Mistakes such as passing the class instead of an instance, a missing method parameter, or an `async` method are reported when `SAW(...)` is created, naming the class and method.

## 6. Connect a model

`saw.agent.run_agent` runs a model until it answers, sending every tool call through SAW and waiting for your approval handler when needed:

```python
import anthropic
from saw.agent import run_agent
from saw.adapters.anthropic import AnthropicModel

def ask_user(request):
    answer = input(f"Allow {request.tool_name} {dict(request.arguments)}? [y/N] ")
    return answer.strip().lower() == "y", "answered at the console"

model = AnthropicModel(anthropic.Anthropic(), model="<model name>")
run = run_agent(assistant, model, "Email the team about the design review",
                approval_desk=desk, approval_handler=ask_user, approver="alice")
print(run.answer)
```

See [LLM integration](llm-integration.md) for OpenAI and OpenAI-compatible services, and for writing your own loop.

## 7. Try the examples

The examples run offline, with no API keys:

```powershell
python examples/agents/run_demos.py                # research, coding and personal-assistant agents
python examples/agents/run_demos.py --interactive  # you approve or deny the emails
python examples/custom_components.py              # replacing each component
```

## Next

* [Reference](reference.md): every public class, option, status, event and exception.
* [Security model](security-model.md): what SAW protects, what it doesn't, and how to deploy it safely. **Read this before relying on SAW.**
* [LLM integration](llm-integration.md): the agent loop, adapters and live tests.

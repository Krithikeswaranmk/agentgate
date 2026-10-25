# SAW: security for LLM agent tool calls

SAW is a Python package that puts a security checkpoint between an LLM agent and its tools. It runs in the same process as the agent:

* **Before a tool runs**, SAW checks that it is registered, allowed for this agent, called with valid arguments, and approved by a person if it needs to be.
* **Before a result reaches the model**, SAW checks its structure and size, and any schema you define.
* **Every decision is recorded** as a security event, and nothing runs without an audit record.

It works with any agent framework. It includes a small agent loop and adapters for the Anthropic and OpenAI APIs, and for services compatible with OpenAI's API.

> **Version 0.1.0 (alpha).** Tested extensively but not independently reviewed. SAW is **not a sandbox** and does **not limit how long tools run**. Read the [security model](docs/security-model.md) before relying on it.

## Install

Requires Python 3.10+. No required dependencies.

```powershell
pip install saw_agent_security-0.1.0-py3-none-any.whl     # or, from a checkout: pip install .
```

Optional extras: `[pydantic]` for Pydantic schemas, `[anthropic]` and `[openai]` for the adapters, `[dev]` for running the tests.

## In 30 seconds

```python
from dataclasses import dataclass
from saw import SAW, AgentConfig, ApprovalDesk, CallStatus

@dataclass
class EmailArgs:
    to: str
    body: str

def send_email(to: str, body: str) -> str:
    """Send an email on the user's behalf."""
    return "sent"

desk = ApprovalDesk(approvers={"alice"})            # kept by your application
security = SAW(
    config=AgentConfig(
        agent_id="personal-assistant",
        allowed_tools={"send_email"},
        approval_required={"send_email"},
    ),
    approval_desk=desk,
)
security.register_tool("send_email", schema=EmailArgs, function=send_email)

decision = security.execute_tool("send_email", {"to": "user@example.com", "body": "Meeting update"})
assert decision.status is CallStatus.APPROVAL_REQUIRED   # nothing has run yet
outcome = desk.approve(decision.approval_request.request_id, approver="alice")
print(outcome.status.value)                              # executed
```

## Try the demos (offline, no API keys)

```powershell
python examples/agents/run_demos.py                # three agents: research, coding, personal assistant
python examples/agents/run_demos.py --interactive  # you approve or deny the emails
python examples/custom_components.py              # replacing the logger, validator and policy
```

| Agent | Allowed tools | Shown being blocked |
|---|---|---|
| Research | `web_search`, `read_document` | `send_email` denied; reads outside `docs/` denied by its policy |
| Coding | `read_file`, `run_tests` | unregistered `shell` cannot run; `.env` denied; malformed test report rejected |
| Personal assistant | `calendar`, `send_email` (needs approval) | one email approved and sent, one denied by the user, one to an outside domain refused |

## Documentation

* **[Getting started](docs/getting-started.md):** install, register tools, set permissions, require approval, plug in your own logger, validator or policy, connect a model.
* **[Reference](docs/reference.md):** every public class, option, status, event and exception.
* **[Security model](docs/security-model.md):** what SAW guarantees, what it assumes, where its protection ends, and a deployment checklist.
* **[LLM integration](docs/llm-integration.md):** the agent loop, provider adapters, live tests, and how to write your own loop safely.
* **[Changelog](CHANGELOG.md)**

## Key limits

* **Only calls through SAW are protected.** Tool functions called directly bypass it.
* **In-process, not a sandbox.** Code in the same process can bypass SAW deliberately; run untrusted code in a separate process.
* **No execution time limit.** Use timeouts in your tools' libraries; SAW records each call's duration.
* **Approvals are in memory.** A restart drops pending requests, which never run.
* **Approver names are labels.** Authenticate people in your application and use the desk's `approvers` allowlist.
* **Real provider APIs have not been called in testing.** The adapters are tested against the real SDKs with mocked HTTP; live tests are included for you to run with your keys.

## Development

```powershell
pip install -e ".[dev]"
pytest                    # live provider tests are skipped unless configured
pytest -m live            # see docs/llm-integration.md
python -m build           # sdist and wheel in dist/
```

CI (`.github/workflows/ci.yml`) runs the tests on Linux, Windows and macOS with Python 3.10–3.13, without Pydantic, against the provider SDKs, and builds the package and tests the installed wheel.

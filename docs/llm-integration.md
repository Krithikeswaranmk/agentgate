# LLM integration

SAW ships a small agent loop (`saw.agent`) and adapters for two provider APIs (`saw.adapters`). The loop is in the package so that the easy path is the safe one; you can also write your own, following the checklist at the end.

## The agent loop

```python
from saw.agent import run_agent

run = run_agent(
    security,                 # the agent's SAW
    model,                    # anything with next_step(messages, tools)
    "Check my Monday and email the team",
    approval_desk=desk,       # only needed with approval_handler
    approval_handler=ask_user,  # (ApprovalRequest) -> (approve: bool, reason: str)
    approver="alice",         # the authenticated name recorded for decisions
    max_steps=20,             # model turns
    max_tool_calls=50,        # tool calls in the whole run
    on_turn=print_progress,   # optional, called after each tool call
)
print(run.answer)
for turn in run.turns:
    print(turn.tool_use.name, turn.outcome.status if turn.outcome else "not attempted")
```

What the loop guarantees:

* Every tool call goes through `security.execute_tool`. The loop never calls a tool function.
* The model is only told about tools the agent is allowed to use (`security.tool_definitions()`).
* The model only sees `outcome.for_model()`. Calls waiting for approval are reported as **not performed**.
* When a call needs approval, the loop waits for `approval_handler`, then decides through the desk. The handler must return exactly `True` to approve; anything else denies. If the handler raises, the request stays pending and nothing runs.
* Every tool use the model asks for gets a result, including uses skipped after `max_tool_calls`, so provider rules about matching tool results always hold.

`Run` has `turns` (each a `Turn` with `tool_use`, `outcome`, `approval`, `skipped`), `answer`, `messages`, and `stopped_early`. `describe_turn(turn)` gives a one-line summary.

## Models

A model is any object with:

```python
def next_step(self, messages: list[dict], tools: list[dict]) -> ModelTurn | FinalAnswer: ...
```

* `ModelTurn(tool_uses=(ToolUse(name, arguments, id), ...), text="")` asks for tools.
* `FinalAnswer(text)` ends the run.

`ScriptedModel([...])` replays fixed steps, for tests and offline demos.

## Anthropic

```powershell
pip install "saw-agent-security[anthropic]"
```

```python
import anthropic
from saw.adapters.anthropic import AnthropicModel

model = AnthropicModel(
    anthropic.Anthropic(),        # reads ANTHROPIC_API_KEY
    model="<model name>",
    max_tokens=1024,
    system="You are a careful assistant.",
)
```

Calls `client.messages.create(...)`. `tool_use` blocks become tool calls; results go back as `tool_result` blocks with `is_error` set for anything not executed. Other block types (for example thinking or server-tool blocks) are ignored, because they are not tools for SAW to run.

## OpenAI and OpenAI-compatible services

```powershell
pip install "saw-agent-security[openai]"
```

```python
import openai
from saw.adapters.openai import OpenAIChatModel

model = OpenAIChatModel(openai.OpenAI(), model="<model name>")   # reads OPENAI_API_KEY

# Services implementing the Chat Completions API, e.g. a local Ollama server:
local = OpenAIChatModel(
    openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    model="<local model>",
)
```

Calls `client.chat.completions.create(...)`. Function `tool_calls` become tool calls; their JSON arguments are parsed, and **if the JSON is invalid, the raw string is passed to SAW, which reports `invalid_arguments` and runs nothing.**

## What has been verified

| Check | Status |
|---|---|
| Message and tool conversion for both adapters | Unit tests |
| Full runs through the **real** Anthropic SDK (2.3.0) and OpenAI SDK (3.37.0), with mocked HTTP: requests sent, responses parsed, approvals, malformed arguments | Tested offline (`tests/test_adapters.py`) |
| OpenAI-compatible endpoint through `base_url` | Tested offline |
| Calls to real provider APIs | **Not yet run.** The live tests below need API keys |

## Live tests

`tests/live/test_live_providers.py` runs a personal-assistant scenario against real models. It includes a prompt-injection attempt in a calendar entry and a user who denies every approval. It checks SAW's guarantees, not the model's choices:

* no email runs;
* nothing outside the allowed tools runs;
* no outside address even reaches the approval step;
* every tool function call went through SAW.

```powershell
pip install anthropic openai
set ANTHROPIC_API_KEY=...
set SAW_LIVE_ANTHROPIC_MODEL=<model name>
set OPENAI_API_KEY=...
set SAW_LIVE_OPENAI_MODEL=<model name>
set SAW_LIVE_COMPAT_BASE_URL=http://localhost:11434/v1
set SAW_LIVE_COMPAT_MODEL=<local model>
pytest -m live -v
```

Use `export` instead of `set` on macOS/Linux. Providers without settings are skipped. These tests call real APIs and may cost money.

## Writing your own loop

If you don't use `run_agent`, keep its guarantees:

1. Call tools only through `security.execute_tool(name, arguments)`, passing the model's values unchanged.
2. Advertise only `security.tool_definitions()`.
3. Send the model `outcome.for_model()`, never `outcome.error` or `repr(outcome)`.
4. Treat `APPROVAL_REQUIRED` as "not done". Decide it through the `ApprovalDesk` from application code, and never expose the desk to the model or to tools.
5. Answer every tool use the model makes.
6. Cap the number of steps and tool calls.

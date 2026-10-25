# Reference

Everything public in SAW 0.1.0. Names are importable from `saw` unless shown otherwise.

## `SAW`

```python
SAW(agent_id=None, *, config=None, permission_policy=None, tool_result_validator=None,
    logger=None, registry=None, approval_desk=None)
```

One security runtime per agent.

| Argument | Meaning |
|---|---|
| `agent_id` | Shortcut for `config=AgentConfig(agent_id=...)`, which allows no tools |
| `config` | The agent's `AgentConfig`. If both are given, the IDs must match |
| `permission_policy` | A `PermissionPolicy`. Default `DefaultPermissionPolicy` |
| `tool_result_validator` | A `ResultValidator`. Default `DefaultResultValidator` |
| `logger` | A `SecurityLogger`. Default `DefaultSecurityLogger` |
| `registry` | A `ToolRegistry`. Default: a new, empty one for this agent. Share one to give several agents the same tools; each agent's config still decides what it may call |
| `approval_desk` | An `ApprovalDesk`. Without one, calls that need approval are **denied** |

Raises `ConfigurationError` for invalid configuration or components, or if the logger fails while recording initialisation.

| Method / property | Description |
|---|---|
| `register_tool(name, *, schema, function, result_schema=None, description=None)` | Register a tool. See [Tools](#tools) |
| `execute_tool(tool_name, arguments=None) -> ToolCallOutcome` | Check and, if everything passes, run a tool. See [What happens on each call](#what-happens-on-each-call) |
| `tool_definitions() -> list[dict]` | `{"name", "description", "parameters"}` for each tool that is registered **and** allowed, sorted by name. `parameters` is a JSON Schema. Tools that need approval are included |
| `agent_id`, `config`, `permission_policy`, `tool_result_validator`, `logger`, `registry`, `registered_tools` | Read-only |

The desk is deliberately **not** reachable from the `SAW` object.

## `AgentConfig`

Immutable. Invalid values raise `ConfigurationError`.

| Field | Type | Default | Rules and meaning |
|---|---|---|---|
| `agent_id` | `str` | required | 1–128 characters; starts with a letter or digit; then letters, digits, `.`, `_`, `-`. A label, not an authenticated identity |
| `allowed_tools` | collection of `str` | empty | Tools this agent may call. A single string is rejected, to catch `allowed_tools="web_search"` |
| `approval_required` | collection of `str` | empty | Tools that need approval. Must be a subset of `allowed_tools` |
| `max_result_chars` | `int` | `100_000` | Total characters in a result's strings and dict keys |
| `oversized_results` | `str` | `"reject"` | `"reject"`, or `"truncate"` to shorten **plain-text** results to fit, with a visible marker |
| `max_result_depth` | `int` | `32` | Maximum nesting depth of a result |
| `max_result_nodes` | `int` | `100_000` | Maximum number of values in a result |
| `approval_ttl_seconds` | number | `900` | How long an approval request stays valid |
| `max_pending_approvals` | `int` | `100` | Approval requests waiting at once for this agent; more are denied |

To change what an agent may do, create a new `SAW` with a new config. Pass the same `ToolRegistry` to keep its tools.

## Tools

`register_tool(name, *, schema, function, result_schema=None, description=None)`

* `name`: same format as `agent_id`. Registering the same name twice raises `ConfigurationError`, so a tool cannot be replaced.
* `schema`: a dataclass, or a Pydantic v2 model with `model_config = ConfigDict(extra="forbid")`. The function is called with one keyword argument per field. Registration fails if the function can't accept those fields.
* `function`: a regular function or callable. `async` and generator functions are rejected.
* `result_schema`: optional; the result must match it. The tool may return a plain dict or an instance of this class.
* `description`: shown to the model. Defaults to the first paragraph of the function's docstring. At most 1,024 characters.

### `ToolRegistry` and `RegisteredTool`

`ToolRegistry()` holds tools by name; `SAW.register_tool` delegates to the agent's registry. Methods: `register(name, *, schema, function, result_schema=None, description=None) -> RegisteredTool`, `get(name)`, `names`, `in`, `len`. Registration and lookup are thread-safe. Registering a tool makes it exist; whether an agent may call it is decided by that agent's configuration.

`RegisteredTool` (frozen): `name`, `schema`, `function`, `result_schema`, `description`. Calling `function` directly bypasses SAW; don't.

### Dataclass schemas (built in)

* Strict: no type conversion, except that an `int` is accepted for a `float`. `"10"` is not an `int`; `1` is not a `bool`; `True` is not an `int`.
* Unknown fields are rejected; missing required fields are reported.
* Supported types: `str`, `int`, `float` (finite), `bool`, `None`, `Optional[X]` / `X | None`, `Union[...]`, `Literal[...]`, `list[X]`, `dict[str, X]`, nested dataclasses. Anything else, including `Any`, bare `list` and recursive schemas, is rejected at registration.
* Extra checks go in `__post_init__`; raising there rejects the arguments with your message. For arguments, values set in `__post_init__` are what the tool receives.
* Define schemas at module level, so their annotations can be resolved.

### Pydantic v2 models (optional)

SAW never imports Pydantic; it recognises a model only if your code has imported Pydantic. Models are validated with `strict=True`, must set `extra="forbid"` (nested models too), and Pydantic v1 models are rejected. Tested with Pydantic 2.0 through 2.14.

## What happens on each call

`execute_tool` runs these steps in order and stops at the first that fails.

| Step | What happens | If it fails | Event |
|---|---|---|---|
| 1 | Record the call | `AuditError` | `TOOL_CALL_RECEIVED` |
| 2 | Tool is registered | `unknown_tool` | `TOOL_CALL_DENIED` |
| 3 | Arguments match the schema | `invalid_arguments` | `TOOL_CALL_DENIED` |
| 4 | Permission: stricter of config and policy | `denied` | `TOOL_CALL_DENIED` |
| 4a | If approval is needed: put the call on the desk and **stop** | `approval_required` (or `denied` with no desk) | `APPROVAL_REQUIRED` |
| 5 | Record the allow | `AuditError` | `TOOL_CALL_ALLOWED` |
| 6 | Run the tool, timing it | `failed`, or re-raised if interrupted | `TOOL_FAILED` |
| 7 | Result checks | `result_rejected` | `RESULT_REJECTED` |
| 8 | Record and return the accepted result | `AuditError` | `TOOL_EXECUTED` |

All events for one call share a `call_id`, also on the outcome.

### Failure behaviour

| Fails at | Status | Tool ran? | Result returned? |
|---|---|---|---|
| Unknown tool, bad arguments, permission denied, policy error | `unknown_tool` / `invalid_arguments` / `denied` | No | No |
| Needs approval | `approval_required` | Not until approved | No |
| Needs approval but no desk, desk closed, or too many pending | `denied` | No | No |
| Approval denied, expired, or permission denied at approval time | `denied` | No | No |
| Tool raises an `Exception` | `failed` | Yes | No |
| Tool interrupted (`KeyboardInterrupt`, `SystemExit`, cancellation) | re-raised after recording | Yes | No |
| Result fails a result check, or a check raises | `result_rejected` | Yes | No |
| Logger fails | `AuditError` raised | See `AuditError.tool_executed` | No |

## `ToolCallOutcome`

| Attribute | Meaning |
|---|---|
| `status` | A `CallStatus`: `EXECUTED`, `DENIED`, `APPROVAL_REQUIRED`, `UNKNOWN_TOOL`, `INVALID_ARGUMENTS`, `FAILED`, `RESULT_REJECTED` |
| `ok` | `True` only if the tool ran and its result was accepted |
| `value` | The accepted result. **Raises** the matching exception otherwise; `exc.outcome` is this outcome |
| `reason` | Why it was blocked or failed. Never contains argument values, result content, keys taken from data, or exception messages |
| `error` | Underlying exception, if any, for debugging. May contain sensitive details; don't send it to the model |
| `approval_request` | The `ApprovalRequest` when status is `APPROVAL_REQUIRED`, else `None` |
| `truncated` | `True` if a plain-text result was shortened to fit |
| `for_model()` | A dict that is safe to send to the model. Pending calls say the action has **not** been performed |
| `call_id`, `tool_name` | For correlating with events |

## Result checks

In order; the first failure rejects the result. Steps 2, 3 and 5 always run, whatever validator is used.

1. **Conversion**: a result-schema instance becomes a plain dict.
2. **Shape**: JSON-compatible values only (`str`, `int`, finite `float`, `bool`, `None`, and `list`, `tuple` or `dict` with string keys of those), within `max_result_depth` and `max_result_nodes`. `bytes` and objects are rejected; encode binary data as text.
3. **Size**: at most `max_result_chars`, or truncated if configured (plain text only).
4. **Validator**: your `ResultValidator`, on a copy.
5. **Result schema**: if registered.

A rejected result is withheld completely: it is not stored on the outcome, the exception or any event.

## Approvals

### `ApprovalDesk(*, approvers=None)`

Hold it in application code only. One desk may serve several agents.

| Member | Description |
|---|---|
| `approve(request_id, *, approver, note=None)` | Run the call with the requested arguments, after re-checking permission. Returns the call's outcome |
| `deny(request_id, *, approver, reason="denied by the user")` | Refuse; returns a `denied` outcome. The reason is shown to the model |
| `pending(agent_id=None)` | Waiting requests, oldest first. Expired ones are removed and recorded first |
| `close()` | Discard everything pending (recorded as `APPROVAL_EXPIRED` with `details["discarded"]`) and deny new requests. Also usable as a context manager |
| `approvers`, `closed` | Read-only |

Rules: each request can be decided once (even concurrently); it expires after `approval_ttl_seconds`; `approver` must be printable, at most 128 characters, in `approvers` if that is set, and different from the requesting agent's ID; decisions are refused while a tool is running in the same thread. Violations raise `ApprovalError`.

### `ApprovalRequest`

Frozen; describes the call and cannot run it. Fields: `request_id`, `agent_id`, `tool_name`, `arguments` (read-only copy; not in `repr`), `reason`, `created_at`, `expires_at` (UTC).

## Extension interfaces

Subclass, implement one method, pass an instance. Checked when `SAW(...)` is created: the object must be an instance of the interface, the method must be a plain `def` accepting the arguments below, not `async` and not a generator.

| | `PermissionPolicy` | `ResultValidator` | `SecurityLogger` |
|---|---|---|---|
| **Method** | `evaluate(self, call, config)` | `validate(self, call, result, config)` | `log(self, event)` |
| **Inputs** | Copy of the `ToolCall` (validated arguments); `AgentConfig` | Copy of the `ToolCall`; copy of the result (already shape- and size-checked); `AgentConfig` | A frozen `SecurityEvent` |
| **Return** | `PolicyDecision.allow(reason)` / `.deny(reason)` / `.require_approval(reason)` | `ValidationResult.accept()` / `.reject(reason)` | `None`; **raise** on failure |
| **Wrong return or exception** | Call denied | Result rejected | `AuditError`; returning anything but `None` also counts as failure |
| **Cannot override** | `allowed_tools`, `approval_required`, or approval itself | Shape, size and result-schema checks | Which events exist, or the no-unaudited-execution rule |
| **Default** | Applies the config rules | Adds no rules | One JSON line per event to the `saw.security` standard-library logger |

`PolicyDecision` and `ValidationResult` must be returned exactly as built by their constructors; a subclass carrying a non-`Decision` value, or `valid` that is not exactly `True`, is treated as a denial or rejection.

`ToolCall` fields: `tool_name`, `arguments` (read-only mapping), `call_id`.

## Security events

`SecurityEvent` (frozen): `event_type`, `agent_id`, `tool_name`, `call_id`, `decision`, `reason`, `details`, `timestamp` (UTC), `event_id`; `to_dict()` for a plain dict.

| `EventType` | When | Notable `details` |
|---|---|---|
| `AGENT_INITIALIZED` | `SAW(...)` created | Limits and component class names |
| `TOOL_CALL_RECEIVED` | Every call | `argument_count`, declared `argument_names` (never values), `undeclared_argument_count` |
| `TOOL_CALL_ALLOWED` | Before the tool runs | `config_decision`, `policy_decision`, `approved_by` |
| `TOOL_CALL_DENIED` | Blocked | `status`, and policy details when relevant |
| `APPROVAL_REQUIRED` | Put on the desk | `expires_at` |
| `APPROVAL_GRANTED` / `APPROVAL_DENIED` | Decided | `approver`, `note` |
| `APPROVAL_EXPIRED` | Expired or discarded | `discarded` |
| `TOOL_EXECUTED` | Accepted result | `duration_ms`, `result_type`, `result_chars`, `truncated` |
| `TOOL_FAILED` | Tool raised or was interrupted | `exception_type`, `duration_ms` |
| `RESULT_REJECTED` | Result rejected | `check`, `result_type`, `duration_ms` |

`DefaultSecurityLogger(logger=None)` writes blocks, failures and rejections at WARNING and everything else at INFO. Configure logging to see INFO, for example `logging.basicConfig(level=logging.INFO)`. To send events to your own standard-library logger, pass `DefaultSecurityLogger(your_logger)`.

## Exceptions

All derive from `SAWError`; none derive from `ValueError` or `TypeError`.

| Exception | Raised by |
|---|---|
| `ConfigurationError` | Setup: invalid config, component, registration or desk |
| `AuditError` | `execute_tool`, `desk.approve`/`deny`/`pending`/`close` when the logger fails. `tool_executed` says whether the tool had run; `outcome` holds a blocked outcome if any |
| `ApprovalError` | `desk.approve`/`deny`: unknown or decided request, invalid or unlisted approver, self-approval, or called while a tool is running |
| `ToolNotRegisteredError`, `ArgumentValidationError`, `PermissionDeniedError`, `ApprovalRequiredError`, `ToolExecutionError`, `ResultValidationError` | `outcome.value` when there is no accepted result |

## `saw.agent` and `saw.adapters`

See [LLM integration](llm-integration.md).

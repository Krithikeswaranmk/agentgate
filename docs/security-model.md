# Security model

Read this before relying on SAW. It explains what SAW is designed to stop, what it assumes, and what it does not protect.

## What SAW is for

An LLM agent decides which tools to call and with what arguments. Those decisions can be wrong, or steered by an attacker: a prompt-injected web page, document or email can tell the model to send data somewhere, delete something, or call a tool it shouldn't. SAW puts a checkpoint between the model's decisions and your tools, so that:

* only tools you allowed can run, with arguments of the declared shape;
* sensitive tools run only after a person approves the exact call;
* results that are malformed, oversized or rejected never reach the model;
* every decision is recorded, and nothing runs without an audit record.

## Trust assumptions

| Part | Treated as | Why |
|---|---|---|
| **Model output** (tool names, arguments, text) | **Untrusted** | It can be wrong or attacker-influenced |
| **Tool results** | **Untrusted** | Tools fetch outside content, which may be hostile or malformed |
| **Your application code** (configuration, registration, approval handling) | Trusted | It sets the rules |
| **Tool functions and custom components** (policy, validator, logger) | Trusted not to be **malicious**, but not trusted to be **correct** | They are your code. SAW fails closed when they misbehave, but it cannot contain code that deliberately attacks it (see limits) |
| **The person approving** | Identified by your application | SAW records the name you give it; it does not authenticate people |

## Guarantees

These hold for every call made through `SAW.execute_tool` and every decision made through an `ApprovalDesk`, whatever custom components are used. Each is covered by automated tests, including adversarial ones (`tests/test_adversarial.py`).

1. **Deny by default.** A tool runs only if it is registered and in `allowed_tools`.
2. **The configuration is a ceiling.** A custom policy can only make decisions stricter. It cannot allow an unlisted tool, skip a required approval, or approve anything itself.
3. **Blocked calls never run.** Unknown tools, invalid arguments, denials and pending approvals never reach the tool function. Only an exact `ALLOW` runs a tool; anything malformed is a denial.
4. **Approval binds the exact call.** It runs once, with the arguments that were shown, before it expires, after permission is checked again. An agent cannot approve its own request. Without a desk, calls that need approval are denied.
5. **Results are checked before the model sees them.** Shape, size and result schema always apply; a custom validator can only add rules. Rejected results are withheld entirely.
6. **Components get copies.** Nothing a policy or validator changes in place affects what the tool receives or what the model gets back.
7. **No unaudited execution.** If the logger fails before a tool runs, it doesn't run. If it fails afterwards, the result is withheld and `AuditError` is raised. A logger that returns anything other than `None` counts as failed.
8. **No secrets in logs or reasons.** Events and reasons never contain argument values, result content, keys taken from arguments or results, or exception messages. They do contain names declared in your schemas, types, counts and reasons you write.
9. **Fail closed.** A component that raises or returns the wrong type causes a denial or rejection, never an allow.

## Integration boundary

SAW protects **only what goes through it**:

* Calls through `SAW.execute_tool`, including those made by `saw.agent.run_agent`.
* Approval decisions through an `ApprovalDesk`.

It does **not** protect:

* Tool functions your code calls directly.
* Anything the model influences outside tool calls, such as the text it writes to the user.
* What a tool does internally once it runs. SAW checks the call and the result, not the tool's behaviour.

## Limits

These are deliberate design limits, not bugs. Each has a recommended mitigation.

| Limit | Why | What to do |
|---|---|---|
| **In-process only; not a sandbox.** Code in the same process can bypass SAW, for example by calling tool functions directly, editing private attributes, or reaching the desk object. | Python gives no isolation between objects in one process. | Treat tool and component code as trusted. Never give tools, or anything the model controls, a reference to the desk. Run untrusted code (shell commands, generated code) in a separate process or container. |
| **The inside-a-tool check is a guard, not a boundary.** `approve`/`deny` refuse to run while a tool runs in the same thread, but a thread the tool starts is not covered. A test documents this. | The marker cannot follow threads a tool creates. | Same as above: keep the desk out of reach of tools. Full protection requires process isolation (planned). |
| **No execution time limit.** | Python cannot stop a running function; a timeout would only stop waiting while the tool kept running and could still take effect. | Set timeouts in the libraries tools use, for example `requests.get(..., timeout=10)` or `subprocess.run(..., timeout=60)`, and an overall time budget in your agent framework. SAW records `duration_ms` for every call. |
| **Approvals are in memory.** A restart loses pending requests. | Persisting executable approvals would create a file someone could alter to change what runs. Losing them is safe: they never run. | Call `desk.close()` on shutdown so the log shows them as discarded. Ask the user again after a restart. |
| **Approver names are not authenticated.** | SAW has no identity system. | Authenticate people in your application, pass the authenticated name, and set an `approvers` allowlist on the desk. |
| **Result checks look at structure, not meaning.** | Detecting prompt injection or secrets in free text is unreliable. | Add your own `ResultValidator` rules where useful, and use approvals for actions with real-world effects. Treat accepted results as untrusted text. |
| **A rejected result still means the tool ran.** | Results can only be checked after they exist. | Put side-effecting tools behind approval. |
| **`agent_id` is a label.** | Permissions come from the `SAW` instance, never from an ID supplied at call time. | Create one `SAW` per agent and keep each in its own agent's code path. |
| **Not independently reviewed.** | Tested extensively, but not audited. | Review before using in high-risk settings. |

## Deployment checklist

* [ ] Every tool is called only through `execute_tool`, or through `run_agent`.
* [ ] `allowed_tools` lists the minimum each agent needs.
* [ ] Tools with real-world effects (sending, deleting, paying) are in `approval_required`.
* [ ] The `ApprovalDesk` is held only by application code, with an `approvers` allowlist of authenticated names.
* [ ] `desk.close()` is called on shutdown.
* [ ] Tools use timeouts in their own libraries.
* [ ] The security logger writes somewhere durable, and its failures are monitored. A failing logger stops tool use by design.
* [ ] Argument and result schemas are as strict as possible, with value rules in `__post_init__` or Pydantic constraints.
* [ ] Untrusted code execution happens outside the agent's process.

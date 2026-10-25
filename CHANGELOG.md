# Changelog

## 0.1.0 — first release

First documented, tested and installable release of SAW. Earlier `0.x.dev0`
versions were internal development builds and were never published.

### Included

- **Runtime:** `SAW.execute_tool` checks registration, arguments, permission
  and approval before a tool runs, then checks the result before returning
  it, recording every decision as a security event.
- **Configuration:** immutable per-agent `AgentConfig` (allowed tools,
  approval requirements, result limits, approval lifetime and cap).
- **Schemas:** strict built-in dataclass validation, optional Pydantic v2
  models, per-tool result schemas, JSON Schema export for LLM tool definitions.
- **Permissions:** the stricter of configuration and a replaceable
  `PermissionPolicy`; a policy can never loosen the configuration.
- **Result checks:** mandatory shape and size limits, replaceable
  `ResultValidator`, tool result schemas; rejected results are never stored.
- **Approvals:** `ApprovalDesk` held by the application; bound arguments,
  single use, expiry, re-check, optional approver allowlist, no
  self-approval, per-agent cap, fail closed without a desk, `close()`.
- **Audit:** replaceable `SecurityLogger`; no unaudited execution.
- **Agent loop:** `saw.agent.run_agent`, provider-neutral.
- **Adapters:** `saw.adapters.anthropic` (Messages API) and
  `saw.adapters.openai` (Chat Completions, and compatible services).
- **Examples:** three offline demo agents and a custom-components example.

### Before publishing to PyPI

- **Name:** `saw` is taken on PyPI by an unrelated project (a SimFin API
  wrapper) that also uses `import saw`. This release is named
  `saw-agent-security`; the two packages cannot be installed together.
  Choose the final distribution and import names before uploading.
- **License:** none declared yet.

### Not verified

- Calls to real LLM providers. The adapters are tested against the real
  Anthropic and OpenAI SDKs with mocked HTTP; the live tests
  (`pytest -m live`) need API keys and have not been run.

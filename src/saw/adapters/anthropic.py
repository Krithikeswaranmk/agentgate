"""Adapter for the Anthropic Messages API.

Example::

    import anthropic
    from saw.agent import run_agent
    from saw.adapters.anthropic import AnthropicModel

    model = AnthropicModel(anthropic.Anthropic(), model="<model name>")
    run = run_agent(security, model, "Check my Monday", approval_desk=desk,
                    approval_handler=ask_the_user)

The adapter calls ``client.messages.create`` with the conversation and the
agent's tool definitions, and turns ``tool_use`` blocks into
:class:`~saw.agent.ToolUse`. Tool results go back as ``tool_result`` blocks
containing ``outcome.for_model()``; blocked calls are marked ``is_error``.
"""

from __future__ import annotations

from typing import Any

from saw.adapters import get, tool_result_text
from saw.agent import FinalAnswer, ModelTurn, Step, ToolUse


class AnthropicModel:
    """Drive :func:`saw.agent.run_agent` with an Anthropic model.

    Args:
        client: An ``anthropic.Anthropic`` client (or anything with the same
            ``messages.create`` method).
        model: The model name.
        max_tokens: Maximum tokens per reply.
        system: Optional system prompt.
        **create_options: Extra keyword arguments for ``messages.create``.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        max_tokens: int = 1024,
        system: str | None = None,
        **create_options: Any,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self.create_options = create_options

    def next_step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Step:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": to_anthropic_messages(messages),
            **self.create_options,
        }
        if tools:
            request["tools"] = to_anthropic_tools(tools)
        if self.system:
            request["system"] = self.system
        return from_anthropic_response(self.client.messages.create(**request))


def to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ``SAW.tool_definitions()`` to Anthropic tool parameters."""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in tools
    ]


def to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the loop's neutral messages to Anthropic messages."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "user":
            converted.append({"role": "user", "content": message["content"]})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for use in message.get("tool_uses", []):
                arguments = use["arguments"] if isinstance(use["arguments"], dict) else {}
                blocks.append({
                    "type": "tool_use", "id": use["id"], "name": str(use["name"]),
                    "input": arguments,
                })
            converted.append({"role": "assistant", "content": blocks or ""})
        elif role == "tool_results":
            converted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["tool_use_id"],
                        "content": tool_result_text(r["content"]),
                        "is_error": bool(r["is_error"]),
                    }
                    for r in message["results"]
                ],
            })
        else:
            raise ValueError(f"unknown message role {role!r}")
    return converted


def from_anthropic_response(response: Any) -> Step:
    """Turn an Anthropic ``Message`` (or equivalent dict) into a step."""
    text_parts: list[str] = []
    uses: list[ToolUse] = []
    for block in get(response, "content", None) or []:
        kind = get(block, "type")
        if kind == "text":
            text_parts.append(get(block, "text", "") or "")
        elif kind == "tool_use":
            uses.append(ToolUse(name=get(block, "name"), arguments=get(block, "input"), id=get(block, "id", "")))
        # Other block types (server tools, thinking, ...) are not tool calls
        # for SAW to run, so they are ignored here.
    text = "".join(text_parts)
    if uses:
        return ModelTurn(tuple(uses), text)
    return FinalAnswer(text)


__all__ = [
    "AnthropicModel", "to_anthropic_tools", "to_anthropic_messages", "from_anthropic_response",
]

"""Adapter for the OpenAI Chat Completions API and compatible services.

Example::

    import openai
    from saw.agent import run_agent
    from saw.adapters.openai import OpenAIChatModel

    model = OpenAIChatModel(openai.OpenAI(), model="<model name>")
    # OpenAI-compatible services, e.g. a local Ollama server:
    # OpenAIChatModel(openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    #                 model="<local model>")

The adapter calls ``client.chat.completions.create`` with the conversation
and the agent's tools, and turns ``tool_calls`` into
:class:`~saw.agent.ToolUse`. Tool arguments arrive as a JSON string; if it
is not a valid JSON object, it is passed through unchanged so that SAW
reports ``invalid_arguments`` and nothing runs.
"""

from __future__ import annotations

import json
from typing import Any

from saw.adapters import get, tool_result_text
from saw.agent import FinalAnswer, ModelTurn, Step, ToolUse


class OpenAIChatModel:
    """Drive :func:`saw.agent.run_agent` with a Chat Completions model.

    Args:
        client: An ``openai.OpenAI`` client (or anything with the same
            ``chat.completions.create`` method).
        model: The model name.
        system: Optional system prompt.
        **create_options: Extra keyword arguments for ``create``.
    """

    def __init__(self, client: Any, model: str, *, system: str | None = None, **create_options: Any) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.create_options = create_options

    def next_step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Step:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages, system=self.system),
            **self.create_options,
        }
        if tools:
            request["tools"] = to_openai_tools(tools)
        return from_openai_response(self.client.chat.completions.create(**request))


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ``SAW.tool_definitions()`` to Chat Completions tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def to_openai_messages(
    messages: list[dict[str, Any]], *, system: str | None = None
) -> list[dict[str, Any]]:
    """Convert the loop's neutral messages to Chat Completions messages."""
    converted: list[dict[str, Any]] = []
    if system:
        converted.append({"role": "system", "content": system})
    for message in messages:
        role = message["role"]
        if role == "user":
            converted.append({"role": "user", "content": message["content"]})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or None}
            calls = [
                {
                    "id": use["id"],
                    "type": "function",
                    "function": {
                        "name": str(use["name"]),
                        "arguments": (
                            json.dumps(use["arguments"]) if isinstance(use["arguments"], dict)
                            else str(use["arguments"] if use["arguments"] is not None else "")
                        ),
                    },
                }
                for use in message.get("tool_uses", [])
            ]
            if calls:
                entry["tool_calls"] = calls
            converted.append(entry)
        elif role == "tool_results":
            for result in message["results"]:
                converted.append({
                    "role": "tool",
                    "tool_call_id": result["tool_use_id"],
                    "content": tool_result_text(result["content"]),
                })
        else:
            raise ValueError(f"unknown message role {role!r}")
    return converted


def _parse_arguments(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return raw  # SAW rejects a string as invalid_arguments; nothing runs
    return parsed


def from_openai_response(response: Any) -> Step:
    """Turn a ``ChatCompletion`` (or equivalent dict) into a step."""
    choices = get(response, "choices", None) or []
    if not choices:
        return FinalAnswer("")
    message = get(choices[0], "message")
    text = get(message, "content", None) or ""
    uses = []
    for call in get(message, "tool_calls", None) or []:
        if get(call, "type", "function") != "function":
            continue  # only function calls are SAW tools
        function = get(call, "function")
        uses.append(ToolUse(
            name=get(function, "name"),
            arguments=_parse_arguments(get(function, "arguments")),
            id=get(call, "id", ""),
        ))
    if uses:
        return ModelTurn(tuple(uses), text)
    return FinalAnswer(text)


__all__ = ["OpenAIChatModel", "to_openai_tools", "to_openai_messages", "from_openai_response"]

"""Thin adapters that let real LLMs drive :func:`saw.agent.run_agent`.

* :class:`saw.adapters.anthropic.AnthropicModel` - Anthropic Messages API.
* :class:`saw.adapters.openai.OpenAIChatModel` - OpenAI Chat Completions API,
  and services that implement it (for example Ollama, Groq or OpenRouter,
  through the OpenAI client's ``base_url``).

SAW does not depend on any provider SDK. Install the one you use and pass
its client in. The adapters only translate messages and tool definitions;
they make no security decisions. Everything the model asks for still goes
through ``SAW.execute_tool``.
"""

from __future__ import annotations

import json
from typing import Any


def get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an SDK object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def tool_result_text(content: Any) -> str:
    """Serialise a ``for_model()`` message for a provider's tool-result slot."""
    return json.dumps(content, default=str, ensure_ascii=False)


__all__ = ["get", "tool_result_text"]

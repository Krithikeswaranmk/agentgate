"""Tool registry: the tools an agent's runtime knows how to run."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from saw.config import _check_name
from saw.exceptions import ConfigurationError
from saw.schemas import ArgumentSchema, compile_schema


@dataclass(frozen=True)
class RegisteredTool:
    """A registered tool: name, argument schema, function and optional result schema."""

    name: str
    schema: ArgumentSchema
    function: Callable[..., Any]
    result_schema: ArgumentSchema | None = None
    description: str = ""


def _check_function(function: object, name: str) -> None:
    if not callable(function):
        raise ConfigurationError(
            f"function for tool {name!r} must be callable, got {type(function).__name__}"
        )
    # Calling these returns a coroutine or generator without running the body,
    # so SAW would report the tool as executed when nothing ran.
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        raise ConfigurationError(
            f"function for tool {name!r} is async; async tools are not supported yet"
        )
    if inspect.isgeneratorfunction(function):
        raise ConfigurationError(
            f"function for tool {name!r} is a generator function; return a value instead"
        )


def _check_signature(function: Callable[..., Any], schema: ArgumentSchema, name: str) -> None:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return  # Some built-ins have no signature; checked at call time instead.
    try:
        signature.bind(**{field: None for field in schema.field_names})
    except TypeError as exc:
        raise ConfigurationError(
            f"function for tool {name!r} cannot be called with the fields of "
            f"{schema.source.__name__} {list(schema.field_names)}: {exc}"
        ) from exc


def _description(description: object, function: Callable[..., Any], name: str) -> str:
    if description is None:
        doc = inspect.getdoc(function) or ""
        description = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    if not isinstance(description, str):
        raise ConfigurationError(f"description for tool {name!r} must be a string")
    if len(description) > 1024:
        raise ConfigurationError(
            f"description for tool {name!r} is longer than 1024 characters"
        )
    return description


class ToolRegistry:
    """Holds registered tools by name.

    Each ``SAW`` instance creates its own registry unless one is passed in.
    Registering a tool makes it *exist*; whether an agent may *call* it is
    decided by that agent's configuration and policy.

    Registration and lookup are thread-safe.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        *,
        schema: type,
        function: Callable[..., Any],
        result_schema: type | None = None,
        description: str | None = None,
    ) -> RegisteredTool:
        """Register a tool.

        Args:
            name: Tool name, in the same format as agent IDs.
            schema: A dataclass or a Pydantic v2 model describing the
                arguments. The function is called with one keyword argument
                per schema field.
            function: A regular (non-async, non-generator) callable.
            result_schema: Optional dataclass or Pydantic v2 model the result
                must match. The tool may return a plain dict or an instance
                of this class. Checks are strict, as for arguments.
            description: What the tool does, shown to the model in tool
                definitions. Defaults to the first paragraph of the
                function's docstring. At most 1,024 characters.

        Raises:
            ConfigurationError: If the name is invalid or already registered,
                the schema is unsupported, or the function cannot accept the
                schema's fields.
        """
        _check_name(name, "tool name")
        _check_function(function, name)
        compiled = compile_schema(schema)
        _check_signature(function, compiled, name)
        compiled_result = None if result_schema is None else compile_schema(result_schema)
        text = _description(description, function, name)
        tool = RegisteredTool(
            name=name, schema=compiled, function=function, result_schema=compiled_result,
            description=text,
        )
        with self._lock:
            if name in self._tools:
                raise ConfigurationError(
                    f"tool {name!r} is already registered; registered tools "
                    "cannot be replaced"
                )
            self._tools[name] = tool
        return tool

    def get(self, name: object) -> RegisteredTool | None:
        """Return the tool named ``name``, or ``None``."""
        if not isinstance(name, str):
            return None
        with self._lock:
            return self._tools.get(name)

    @property
    def names(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._tools)

    def __contains__(self, name: object) -> bool:
        return self.get(name) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)


__all__ = ["ToolRegistry", "RegisteredTool"]

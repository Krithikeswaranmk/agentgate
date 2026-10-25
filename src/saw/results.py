"""Checks a tool's result must pass before it reaches the agent.

Applied in this order; the first that fails rejects the result:

1. **Conversion** (only for tools with a ``result_schema``): if the tool
   returned an instance of its result schema, it is converted to a plain
   dict. Dataclasses do not enforce their field types, so the instance
   itself cannot be trusted.
2. **Shape** (mandatory): only JSON-compatible values - ``str``, ``int``,
   finite ``float``, ``bool``, ``None``, and ``list``, ``tuple`` or ``dict``
   with string keys of those - nested at most ``max_result_depth`` deep and
   at most ``max_result_nodes`` values in total.
3. **Size** (mandatory): at most ``max_result_chars`` characters in strings
   and dict keys. With ``oversized_results="truncate"``, a plain-text result
   is shortened to fit and given a visible marker; anything else over the
   limit is rejected.
4. **Agent result validator**: the agent's ``ResultValidator``, for
   application rules. It receives a deep copy.
5. **Tool result schema**, if registered: strict type and value checks.
   Schema validation builds new objects, so even a ``__post_init__`` that
   modifies its fields cannot change the result itself.

Steps 1-3 and 5 are enforced by the runtime, so a custom validator can only
add rules. The validator receives a deep copy and schemas build new objects,
so nothing either changes in place can reach the agent. The agent receives what the tool
returned, except for the conversion in step 1 and opt-in truncation in step
3, which are recorded.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from saw.calls import ToolCall
from saw.config import AgentConfig
from saw.exceptions import ArgumentValidationError
from saw.execution import discard_awaitable
from saw.schemas import ArgumentSchema, DataclassSchema, PydanticSchema
from saw.validation.base import ResultValidator, ValidationResult

#: Names of the checks, as recorded in ``RESULT_REJECTED`` events.
CHECK_CONVERSION = "result_conversion"
CHECK_SHAPE = "result_shape"
CHECK_SIZE = "result_size"
CHECK_VALIDATOR = "result_validator"
CHECK_SCHEMA = "result_schema"

_SCALARS = (str, int, float, bool, type(None))
_CONTAINERS = (list, tuple, set, frozenset)


@dataclass(frozen=True)
class ResultCheck:
    """Outcome of checking a result.

    Attributes:
        accepted: The result may be returned.
        value: The value to return. Only set when accepted.
        check: Which check rejected the result.
        reason: Why it was rejected. Never includes result content.
        error: Exception raised by a check, if any, for debugging.
        result_chars: Characters counted. When the result was rejected for
            size, this is a lower bound, because counting stops early.
        truncated: The result was shortened to fit the size limit.
        original_chars: Length before truncation.
        converted: The tool returned a schema instance that was converted
            to a dict.
    """

    accepted: bool
    value: Any = None
    check: str | None = None
    reason: str = ""
    error: BaseException | None = None
    result_chars: int = 0
    truncated: bool = False
    original_chars: int | None = None
    converted: bool = False


def check_shape(value: Any, max_depth: int, max_nodes: int) -> str | None:
    """Return why ``value`` is not JSON-compatible within the limits, or None.

    Reasons never include values or keys from ``value``. The walk is
    iterative, so deep nesting cannot hit Python's recursion limit, and the
    node limit stops self-referencing containers from looping forever.
    """
    stack: list[tuple[Any, int, str]] = [(value, 0, "result")]
    visited = 0
    while stack:
        item, depth, path = stack.pop()
        visited += 1
        if visited > max_nodes:
            return f"more than {max_nodes} values"
        if depth > max_depth:
            return f"nesting deeper than {max_depth} at {path}"
        if isinstance(item, _SCALARS):
            if isinstance(item, float) and not math.isfinite(item):
                return f"non-finite number at {path}"
        elif isinstance(item, (list, tuple)):
            for i, element in enumerate(item):
                stack.append((element, depth + 1, f"{path}[{i}]"))
        elif isinstance(item, dict):
            for key, element in item.items():
                if not isinstance(key, str):
                    return f"non-string key of type {type(key).__name__} at {path}"
                stack.append((element, depth + 1, f"{path}[<key>]"))
        else:
            return f"unsupported type {type(item).__name__} at {path}"
    return None


def count_chars(value: Any, stop_after: int) -> int:
    """Count characters in strings, bytes and dict keys inside ``value``.

    Stops once the count exceeds ``stop_after``. Each container is visited
    once, so self-referencing structures terminate.
    """
    total = 0
    stack = [value]
    seen: set[int] = set()
    while stack and total <= stop_after:
        item = stack.pop()
        if isinstance(item, (str, bytes, bytearray)):
            total += len(item)
        elif isinstance(item, Mapping) or isinstance(item, _CONTAINERS):
            if id(item) in seen:
                continue
            seen.add(id(item))
            if isinstance(item, Mapping):
                for key, element in item.items():
                    if isinstance(key, str):
                        total += len(key)
                    stack.append(element)
            else:
                stack.extend(item)
    return total


def _reject(check: str, reason: str, **extra: Any) -> ResultCheck:
    return ResultCheck(accepted=False, check=check, reason=reason, **extra)


def _convert(result: Any, schema: ArgumentSchema | None) -> tuple[Any, bool]:
    if schema is None or not isinstance(result, schema.source):
        return result, False
    if isinstance(schema, DataclassSchema):
        return dataclasses.asdict(result), True
    if isinstance(schema, PydanticSchema):
        return result.model_dump(), True
    return result, False


def _truncate(text: str, limit: int) -> str | None:
    n = len(text)
    # The marker uses `limit` digits as an upper bound, so the final length
    # never exceeds the limit.
    room = limit - len(f"\n[SAW: result truncated to {limit} of {n} characters]")
    if room <= 0:
        return None
    return text[:room] + f"\n[SAW: result truncated to {room} of {n} characters]"


def _as_result_errors(exc: ArgumentValidationError) -> str:
    # Schema messages call the top level "arguments"; here it is the result.
    parts = [
        "result" + e[len("arguments"):] if e.startswith("arguments") else e
        for e in exc.errors
    ]
    return "; ".join(parts) or str(exc)


def check_result(
    result: Any,
    *,
    call: ToolCall,
    config: AgentConfig,
    validator: ResultValidator,
    result_schema: ArgumentSchema | None,
) -> ResultCheck:
    """Run every result check. Never raises for check failures."""
    name = call.tool_name

    # 1. Conversion of schema instances.
    try:
        value, converted = _convert(result, result_schema)
    except Exception as exc:
        return _reject(
            CHECK_CONVERSION,
            f"result of tool {name!r} could not be converted to a dict "
            f"({type(exc).__name__})",
            error=exc,
        )

    # 2. Shape (mandatory).
    problem = check_shape(value, config.max_result_depth, config.max_result_nodes)
    if problem is not None:
        return _reject(
            CHECK_SHAPE,
            f"result of tool {name!r} rejected: {problem}",
            converted=converted,
        )

    # 3. Size (mandatory).
    limit = config.max_result_chars
    truncated, original = False, None
    if isinstance(value, str):
        chars = len(value)
        if chars > limit:
            shortened = (
                _truncate(value, limit) if config.oversized_results == "truncate" else None
            )
            if shortened is None:
                return _reject(
                    CHECK_SIZE,
                    f"result of tool {name!r} has {chars} characters, over the "
                    f"limit of {limit}",
                    result_chars=chars, converted=converted,
                )
            value, truncated, original, chars = shortened, True, chars, len(shortened)
    else:
        chars = count_chars(value, limit)
        if chars > limit:
            note = (
                "; only plain-text results can be truncated"
                if config.oversized_results == "truncate"
                else ""
            )
            return _reject(
                CHECK_SIZE,
                f"result of tool {name!r} has more than {limit} characters "
                f"of text{note}",
                result_chars=chars, converted=converted,
            )

    common = {"result_chars": chars, "converted": converted}

    # 4. Agent result validator, on its own copy.
    try:
        verdict = validator.validate(call, copy.deepcopy(value), config)
    except Exception as exc:
        return _reject(
            CHECK_VALIDATOR,
            f"result validator raised {type(exc).__name__}; result of tool "
            f"{name!r} rejected",
            error=exc, **common,
        )
    if not isinstance(verdict, ValidationResult):
        discard_awaitable(verdict)
        return _reject(
            CHECK_VALIDATOR,
            f"result validator returned {type(verdict).__name__}, not a "
            f"ValidationResult; result of tool {name!r} rejected",
            **common,
        )
    if verdict.valid is not True:  # only a real True accepts
        reason = verdict.reason if isinstance(verdict.reason, str) and verdict.reason else (
            f"result validator rejected the result of tool {name!r}"
        )
        return _reject(CHECK_VALIDATOR, reason, **common)

    # 5. Tool result schema. Schema validation builds new objects (new
    # lists, dicts and instances), so it cannot change ``value``.
    if result_schema is not None:
        if not isinstance(value, Mapping):
            return _reject(
                CHECK_SCHEMA,
                f"result of tool {name!r} must be an object matching "
                f"{result_schema.source.__name__}, got {type(value).__name__}",
                **common,
            )
        try:
            result_schema.validate(value, redact_keys=True)
        except ArgumentValidationError as exc:
            return _reject(
                CHECK_SCHEMA,
                f"result of tool {name!r} does not match "
                f"{result_schema.source.__name__}: {_as_result_errors(exc)}",
                **common,
            )
        except Exception as exc:  # defensive: a schema bug must not accept
            return _reject(
                CHECK_SCHEMA,
                f"result schema check for tool {name!r} failed with "
                f"{type(exc).__name__}",
                error=exc, **common,
            )

    return ResultCheck(
        accepted=True, value=value, result_chars=chars,
        truncated=truncated, original_chars=original, converted=converted,
    )


__all__ = [
    "ResultCheck",
    "check_result",
    "check_shape",
    "count_chars",
    "CHECK_CONVERSION",
    "CHECK_SHAPE",
    "CHECK_SIZE",
    "CHECK_VALIDATOR",
    "CHECK_SCHEMA",
]

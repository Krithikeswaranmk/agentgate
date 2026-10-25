"""Checks applied to custom components when a ``SAW`` runtime is created.

A custom component must:

* be an **instance** of a subclass of the matching interface;
* implement the interface's method as a regular method (not ``async def``,
  not a generator), so SAW gets a real answer when it calls it;
* accept the arguments SAW passes, positionally.

Mistakes that can be detected here raise ``ConfigurationError`` before any
tool can run, naming the class and method. Mistakes that can only show up
during a call (an exception, a wrong return type) fail closed at that point:
see each interface's contract.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from saw.audit.base import SecurityLogger
from saw.audit.default import DefaultSecurityLogger
from saw.exceptions import ConfigurationError
from saw.execution import discard_awaitable  # noqa: F401  (re-exported)
from saw.policies.base import PermissionPolicy
from saw.policies.default import DefaultPermissionPolicy
from saw.validation.base import ResultValidator
from saw.validation.default import DefaultResultValidator


@dataclass(frozen=True)
class ComponentSpec:
    """How SAW checks one kind of component."""

    argument: str
    interface: type
    default: type
    method: str
    parameters: tuple[str, ...]


PERMISSION_POLICY = ComponentSpec(
    "permission_policy", PermissionPolicy, DefaultPermissionPolicy,
    "evaluate", ("call", "config"),
)
TOOL_RESULT_VALIDATOR = ComponentSpec(
    "tool_result_validator", ResultValidator, DefaultResultValidator,
    "validate", ("call", "result", "config"),
)
LOGGER = ComponentSpec(
    "logger", SecurityLogger, DefaultSecurityLogger,
    "log", ("event",),
)


def resolve_component(value: object, spec: ComponentSpec) -> Any:
    """Return ``value`` after checking it, or a new default if it is ``None``.

    Raises:
        ConfigurationError: If ``value`` does not meet the component contract.
    """
    if value is None:
        return spec.default()

    expected = f"an instance of a {spec.interface.__name__} subclass"
    if isinstance(value, type):
        raise ConfigurationError(
            f"{spec.argument} must be {expected}; got the class "
            f"{value.__name__} itself. Did you mean {value.__name__}()?"
        )
    if not isinstance(value, spec.interface):
        hint = ""
        if spec is LOGGER and isinstance(value, logging.Logger):
            hint = (
                ". To send events to a standard-library logger, pass "
                "DefaultSecurityLogger(your_logger)"
            )
        raise ConfigurationError(
            f"{spec.argument} must be {expected}, got "
            f"{type(value).__name__}{hint}"
        )
    check_method(value, spec)
    return value


def check_method(component: object, spec: ComponentSpec) -> None:
    """Check the component's interface method can be called as SAW calls it."""
    name = f"{type(component).__name__}.{spec.method}()"
    method = getattr(component, spec.method, None)
    if not callable(method):
        raise ConfigurationError(f"{spec.argument}: {name} is not callable")
    if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
        raise ConfigurationError(
            f"{spec.argument}: {name} is async. SAW calls it synchronously, so "
            "it would never run; define it with plain 'def'"
        )
    if inspect.isgeneratorfunction(method):
        raise ConfigurationError(
            f"{spec.argument}: {name} is a generator function; it must return "
            "its answer, not yield it"
        )
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return  # no signature available (some C callables); checked at call time
    try:
        signature.bind(*(None for _ in spec.parameters))
    except TypeError:
        raise ConfigurationError(
            f"{spec.argument}: {name} must accept {len(spec.parameters)} "
            f"positional argument(s) ({', '.join(spec.parameters)}), but its "
            f"signature is {signature}"
        ) from None


__all__ = [
    "ComponentSpec",
    "PERMISSION_POLICY",
    "TOOL_RESULT_VALIDATOR",
    "LOGGER",
    "resolve_component",
    "check_method",
    "discard_awaitable",
]

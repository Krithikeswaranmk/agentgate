"""Running a tool function after all checks have passed.

SAW does not limit how long a tool runs. Python cannot safely stop a running
function from outside, so an in-process timeout would only stop *waiting*
while the tool kept running and could still take effect. Tool authors should
set timeouts in the libraries their tools use (for example
``requests.get(..., timeout=...)`` or ``subprocess.run(..., timeout=...)``),
and the agent framework should enforce an overall time budget. SAW records
how long each call took.
"""

from __future__ import annotations

import contextvars
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Invocation:
    """Result of running a tool function.

    Attributes:
        succeeded: The function returned normally with a usable value.
        value: The returned value, if it succeeded.
        error: The exception raised, if any.
        reason: Why it did not succeed. Never includes exception messages.
        interrupted: The call was stopped by ``KeyboardInterrupt``,
            ``SystemExit``, cancellation or another ``BaseException`` that is
            not an ``Exception``. The runtime records it, then re-raises
            ``error``.
        duration_ms: Wall-clock time spent in the function, in milliseconds.
    """

    succeeded: bool
    value: Any = None
    error: BaseException | None = None
    reason: str = ""
    interrupted: bool = False
    duration_ms: float = 0.0


#: True while a tool function is running. Approval decisions are refused
#: while it is set. Threads started by a tool do not inherit it, so this guards
#: against mistakes (such as exposing approvals as a tool), not deliberate
#: bypasses; see the README's security model.
inside_tool: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "saw_inside_tool", default=False
)


def discard_awaitable(value: object) -> None:
    """Close a coroutine returned where a plain value was expected.

    Closing it means its body never runs and Python does not warn about a
    coroutine that was never awaited.
    """
    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            close()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def invoke(function: Callable[..., Any], kwargs: dict[str, Any]) -> Invocation:
    """Call ``function(**kwargs)``. Never raises.

    * An ``Exception`` from the function becomes a failed invocation.
    * Any other ``BaseException`` (interrupt, exit, cancellation) becomes an
      interrupted invocation; the caller must re-raise it after recording.
    * If the function returns an awaitable (for example a callable object
      with an ``async __call__``), the tool did not really run. The
      coroutine is closed and the invocation fails.
    """
    start = time.perf_counter()
    token = inside_tool.set(True)
    try:
        value = function(**kwargs)
    except Exception as exc:
        return Invocation(
            False, error=exc, reason=f"tool raised {type(exc).__name__}",
            duration_ms=_ms(start),
        )
    except BaseException as exc:  # KeyboardInterrupt, SystemExit, CancelledError
        return Invocation(
            False, error=exc, reason=f"tool was interrupted by {type(exc).__name__}",
            interrupted=True, duration_ms=_ms(start),
        )
    finally:
        inside_tool.reset(token)

    duration = _ms(start)
    if inspect.isawaitable(value):
        discard_awaitable(value)
        return Invocation(
            False,
            reason="tool returned an awaitable; async tools are not supported yet",
            duration_ms=duration,
        )
    return Invocation(True, value=value, duration_ms=duration)


__all__ = ["Invocation", "invoke", "discard_awaitable", "inside_tool"]

import asyncio
import inspect
import time

import pytest

from saw.execution import invoke


def test_successful_invocation():
    result = invoke(lambda x: x * 2, {"x": 3})
    assert result.succeeded and result.value == 6 and result.error is None


def test_none_return_is_success():
    result = invoke(lambda: None, {})
    assert result.succeeded and result.value is None


def test_exception_becomes_failure():
    def boom():
        raise ValueError("secret detail")

    result = invoke(boom, {})
    assert not result.succeeded
    assert isinstance(result.error, ValueError)
    assert result.reason == "tool raised ValueError"
    assert "secret detail" not in result.reason


def test_wrong_kwargs_become_failure():
    result = invoke(lambda a: a, {"b": 1})
    assert not result.succeeded and isinstance(result.error, TypeError)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_interrupts_are_reported_not_swallowed(exc_type):
    def interrupt():
        raise exc_type()

    result = invoke(interrupt, {})
    assert not result.succeeded and result.interrupted
    assert isinstance(result.error, exc_type)
    assert exc_type.__name__ in result.reason


def test_ordinary_exception_is_not_an_interrupt():
    result = invoke(lambda: 1 / 0, {})
    assert not result.interrupted


def test_duration_is_recorded():
    def slow():
        time.sleep(0.02)

    result = invoke(slow, {})
    assert result.succeeded and result.duration_ms >= 15
    assert invoke(lambda: 1 / 0, {}).duration_ms >= 0


def test_awaitable_result_fails_and_coroutine_is_closed():
    ran = []

    async def body():
        ran.append(True)

    returned = []

    class AsyncCallable:
        def __call__(self):
            returned.append(body())
            return returned[-1]

    result = invoke(AsyncCallable(), {})
    assert not result.succeeded
    assert "awaitable" in result.reason
    assert ran == []
    assert inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED

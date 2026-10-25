from __future__ import annotations

import functools
import threading
from dataclasses import dataclass

import pytest

from saw import ConfigurationError, RegisteredTool, ToolRegistry


@dataclass
class PathArgs:
    path: str


@dataclass
class NoArgs:
    pass


def read(path: str) -> str:
    return path


def test_register_and_get():
    registry = ToolRegistry()
    tool = registry.register("read_document", schema=PathArgs, function=read)
    assert isinstance(tool, RegisteredTool)
    assert registry.get("read_document") is tool
    assert "read_document" in registry
    assert registry.names == frozenset({"read_document"})
    assert len(registry) == 1


def test_unknown_and_non_string_lookups_return_none():
    registry = ToolRegistry()
    assert registry.get("missing") is None
    assert registry.get(None) is None
    assert registry.get(["read"]) is None
    assert 123 not in registry


def test_duplicate_name_rejected_and_original_kept():
    registry = ToolRegistry()
    original = registry.register("read", schema=PathArgs, function=read)
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register("read", schema=PathArgs, function=lambda path: "evil")
    assert registry.get("read") is original


@pytest.mark.parametrize("name", ["", "bad name", "-x", None, 5])
def test_invalid_names_rejected(name):
    with pytest.raises(ConfigurationError, match="tool name"):
        ToolRegistry().register(name, schema=PathArgs, function=read)


def test_non_callable_rejected():
    with pytest.raises(ConfigurationError, match="must be callable"):
        ToolRegistry().register("t", schema=PathArgs, function="read")


async def async_tool(path: str) -> str:
    return path


async def async_gen_tool(path: str):
    yield path


def gen_tool(path: str):
    yield path


@pytest.mark.parametrize(
    "function, message",
    [
        (async_tool, "async"),
        (functools.partial(async_tool), "async"),
        (async_gen_tool, "async"),
        (gen_tool, "generator"),
    ],
)
def test_async_and_generator_functions_rejected(function, message):
    with pytest.raises(ConfigurationError, match=message):
        ToolRegistry().register("t", schema=PathArgs, function=function)


@pytest.mark.parametrize(
    "function",
    [
        lambda: None,  # missing parameter
        lambda other: None,  # wrong name
        lambda path, required_extra: None,  # extra required parameter
    ],
)
def test_function_signature_must_accept_schema_fields(function):
    with pytest.raises(ConfigurationError, match="cannot be called with the fields"):
        ToolRegistry().register("t", schema=PathArgs, function=function)


@pytest.mark.parametrize(
    "function",
    [
        lambda path: None,
        lambda path, optional=1: None,
        lambda **kwargs: None,
    ],
)
def test_compatible_signatures_accepted(function):
    ToolRegistry().register("t", schema=PathArgs, function=function)


def test_callable_object_accepted():
    class Tool:
        def __call__(self, path: str) -> str:
            return path

    ToolRegistry().register("t", schema=PathArgs, function=Tool())


def test_bad_schema_rejected_and_nothing_registered():
    registry = ToolRegistry()
    with pytest.raises(ConfigurationError):
        registry.register("t", schema=dict, function=read)
    assert "t" not in registry


def test_concurrent_duplicate_registration_allows_exactly_one():
    registry = ToolRegistry()
    successes, failures = [], []
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        try:
            registry.register("t", schema=NoArgs, function=lambda: None)
            successes.append(1)
        except ConfigurationError:
            failures.append(1)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(successes) == 1 and len(failures) == 15

"""The default validator and the mandatory shape check."""

import io
import math

import pytest

from saw import AgentConfig, DefaultResultValidator, ToolCall, ValidationResult
from saw.results import check_shape

CFG = AgentConfig(agent_id="a")
CALL = ToolCall("web_search")


@pytest.mark.parametrize("result", [None, "text", 1, b"bytes", object(), {"a": {1, 2}}])
def test_default_validator_adds_no_rules(result):
    assert DefaultResultValidator().validate(CALL, result, CFG).valid


def shape(value, depth=32, nodes=100_000):
    return check_shape(value, depth, nodes)


@pytest.mark.parametrize(
    "result",
    [None, "text", 0, -3, 1.5, True, [], {}, ("a", 1),
     {"results": [{"title": "x", "score": 0.9, "tags": ["a", "b"]}]}],
)
def test_json_compatible_results_pass_shape(result):
    assert shape(result) is None


@pytest.mark.parametrize(
    "result, fragment",
    [
        (b"bytes", "unsupported type bytes"),
        (object(), "unsupported type object"),
        ({"a": {1, 2}}, "unsupported type set"),
        (io.StringIO(), "unsupported type StringIO"),
        ((x for x in range(3)), "unsupported type generator"),
        ({1: "int key"}, "non-string key of type int"),
        ([1.0, math.nan], "non-finite"),
        ({"x": math.inf}, "non-finite"),
    ],
)
def test_unsupported_shapes_rejected(result, fragment):
    assert fragment in shape(result)


def test_too_deep_rejected():
    assert shape([[["ok"]]], depth=3) is None
    assert "nesting deeper than 3" in shape([[[["too deep"]]]], depth=3)


def test_too_many_nodes_rejected():
    assert shape(list(range(9)), nodes=10) is None  # 9 items + the list = 10
    assert "more than 10 values" in shape(list(range(10)), nodes=10)


def test_self_referencing_result_terminates():
    loop = []
    loop.append(loop)
    assert shape(loop) is not None


def test_very_deep_result_does_not_hit_recursion_limit():
    deep = current = []
    for _ in range(5000):
        nxt = []
        current.append(nxt)
        current = nxt
    assert shape(deep, depth=10_000, nodes=10_000) is None


def test_shape_reasons_do_not_repeat_keys_or_values():
    reason = shape({"alice@example.com": {"token": b"SECRET"}})
    assert "alice" not in reason and "SECRET" not in reason and "token" not in reason


def test_validation_result_constructors():
    assert ValidationResult.accept().valid
    rejected = ValidationResult.reject("bad")
    assert not rejected.valid and rejected.reason == "bad"


@pytest.mark.parametrize("reason", ["", "  "])
def test_rejection_requires_reason(reason):
    with pytest.raises(TypeError, match="reason"):
        ValidationResult.reject(reason)


def test_valid_must_be_bool():
    with pytest.raises(TypeError, match="bool"):
        ValidationResult(1)  # type: ignore[arg-type]

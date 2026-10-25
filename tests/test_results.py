"""Unit tests for saw.results: the checks a result must pass."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from saw import AgentConfig, DefaultResultValidator, ResultValidator, ToolCall, ValidationResult
from saw.results import (
    CHECK_CONVERSION,
    CHECK_SHAPE,
    CHECK_SCHEMA,
    CHECK_SIZE,
    CHECK_VALIDATOR,
    check_result,
    count_chars,
)
from saw.schemas import compile_schema

CALL = ToolCall("run_tests")


@dataclass
class RunTestsResult:
    passed: bool
    failures: int

    def __post_init__(self):
        if self.failures < 0:
            raise ValueError("failures must be >= 0")


@dataclass
class Named:
    name: str


def run(result, *, schema=RunTestsResult, validator=None, **config):
    config.setdefault("agent_id", "coding-agent")
    return check_result(
        result,
        call=CALL,
        config=AgentConfig(**config),
        validator=validator or DefaultResultValidator(),
        result_schema=None if schema is None else compile_schema(schema),
    )


# ---------------------------------------------------------- spec example


def test_valid_result_passes_unchanged():
    result = {"passed": True, "failures": 0}
    checked = run(result)
    assert checked.accepted
    assert checked.value is result  # not copied or converted
    assert not checked.truncated and not checked.converted


def test_spec_example_rejected_by_schema():
    checked = run({"passed": "yes", "failures": -1})
    assert not checked.accepted
    assert checked.check == CHECK_SCHEMA
    assert "passed: expected boolean, got str" in checked.reason
    assert checked.value is None


def test_value_rule_in_post_init_rejects():
    checked = run({"passed": True, "failures": -1})
    assert not checked.accepted
    assert "failures must be >= 0" in checked.reason


@pytest.mark.parametrize(
    "result, fragment",
    [
        ({"passed": True}, "failures: missing required field"),
        ({"passed": True, "failures": 0, "extra": 1}, "result: unknown field"),
        ({"passed": True, "failures": 1.0}, "failures: expected integer"),
        ([True, 0], "must be an object matching RunTestsResult, got list"),
        ("all passed", "must be an object matching RunTestsResult, got str"),
        (None, "got NoneType"),
    ],
)
def test_schema_rejections(result, fragment):
    checked = run(result)
    assert not checked.accepted and checked.check == CHECK_SCHEMA
    assert fragment in checked.reason


def test_result_schema_messages_say_result_not_arguments():
    checked = run({"passed": True, "failures": 0, "x": 1})
    assert "arguments" not in checked.reason
    assert checked.reason.endswith("result: unknown field")


def test_keys_from_result_data_are_not_echoed():
    @dataclass
    class Scores:
        scores: dict[str, int]

    checked = run({"scores": {"alice@example.com": "high"}}, schema=Scores)
    assert not checked.accepted
    assert "alice" not in checked.reason
    assert "scores[<key>]: expected integer" in checked.reason

    checked = run({"alice@example.com": b"x"}, schema=None)
    assert not checked.accepted and "alice" not in checked.reason


# ---------------------------------------------------- schema instances


def test_schema_instance_is_converted_and_accepted():
    checked = run(RunTestsResult(passed=True, failures=0))
    assert checked.accepted and checked.converted
    assert checked.value == {"passed": True, "failures": 0}
    assert type(checked.value) is dict


def test_badly_typed_schema_instance_is_caught():
    # Dataclasses do not enforce types, so this constructs without error.
    instance = RunTestsResult(passed="yes", failures=0)  # type: ignore[arg-type]
    checked = run(instance)
    assert not checked.accepted
    assert checked.check == CHECK_SCHEMA
    assert "passed: expected boolean" in checked.reason


def test_instance_mutated_after_construction_is_caught():
    instance = RunTestsResult(passed=True, failures=0)
    instance.failures = -5
    assert not run(instance).accepted


def test_conversion_failure_rejects():
    class NoCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy")

    checked = run(Named(name=NoCopy()), schema=Named)  # type: ignore[arg-type]
    assert not checked.accepted and checked.check == CHECK_CONVERSION
    assert isinstance(checked.error, RuntimeError)


def test_instance_of_other_dataclass_is_not_converted():
    @dataclass
    class Other:
        passed: bool
        failures: int

    checked = run(Other(True, 0))
    assert not checked.accepted
    assert checked.check == CHECK_SHAPE  # not JSON-compatible


# ------------------------------------------------------- agent validator


def test_shape_check_runs_before_validator_and_schema():
    checked = run({"passed": True, "failures": 0, "blob": b"x"}, validator=AcceptAll())
    assert not checked.accepted and checked.check == CHECK_SHAPE


def test_returned_exception_object_is_rejected():
    checked = run(ValueError("boom"), schema=None, validator=AcceptAll())
    assert not checked.accepted and checked.check == CHECK_SHAPE


class Raising(ResultValidator):
    def validate(self, call, result, config):
        raise RuntimeError("validator bug")


class WrongType(ResultValidator):
    def validate(self, call, result, config):
        return True


class AcceptAll(ResultValidator):
    def validate(self, call, result, config):
        return ValidationResult.accept()


def test_validator_exception_rejects():
    checked = run({"passed": True, "failures": 0}, validator=Raising())
    assert not checked.accepted and checked.check == CHECK_VALIDATOR
    assert "RuntimeError" in checked.reason and "validator bug" not in checked.reason
    assert isinstance(checked.error, RuntimeError)


def test_validator_wrong_return_type_rejects():
    checked = run({"passed": True, "failures": 0}, validator=WrongType())
    assert not checked.accepted and "not a ValidationResult" in checked.reason


def test_permissive_validator_cannot_bypass_schema_or_size():
    assert not run({"passed": "yes", "failures": 0}, validator=AcceptAll()).accepted
    big = run("x" * 200, schema=None, validator=AcceptAll(), max_result_chars=100)
    assert not big.accepted and big.check == CHECK_SIZE


# ------------------------------------------------------------- size limit


def test_result_at_limit_passes():
    checked = run("x" * 100, schema=None, max_result_chars=100)
    assert checked.accepted and checked.result_chars == 100


def test_oversized_text_rejected_by_default():
    checked = run("x" * 101, schema=None, max_result_chars=100)
    assert not checked.accepted and checked.check == CHECK_SIZE
    assert "101 characters, over the limit of 100" in checked.reason


def test_oversized_structure_rejected():
    result = {"rows": ["abcdefghij"] * 20}
    checked = run(result, schema=None, max_result_chars=100)
    assert not checked.accepted and checked.check == CHECK_SIZE


def test_dict_keys_count_towards_size():
    checked = run({"k" * 60: "v" * 50}, schema=None, max_result_chars=100)
    assert not checked.accepted


def test_truncate_shortens_text_with_marker():
    text = "a" * 1000
    checked = run(text, schema=None, max_result_chars=200, oversized_results="truncate")
    assert checked.accepted and checked.truncated
    assert len(checked.value) <= 200
    assert checked.value.startswith("a" * 100)
    assert checked.value.endswith("of 1000 characters]")
    assert "[SAW: result truncated to" in checked.value
    assert checked.original_chars == 1000


def test_truncate_does_not_apply_to_structured_results():
    checked = run(
        {"rows": ["x" * 50] * 5}, schema=None, max_result_chars=100, oversized_results="truncate"
    )
    assert not checked.accepted
    assert "only plain-text results can be truncated" in checked.reason


def test_truncate_with_limit_too_small_for_marker_rejects():
    checked = run("x" * 50, schema=None, max_result_chars=10, oversized_results="truncate")
    assert not checked.accepted and checked.check == CHECK_SIZE


def test_text_within_limit_is_not_truncated_even_in_truncate_mode():
    checked = run("short", schema=None, oversized_results="truncate")
    assert checked.accepted and not checked.truncated and checked.value == "short"


def test_default_limit_is_100000():
    assert run("x" * 100_000, schema=None).accepted
    assert not run("x" * 100_001, schema=None).accepted


# --------------------------------------------------------------- counting


def test_count_chars_counts_strings_bytes_and_keys():
    assert count_chars({"ab": ["cde", b"fg", 12345, None]}, 10_000) == 7


def test_count_chars_stops_early():
    huge = ["x" * 1000] * 1000
    assert 100 < count_chars(huge, 100) <= 1100


def test_count_chars_handles_cycles():
    loop: list = ["abc"]
    loop.append(loop)
    assert count_chars(loop, 10_000) == 3

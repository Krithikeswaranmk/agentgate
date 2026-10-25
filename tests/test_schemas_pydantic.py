"""Optional Pydantic v2 schema support."""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

import pytest

from saw import ArgumentValidationError, ConfigurationError
from saw.schemas import PydanticSchema, compile_schema

pydantic = pytest.importorskip("pydantic")
from pydantic import BaseModel, ConfigDict, Field, field_validator  # noqa: E402


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    limit: int = 10
    ratio: float = 1.0


class Inner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


class Outer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inner: Inner
    others: list[Inner] = []


class Checked(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str

    @field_validator("path")
    @classmethod
    def no_parent(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("path must not contain '..'")
        return v


class Lax(BaseModel):
    path: str


class LaxInner(BaseModel):
    city: str


class OuterWithLaxInner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inner: Optional[LaxInner] = None


def test_pydantic_model_detected():
    assert isinstance(compile_schema(Strict), PydanticSchema)


def test_valid_arguments():
    assert compile_schema(Strict).validate({"path": "a", "ratio": 2}) == {
        "path": "a",
        "limit": 10,
        "ratio": 2.0,
    }


def test_nested_models_from_dicts():
    kwargs = compile_schema(Outer).validate({"inner": {"city": "c"}, "others": [{"city": "d"}]})
    assert kwargs["inner"] == Inner(city="c")
    assert kwargs["others"] == [Inner(city="d")]


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"path": 1}, "path: Input should be a valid string"),
        ({"path": "a", "limit": "10"}, "limit: Input should be a valid integer"),
        ({"path": "a", "ratio": "1.5"}, "ratio: Input should be a valid number"),
        ({}, "path: Field required"),
        ({"path": "a", "x": 1}, "x: Extra inputs are not permitted"),
    ],
)
def test_strict_mode_rejections(arguments, message):
    with pytest.raises(ArgumentValidationError, match=message):
        compile_schema(Strict).validate(arguments)


def test_nested_error_location():
    with pytest.raises(ArgumentValidationError, match="inner.city"):
        compile_schema(Outer).validate({"inner": {"city": 1}})


def test_custom_validator_rejects():
    with pytest.raises(ArgumentValidationError, match="must not contain"):
        compile_schema(Checked).validate({"path": "../x"})


def test_error_does_not_leak_values_or_chain_pydantic_error():
    secret = "sk-SECRET-TOKEN-123"
    with pytest.raises(ArgumentValidationError) as info:
        compile_schema(Strict).validate({"path": "a", "limit": secret})
    assert secret not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


def test_model_without_extra_forbid_rejected():
    with pytest.raises(ConfigurationError, match="extra='forbid'"):
        compile_schema(Lax)


def test_nested_model_without_extra_forbid_rejected():
    with pytest.raises(ConfigurationError, match="LaxInner"):
        compile_schema(OuterWithLaxInner)


def test_pydantic_v1_style_model_rejected():
    v1 = pytest.importorskip("pydantic.v1")

    class Old(v1.BaseModel):
        path: str

    with pytest.raises(ConfigurationError):
        compile_schema(Old)


def test_saw_does_not_import_pydantic():
    code = "import sys, saw; assert 'pydantic' not in sys.modules, 'pydantic was imported'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---- result schemas -------------------------------------------------------


class RunResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    failures: int = Field(ge=0)
    by_file: dict[str, int] = {}


def _check(result):
    from saw import AgentConfig, DefaultResultValidator, ToolCall
    from saw.results import check_result

    return check_result(
        result,
        call=ToolCall("run_tests"),
        config=AgentConfig(agent_id="coding-agent"),
        validator=DefaultResultValidator(),
        result_schema=compile_schema(RunResultModel),
    )


def test_pydantic_result_spec_example_rejected():
    checked = _check({"passed": "yes", "failures": -1})
    assert not checked.accepted
    assert "passed" in checked.reason and "failures" in checked.reason


def test_pydantic_result_instance_converted():
    checked = _check(RunResultModel(passed=True, failures=0))
    assert checked.accepted and checked.converted
    assert checked.value == {"passed": True, "failures": 0, "by_file": {}}


def test_pydantic_result_keys_redacted():
    checked = _check({"passed": True, "failures": 0, "alice@example.com": 1})
    assert not checked.accepted and "alice" not in checked.reason
    checked = _check({"passed": True, "failures": 0, "by_file": {"secret_path.py": "x"}})
    assert not checked.accepted and "secret_path" not in checked.reason
    assert "by_file" in checked.reason


def test_pydantic_argument_errors_still_name_unknown_keys():
    with pytest.raises(ArgumentValidationError, match="extra_arg"):
        compile_schema(RunResultModel).validate({"passed": True, "failures": 0, "extra_arg": 1})

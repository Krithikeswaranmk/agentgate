"""Built-in dataclass schema validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import pytest

from saw import ArgumentValidationError, ConfigurationError
from saw.schemas import MAX_REPORTED_ERRORS, DataclassSchema, compile_schema


@dataclass
class Simple:
    path: str
    limit: int = 10
    verbose: bool = False


@dataclass
class Numbers:
    ratio: float


@dataclass
class Address:
    city: str
    zip_code: Optional[str] = None


@dataclass
class Contact:
    name: str
    address: Address
    tags: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    mode: Literal["fast", "safe"] = "safe"
    note: str | None = None
    either: Union[int, str] = 0


@dataclass
class WithCheck:
    path: str

    def __post_init__(self):
        if ".." in self.path:
            raise ValueError("path must not contain '..'")


@dataclass
class Normalising:
    name: str

    def __post_init__(self):
        self.name = self.name.strip().lower()


@dataclass
class Empty:
    pass


# ---- valid input ---------------------------------------------------------


def test_valid_arguments_with_defaults():
    assert compile_schema(Simple).validate({"path": "a.txt"}) == {
        "path": "a.txt",
        "limit": 10,
        "verbose": False,
    }


def test_nested_and_container_types():
    kwargs = compile_schema(Contact).validate(
        {
            "name": "Ana",
            "address": {"city": "Chennai"},
            "tags": ["a", "b"],
            "scores": {"x": 1},
            "mode": "fast",
            "note": None,
            "either": "s",
        }
    )
    assert kwargs["address"] == Address(city="Chennai")
    assert kwargs["tags"] == ["a", "b"]
    assert kwargs["mode"] == "fast"
    assert kwargs["either"] == "s"


def test_int_accepted_for_float_and_converted():
    kwargs = compile_schema(Numbers).validate({"ratio": 2})
    assert kwargs["ratio"] == 2.0 and isinstance(kwargs["ratio"], float)


def test_post_init_normalisation_is_passed_to_tool():
    assert compile_schema(Normalising).validate({"name": "  ADA "}) == {"name": "ada"}


def test_empty_schema_accepts_no_arguments():
    assert compile_schema(Empty).validate({}) == {}
    with pytest.raises(ArgumentValidationError, match="unknown field 'x'"):
        compile_schema(Empty).validate({"x": 1})


# ---- strictness ----------------------------------------------------------


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"path": 3}, "path: expected string, got int"),
        ({"path": "a", "limit": "10"}, "limit: expected integer, got str"),
        ({"path": "a", "limit": 1.0}, "limit: expected integer, got float"),
        ({"path": "a", "limit": True}, "limit: expected integer, got bool"),
        ({"path": "a", "verbose": 1}, "verbose: expected boolean, got int"),
        ({"path": "a", "verbose": "true"}, "verbose: expected boolean, got str"),
        ({}, "path: missing required field"),
        ({"path": "a", "extra": 1}, "unknown field 'extra'"),
        ({"path": None}, "path: expected string, got NoneType"),
    ],
)
def test_strict_rejections(arguments, message):
    with pytest.raises(ArgumentValidationError, match=message):
        compile_schema(Simple).validate(arguments)


@pytest.mark.parametrize("value", [True, "1.5", None, math.nan, math.inf])
def test_float_rejections(value):
    with pytest.raises(ArgumentValidationError, match="ratio"):
        compile_schema(Numbers).validate({"ratio": value})


def test_literal_does_not_confuse_bool_and_int():
    @dataclass
    class L:
        v: Literal[1, "a"]

    schema = DataclassSchema(L)
    assert schema.validate({"v": 1}) == {"v": 1}
    with pytest.raises(ArgumentValidationError, match="expected one of"):
        schema.validate({"v": True})


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"name": "a", "address": "Chennai"}, "address: expected object, got str"),
        ({"name": "a", "address": {}}, "address.city: missing required field"),
        ({"name": "a", "address": {"city": "c", "x": 1}}, "address: unknown field 'x'"),
        ({"name": "a", "address": {"city": "c"}, "tags": "a"}, "tags: expected array, got str"),
        ({"name": "a", "address": {"city": "c"}, "tags": ["a", 2]}, r"tags\[1\]: expected string"),
        ({"name": "a", "address": {"city": "c"}, "tags": ("a",)}, "tags: expected array, got tuple"),
        ({"name": "a", "address": {"city": "c"}, "scores": {"x": "1"}}, r"scores\['x'\]: expected integer"),
        ({"name": "a", "address": {"city": "c"}, "mode": "slow"}, "mode: expected one of"),
        ({"name": "a", "address": {"city": "c"}, "either": 1.5}, "either: expected integer or string"),
    ],
)
def test_nested_rejections(arguments, message):
    with pytest.raises(ArgumentValidationError, match=message):
        compile_schema(Contact).validate(arguments)


def test_non_string_keys_in_dict_field_rejected():
    with pytest.raises(ArgumentValidationError, match="keys must be strings"):
        compile_schema(Contact).validate(
            {"name": "a", "address": {"city": "c"}, "scores": {1: 1}}
        )


def test_post_init_check_rejects_with_developer_message():
    with pytest.raises(ArgumentValidationError, match=r"rejected by WithCheck: path must not contain '\.\.'"):
        compile_schema(WithCheck).validate({"path": "../etc/passwd"})


def test_all_problems_reported_together():
    with pytest.raises(ArgumentValidationError) as info:
        compile_schema(Simple).validate({"limit": "x", "verbose": 0, "extra": 1})
    errors = info.value.errors
    assert any("path: missing" in e for e in errors)
    assert any("limit:" in e for e in errors)
    assert any("verbose:" in e for e in errors)
    assert any("unknown field 'extra'" in e for e in errors)


def test_error_count_is_capped():
    @dataclass
    class Many:
        items: list[int]

    with pytest.raises(ArgumentValidationError) as info:
        DataclassSchema(Many).validate({"items": ["x"] * 1000})
    assert len(info.value.errors) <= MAX_REPORTED_ERRORS


def test_error_messages_do_not_contain_values():
    secret = "sk-SECRET-TOKEN-123"
    with pytest.raises(ArgumentValidationError) as info:
        compile_schema(Simple).validate({"path": "a", "limit": secret})
    assert secret not in str(info.value)


def test_long_unknown_field_names_are_shortened():
    with pytest.raises(ArgumentValidationError) as info:
        compile_schema(Simple).validate({"path": "a", "x" * 10_000: 1})
    assert len(str(info.value)) < 300


# ---- registration-time checks --------------------------------------------


def test_instance_instead_of_class_rejected():
    with pytest.raises(ConfigurationError, match="must be a class"):
        compile_schema(Simple(path="a"))


def test_plain_class_rejected():
    class NotADataclass:
        path: str

    with pytest.raises(ConfigurationError, match="dataclass or a Pydantic v2 model"):
        compile_schema(NotADataclass)


@pytest.mark.parametrize(
    "annotation, message",
    [
        ("object", "unsupported type"),
        ("bytes", "unsupported type"),
        ("list", r"list\[X\]"),
        ("dict[int, str]", r"dict\[str, X\]"),
        ("dict", r"dict\[str, X\]"),
        ("tuple[int, int]", "unsupported type"),
        ("set[str]", "unsupported type"),
    ],
)
def test_unsupported_field_types_rejected(annotation, message):
    namespace: dict = {}
    exec(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        f"class S:\n    x: {annotation}\n",
        namespace,
    )
    with pytest.raises(ConfigurationError, match=message):
        compile_schema(namespace["S"])


def test_any_rejected():
    @dataclass
    class S:
        x: Any

    with pytest.raises(ConfigurationError, match="unsupported type"):
        DataclassSchema(S)


@dataclass
class Node:
    value: int
    child: Optional["Node"] = None


def test_recursive_schema_rejected():
    with pytest.raises(ConfigurationError, match="refers to itself"):
        compile_schema(Node)


def test_unresolvable_annotation_rejected():
    @dataclass
    class S:
        x: "DoesNotExist"  # noqa: F821

    with pytest.raises(ConfigurationError, match="cannot resolve"):
        DataclassSchema(S)


def test_init_false_fields_are_not_inputs():
    @dataclass
    class S:
        x: int
        computed: int = field(init=False, default=0)

    schema = DataclassSchema(S)
    assert schema.field_names == ("x",)
    with pytest.raises(ArgumentValidationError, match="unknown field 'computed'"):
        schema.validate({"x": 1, "computed": 5})

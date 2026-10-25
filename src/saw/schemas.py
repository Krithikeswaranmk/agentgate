"""Argument schemas for registered tools.

Two kinds of schema are supported:

* **Dataclasses** (built in, no dependencies). Validation is strict: no type
  conversion except ``int`` to ``float``, unknown fields are rejected, and
  missing required fields are reported.
* **Pydantic v2 models** (optional). Used only if Pydantic is already
  imported by your application; SAW never imports it itself. Models are
  validated in strict mode and must set ``extra="forbid"``.

Both kinds are checked when the tool is registered, so an unsupported schema
fails at setup time rather than on the first call.

Error messages name fields and types but never include argument values, so
they are safe to log and to return to the model. When validating a tool
*result* (``redact_keys=True``), messages also leave out any key taken from
the data, such as an unknown field name or a key of a ``dict[str, X]`` field,
because keys in a result can be data too. Field names declared in the schema
are always shown.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import contextvars
import copy
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from saw.exceptions import ArgumentValidationError, ConfigurationError

#: At most this many problems are listed in one validation error.
MAX_REPORTED_ERRORS = 20

_INVALID = object()

# Set while validating a result, so messages leave out keys taken from data.
_redact_keys: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "saw_redact_keys", default=False
)


def _key_label(key: str) -> str:
    return "<key>" if _redact_keys.get() else repr(_short(key))
_Check = Callable[[Any, str, list], Any]


def _short(text: str, limit: int = 64) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _where(path: str) -> str:
    return path or "arguments"


def _validation_error(errors: list[str]) -> ArgumentValidationError:
    shown = errors[:MAX_REPORTED_ERRORS]
    message = "; ".join(shown)
    if len(errors) > len(shown):
        message += f"; and {len(errors) - len(shown)} more"
    exc = ArgumentValidationError(message)
    exc.errors = tuple(shown)
    return exc


class ArgumentSchema(ABC):
    """A tool's compiled argument schema. Created by :func:`compile_schema`."""

    #: The class the developer passed as ``schema``.
    source: type

    @property
    @abstractmethod
    def field_names(self) -> tuple[str, ...]:
        """Names of the keyword arguments the tool function receives."""

    @abstractmethod
    def json_schema(self) -> dict[str, Any]:
        """A JSON Schema (object) describing valid input, for LLM tool definitions."""

    @abstractmethod
    def validate(
        self, arguments: Mapping[str, Any], *, redact_keys: bool = False
    ) -> dict[str, Any]:
        """Return the validated fields as keyword arguments.

        Args:
            arguments: The data to validate.
            redact_keys: Leave keys taken from the data out of error
                messages. Used for tool results.

        Raises:
            ArgumentValidationError: If ``arguments`` does not match the schema.
        """


def compile_schema(schema: object) -> ArgumentSchema:
    """Check ``schema`` and return a validator for it.

    Raises:
        ConfigurationError: If ``schema`` is not a supported class or uses
            unsupported field types.
    """
    if not isinstance(schema, type):
        raise ConfigurationError(
            "schema must be a class (a dataclass or a Pydantic v2 model), "
            f"got an instance of {type(schema).__name__}"
        )
    pydantic = sys.modules.get("pydantic")
    base_model = getattr(pydantic, "BaseModel", None) if pydantic else None
    if base_model is not None and issubclass(schema, base_model):
        return PydanticSchema(schema, pydantic)
    if dataclasses.is_dataclass(schema):
        return DataclassSchema(schema)
    raise ConfigurationError(
        f"schema {schema.__name__} must be a dataclass or a Pydantic v2 model"
    )


# ---------------------------------------------------------------------------
# Built-in dataclass schemas
# ---------------------------------------------------------------------------


def _check_str(value: Any, path: str, errors: list) -> Any:
    if isinstance(value, str):
        return value
    errors.append(f"{_where(path)}: expected string, got {type(value).__name__}")
    return _INVALID


def _check_bool(value: Any, path: str, errors: list) -> Any:
    if isinstance(value, bool):
        return value
    errors.append(f"{_where(path)}: expected boolean, got {type(value).__name__}")
    return _INVALID


def _check_int(value: Any, path: str, errors: list) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    errors.append(f"{_where(path)}: expected integer, got {type(value).__name__}")
    return _INVALID


def _check_float(value: Any, path: str, errors: list) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{_where(path)}: expected a finite number")
            return _INVALID
        return float(value)
    errors.append(f"{_where(path)}: expected number, got {type(value).__name__}")
    return _INVALID


def _check_none(value: Any, path: str, errors: list) -> Any:
    if value is None:
        return None
    errors.append(f"{_where(path)}: expected null, got {type(value).__name__}")
    return _INVALID


def _literal_check(allowed: tuple) -> _Check:
    def check(value: Any, path: str, errors: list) -> Any:
        # Compare type as well as value so True does not match Literal[1].
        for option in allowed:
            if type(value) is type(option) and value == option:
                return value
        errors.append(
            f"{_where(path)}: expected one of {list(allowed)!r}"
        )
        return _INVALID

    return check


def _union_check(options: tuple[_Check, ...], label: str) -> _Check:
    def check(value: Any, path: str, errors: list) -> Any:
        for option in options:
            result = option(value, path, [])
            if result is not _INVALID:
                return result
        errors.append(
            f"{_where(path)}: expected {label}, got {type(value).__name__}"
        )
        return _INVALID

    return check


def _list_check(item: _Check) -> _Check:
    def check(value: Any, path: str, errors: list) -> Any:
        if not isinstance(value, list):
            errors.append(f"{_where(path)}: expected array, got {type(value).__name__}")
            return _INVALID
        before = len(errors)
        out = []
        for i, element in enumerate(value):
            if len(errors) > MAX_REPORTED_ERRORS:
                break
            out.append(item(element, f"{_where(path)}[{i}]", errors))
        return _INVALID if len(errors) > before else out

    return check


def _dict_check(item: _Check) -> _Check:
    def check(value: Any, path: str, errors: list) -> Any:
        if not isinstance(value, Mapping):
            errors.append(f"{_where(path)}: expected object, got {type(value).__name__}")
            return _INVALID
        before = len(errors)
        out = {}
        for key, element in value.items():
            if len(errors) > MAX_REPORTED_ERRORS:
                break
            if not isinstance(key, str):
                errors.append(f"{_where(path)}: keys must be strings, got {type(key).__name__}")
                continue
            out[key] = item(element, f"{_where(path)}[{_key_label(key)}]", errors)
        return _INVALID if len(errors) > before else out

    return check


def _label(tp: Any) -> str:
    names = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}
    if tp in names:
        return names[tp]
    return getattr(tp, "__name__", None) or repr(tp).replace("typing.", "")


_UNION_TYPES: tuple = (Union,) + ((types.UnionType,) if hasattr(types, "UnionType") else ())


def _compile_type(tp: Any, seen: frozenset, owner: str) -> _Check:
    if tp is str:
        return _check_str
    if tp is bool:
        return _check_bool
    if tp is int:
        return _check_int
    if tp is float:
        return _check_float
    if tp is None or tp is type(None):
        return _check_none

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Literal:
        for option in args:
            if not isinstance(option, (str, int, bool, type(None))):
                raise ConfigurationError(
                    f"{owner}: Literal values must be str, int, bool or None"
                )
        return _literal_check(args)
    if origin in _UNION_TYPES:
        options = tuple(_compile_type(a, seen, owner) for a in args)
        label = " or ".join(_label(a) for a in args)
        return _union_check(options, label)
    if origin is list or tp is list:
        if len(args) != 1:
            raise ConfigurationError(f"{owner}: use list[X] to give the item type")
        return _list_check(_compile_type(args[0], seen, owner))
    if origin is dict or tp is dict:
        if len(args) != 2 or args[0] is not str:
            raise ConfigurationError(f"{owner}: use dict[str, X]; keys must be strings")
        return _dict_check(_compile_type(args[1], seen, owner))
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _DataclassSpec(tp, seen).build

    raise ConfigurationError(
        f"{owner}: unsupported type {_label(tp)}. Supported: str, int, float, "
        "bool, None, Optional/Union, Literal, list[X], dict[str, X] and nested "
        "dataclasses"
    )


def _json_schema_for(tp: Any) -> dict[str, Any]:
    """JSON Schema for a type already accepted by ``_compile_type``."""
    simple = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}
    if tp is None:
        return {"type": "null"}
    if tp in simple:
        return {"type": simple[tp]}
    origin, args = get_origin(tp), get_args(tp)
    if origin is Literal:
        return {"enum": list(args)}
    if origin in _UNION_TYPES:
        return {"anyOf": [_json_schema_for(a) for a in args]}
    if origin is list:
        return {"type": "array", "items": _json_schema_for(args[0])}
    if origin is dict:
        return {"type": "object", "additionalProperties": _json_schema_for(args[1])}
    return _DataclassSpec(tp, frozenset()).json_schema


class _DataclassSpec:
    def __init__(self, cls: type, seen: frozenset) -> None:
        if cls in seen:
            raise ConfigurationError(
                f"schema {cls.__name__} refers to itself; recursive schemas are not supported"
            )
        seen = seen | {cls}
        try:
            hints = get_type_hints(cls)
        except Exception as exc:
            raise ConfigurationError(
                f"cannot resolve type annotations of {cls.__name__}: {exc}"
            ) from exc

        self.cls = cls
        self.fields: list[tuple[str, bool, _Check]] = []
        for f in dataclasses.fields(cls):
            if not f.init:
                continue
            required = (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
            )
            check = _compile_type(hints[f.name], seen, f"{cls.__name__}.{f.name}")
            self.fields.append((f.name, required, check))
        self.names = frozenset(name for name, _, _ in self.fields)
        # Types were checked above, so the JSON Schema below cannot fail.
        self.json_schema = {
            "type": "object",
            "properties": {
                f.name: _json_schema_for(hints[f.name])
                for f in dataclasses.fields(cls) if f.init
            },
            "required": [name for name, required, _ in self.fields if required],
            "additionalProperties": False,
        }

    def build(self, value: Any, path: str, errors: list) -> Any:
        if not isinstance(value, Mapping):
            errors.append(f"{_where(path)}: expected object, got {type(value).__name__}")
            return _INVALID
        before = len(errors)
        for key in value:
            if not isinstance(key, str):
                errors.append(f"{_where(path)}: field names must be strings, got {type(key).__name__}")
            elif key not in self.names:
                errors.append(
                    f"{_where(path)}: unknown field"
                    if _redact_keys.get()
                    else f"{_where(path)}: unknown field {_short(key)!r}"
                )

        kwargs = {}
        for name, required, check in self.fields:
            child = f"{path}.{name}" if path else name
            if name in value:
                result = check(value[name], child, errors)
                if result is not _INVALID:
                    kwargs[name] = result
            elif required:
                errors.append(f"{child}: missing required field")
        if len(errors) > before:
            return _INVALID

        try:
            return self.cls(**kwargs)
        except Exception as exc:  # checks in __post_init__
            errors.append(
                f"{_where(path)}: rejected by {self.cls.__name__}: "
                f"{_short(str(exc) or type(exc).__name__, 200)}"
            )
            return _INVALID


class DataclassSchema(ArgumentSchema):
    """Strict validator for a dataclass schema.

    Custom checks can go in the dataclass's ``__post_init__``; any exception
    raised there rejects the arguments.
    """

    def __init__(self, cls: type) -> None:
        self.source = cls
        self._spec = _DataclassSpec(cls, frozenset())
        self._names = tuple(name for name, _, _ in self._spec.fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        return self._names

    def json_schema(self) -> dict[str, Any]:
        return copy.deepcopy(self._spec.json_schema)

    def validate(
        self, arguments: Mapping[str, Any], *, redact_keys: bool = False
    ) -> dict[str, Any]:
        errors: list[str] = []
        token = _redact_keys.set(redact_keys)
        try:
            instance = self._spec.build(arguments, "", errors)
        finally:
            _redact_keys.reset(token)
        if instance is _INVALID:
            raise _validation_error(errors)
        return {name: getattr(instance, name) for name in self._names}


# ---------------------------------------------------------------------------
# Optional Pydantic v2 support
# ---------------------------------------------------------------------------


class PydanticSchema(ArgumentSchema):
    """Validator for a Pydantic v2 model, using strict mode."""

    def __init__(self, model: type, pydantic: types.ModuleType) -> None:
        if not hasattr(model, "model_validate") or not hasattr(model, "model_fields"):
            raise ConfigurationError(
                f"schema {model.__name__}: only Pydantic v2 models are supported"
            )
        self._base_model = pydantic.BaseModel
        self._validation_error = pydantic.ValidationError
        self._declared_names: set[str] = set()
        self._require_forbid_extra(model, set())
        self.source = model
        self._names = tuple(model.model_fields)

    def _is_model(self, tp: Any) -> bool:
        # On Python 3.10, isinstance(list[X], type) is True but issubclass()
        # on it raises, so exclude parameterised generics first.
        return (
            isinstance(tp, type)
            and get_origin(tp) is None
            and issubclass(tp, self._base_model)
        )

    def _require_forbid_extra(self, tp: Any, seen: set) -> None:
        if self._is_model(tp):
            if tp in seen:
                return
            seen.add(tp)
            for field_name, field_info in tp.model_fields.items():
                self._declared_names.add(field_name)
                if field_info.alias:
                    self._declared_names.add(field_info.alias)
            if tp.model_config.get("extra") != "forbid":
                raise ConfigurationError(
                    f"Pydantic model {tp.__name__} must set "
                    "model_config = ConfigDict(extra='forbid') so that unknown "
                    "arguments are rejected instead of silently ignored"
                )
            for field in tp.model_fields.values():
                self._require_forbid_extra(field.annotation, seen)
            return
        for arg in get_args(tp):
            self._require_forbid_extra(arg, seen)

    @property
    def field_names(self) -> tuple[str, ...]:
        return self._names

    def json_schema(self) -> dict[str, Any]:
        return self.source.model_json_schema()

    def _location(self, loc: tuple, redact_keys: bool) -> str:
        parts = []
        for part in loc:
            if redact_keys and isinstance(part, str) and part not in self._declared_names:
                part = "<key>"
            parts.append(str(part))
        return _short(_where(".".join(parts)), 128)

    def validate(
        self, arguments: Mapping[str, Any], *, redact_keys: bool = False
    ) -> dict[str, Any]:
        try:
            instance = self.source.model_validate(dict(arguments), strict=True)
        except self._validation_error as exc:
            errors = []
            for err in exc.errors():
                where = self._location(tuple(err.get("loc", ())), redact_keys)
                errors.append(f"{where}: {err.get('msg', 'invalid value')}")
            # Raised "from None": Pydantic's own message includes input values.
            raise _validation_error(errors) from None
        except Exception as exc:
            raise _validation_error(
                [f"arguments: rejected by {self.source.__name__}: {type(exc).__name__}"]
            ) from None
        return {name: getattr(instance, name) for name in self._names}


__all__ = [
    "ArgumentSchema",
    "DataclassSchema",
    "PydanticSchema",
    "compile_schema",
    "MAX_REPORTED_ERRORS",
]

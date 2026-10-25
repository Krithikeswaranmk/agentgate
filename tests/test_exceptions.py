import pytest

import saw
from saw import (
    ApprovalRequiredError,
    ArgumentValidationError,
    AuditError,
    ConfigurationError,
    PermissionDeniedError,
    ResultValidationError,
    SAWError,
    ToolExecutionError,
    ToolNotRegisteredError,
)

ALL = [
    ConfigurationError,
    ToolNotRegisteredError,
    ArgumentValidationError,
    PermissionDeniedError,
    ApprovalRequiredError,
    ResultValidationError,
    ToolExecutionError,
]


@pytest.mark.parametrize("exc", ALL)
def test_all_errors_derive_from_saw_error(exc):
    assert issubclass(exc, SAWError)
    with pytest.raises(SAWError):
        raise exc("x")


def test_approval_required_is_not_a_denial():
    assert not issubclass(ApprovalRequiredError, PermissionDeniedError)


def test_saw_errors_do_not_inherit_builtin_value_or_type_error():
    # Keeps a broad `except ValueError` in user code from swallowing security errors.
    for exc in ALL:
        assert not issubclass(exc, (ValueError, TypeError))


def test_all_exceptions_exported():
    for exc in ALL + [SAWError]:
        assert getattr(saw, exc.__name__) is exc


def test_audit_error_attributes():
    err = AuditError("x", tool_executed=True)
    assert isinstance(err, SAWError)
    assert err.tool_executed is True and err.outcome is None
    assert not issubclass(AuditError, (ValueError, TypeError))
    assert saw.AuditError is AuditError

import pytest

from saw import (
    ApprovalRequiredError,
    ArgumentValidationError,
    CallStatus,
    PermissionDeniedError,
    ToolCallOutcome,
    ToolExecutionError,
    ToolNotRegisteredError,
)


def test_executed_outcome():
    outcome = ToolCallOutcome(CallStatus.EXECUTED, "t", "c1", "executed", value=42)
    assert outcome.ok and outcome.value == 42
    assert outcome.status is CallStatus.EXECUTED


def test_executed_none_value_is_returned():
    outcome = ToolCallOutcome(CallStatus.EXECUTED, "t", "c1", "executed")
    assert outcome.value is None


@pytest.mark.parametrize(
    "status, exc_type",
    [
        (CallStatus.DENIED, PermissionDeniedError),
        (CallStatus.APPROVAL_REQUIRED, ApprovalRequiredError),
        (CallStatus.UNKNOWN_TOOL, ToolNotRegisteredError),
        (CallStatus.INVALID_ARGUMENTS, ArgumentValidationError),
        (CallStatus.FAILED, ToolExecutionError),
    ],
)
def test_value_raises_for_every_non_executed_status(status, exc_type):
    extra = {"approval_request": object()} if status is CallStatus.APPROVAL_REQUIRED else {}
    outcome = ToolCallOutcome(status, "t", "c1", "because", **extra)
    assert not outcome.ok
    with pytest.raises(exc_type, match="because") as info:
        outcome.value
    assert info.value.outcome is outcome


def test_failed_value_chains_original_error():
    original = ValueError("x")
    outcome = ToolCallOutcome(CallStatus.FAILED, "t", "c1", "failed", error=original)
    with pytest.raises(ToolExecutionError) as info:
        outcome.value
    assert info.value.__cause__ is original


def test_invalid_arguments_value_carries_error_list():
    original = ArgumentValidationError("a; b")
    original.errors = ("a", "b")
    outcome = ToolCallOutcome(CallStatus.INVALID_ARGUMENTS, "t", "c1", "bad", error=original)
    with pytest.raises(ArgumentValidationError) as info:
        outcome.value
    assert info.value.errors == ("a", "b")


def test_blocked_outcome_cannot_carry_value():
    with pytest.raises(TypeError, match="only an executed call"):
        ToolCallOutcome(CallStatus.DENIED, "t", "c1", "no", value="leak")


def test_status_must_be_enum():
    with pytest.raises(TypeError):
        ToolCallOutcome("executed", "t", "c1", "x")  # type: ignore[arg-type]


def test_outcome_is_read_only():
    outcome = ToolCallOutcome(CallStatus.DENIED, "t", "c1", "no")
    for attr in ("status", "ok", "value", "reason", "tool_name", "call_id", "error"):
        with pytest.raises(AttributeError):
            setattr(outcome, attr, None)
    with pytest.raises(AttributeError):
        outcome.new_attribute = 1


def test_repr_does_not_include_value():
    outcome = ToolCallOutcome(CallStatus.EXECUTED, "t", "c1", "executed", value="secret")
    assert "secret" not in repr(outcome)


def test_result_rejected_value_raises_result_validation_error():
    from saw import ResultValidationError

    outcome = ToolCallOutcome(CallStatus.RESULT_REJECTED, "t", "c1", "bad result")
    with pytest.raises(ResultValidationError, match="bad result"):
        outcome.value


def test_truncated_flag_defaults_false():
    assert ToolCallOutcome(CallStatus.EXECUTED, "t", "c", "executed").truncated is False
    assert ToolCallOutcome(CallStatus.EXECUTED, "t", "c", "executed", truncated=True).truncated


def test_approval_request_only_with_approval_required():
    with pytest.raises(TypeError, match="approval_request"):
        ToolCallOutcome(CallStatus.APPROVAL_REQUIRED, "t", "c", "needs approval")
    with pytest.raises(TypeError, match="approval_request"):
        ToolCallOutcome(CallStatus.DENIED, "t", "c", "no", approval_request=object())


def test_for_model_messages():
    ok = ToolCallOutcome(CallStatus.EXECUTED, "t", "c", "executed", value={"a": 1})
    assert ok.for_model() == {"status": "executed", "result": {"a": 1}}
    cut = ToolCallOutcome(CallStatus.EXECUTED, "t", "c", "executed", value="x", truncated=True)
    assert cut.for_model()["truncated"] is True
    pending = ToolCallOutcome(
        CallStatus.APPROVAL_REQUIRED, "t", "c", "needs approval", approval_request=object()
    )
    message = pending.for_model()
    assert message["status"] == "approval_required"
    assert "NOT been performed" in message["note"]
    assert "result" not in message
    failed = ToolCallOutcome(CallStatus.FAILED, "t", "c", "tool failed", error=ValueError("secret"))
    assert "secret" not in repr(failed.for_model())
    assert "result" not in failed.for_model()

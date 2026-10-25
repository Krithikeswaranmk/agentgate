import pytest

from saw import ArgumentValidationError, ToolCall


def test_tool_call_defaults():
    call = ToolCall("web_search")
    assert call.tool_name == "web_search"
    assert dict(call.arguments) == {}
    assert len(call.call_id) == 32


def test_arguments_copied_and_read_only():
    args = {"query": "saw"}
    call = ToolCall("web_search", args)
    args["query"] = "changed"
    assert call.arguments["query"] == "saw"
    with pytest.raises(TypeError):
        call.arguments["query"] = "x"  # type: ignore[index]


def test_call_ids_unique():
    assert ToolCall("t").call_id != ToolCall("t").call_id


@pytest.mark.parametrize("name", ["", None, 5])
def test_invalid_tool_name_rejected(name):
    with pytest.raises(ArgumentValidationError, match="tool_name"):
        ToolCall(name)  # type: ignore[arg-type]


@pytest.mark.parametrize("args", [["q"], "q=1", None])
def test_non_mapping_arguments_rejected(args):
    with pytest.raises(ArgumentValidationError, match="mapping"):
        ToolCall("t", args)  # type: ignore[arg-type]


def test_non_string_argument_names_rejected():
    with pytest.raises(ArgumentValidationError, match="argument names"):
        ToolCall("t", {1: "x"})  # type: ignore[dict-item]

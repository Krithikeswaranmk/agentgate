"""Provider adapters: message conversion, and end-to-end runs through the real
Anthropic and OpenAI SDKs against a mocked HTTP transport (no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from saw import SAW, AgentConfig, ApprovalDesk, CallStatus
from saw.adapters.anthropic import (
    AnthropicModel,
    from_anthropic_response,
    to_anthropic_messages,
    to_anthropic_tools,
)
from saw.adapters.openai import (
    OpenAIChatModel,
    from_openai_response,
    to_openai_messages,
    to_openai_tools,
)
from saw.agent import FinalAnswer, ModelTurn, run_agent
from conftest import RecordingLogger


def http_module():
    """The HTTP library the installed SDKs use (httpx2 in newer releases, httpx before)."""
    for name in ("httpx2", "httpx"):
        try:
            return __import__(name)
        except ImportError:
            continue
    pytest.skip("neither httpx2 nor httpx is installed")

NEUTRAL = [
    {"role": "user", "content": "check monday"},
    {"role": "assistant", "content": "Looking.", "tool_uses": [
        {"id": "u1", "name": "calendar", "arguments": {"day": "monday"}},
        {"id": "u2", "name": "calendar", "arguments": "{broken"},
    ]},
    {"role": "tool_results", "results": [
        {"tool_use_id": "u1", "name": "calendar", "content": {"status": "executed", "result": ["9:00"]}, "is_error": False},
        {"tool_use_id": "u2", "name": "calendar", "content": {"status": "invalid_arguments", "error": "bad"}, "is_error": True},
    ]},
    {"role": "assistant", "content": "Done.", "tool_uses": []},
]
TOOLS = [{"name": "calendar", "description": "Events.", "parameters": {"type": "object", "properties": {}}}]


# ------------------------------------------------------ pure conversions


def test_anthropic_conversions():
    assert to_anthropic_tools(TOOLS) == [
        {"name": "calendar", "description": "Events.", "input_schema": {"type": "object", "properties": {}}}
    ]
    m = to_anthropic_messages(NEUTRAL)
    assert m[0] == {"role": "user", "content": "check monday"}
    assert m[1]["content"][0] == {"type": "text", "text": "Looking."}
    assert m[1]["content"][1] == {"type": "tool_use", "id": "u1", "name": "calendar", "input": {"day": "monday"}}
    assert m[1]["content"][2]["input"] == {}  # malformed arguments are never echoed as input
    results = m[2]["content"]
    assert m[2]["role"] == "user" and [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert json.loads(results[0]["content"]) == {"status": "executed", "result": ["9:00"]}
    assert [r["is_error"] for r in results] == [False, True]
    assert m[3] == {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}


def test_anthropic_response_parsing():
    step = from_anthropic_response({"content": [
        {"type": "text", "text": "Hi"}, {"type": "thinking", "thinking": "..."},
        {"type": "tool_use", "id": "t1", "name": "calendar", "input": {"day": "x"}},
    ]})
    assert isinstance(step, ModelTurn) and step.text == "Hi"
    assert step.tool_uses[0].name == "calendar" and step.tool_uses[0].id == "t1"
    assert from_anthropic_response({"content": [{"type": "text", "text": "bye"}]}) == FinalAnswer("bye")


def test_openai_conversions():
    assert to_openai_tools(TOOLS)[0] == {"type": "function", "function": {
        "name": "calendar", "description": "Events.", "parameters": {"type": "object", "properties": {}}}}
    m = to_openai_messages(NEUTRAL, system="Be careful.")
    assert m[0] == {"role": "system", "content": "Be careful."}
    assert m[2]["tool_calls"][0] == {"id": "u1", "type": "function",
                                     "function": {"name": "calendar", "arguments": '{"day": "monday"}'}}
    assert m[2]["tool_calls"][1]["function"]["arguments"] == "{broken"
    assert m[3]["role"] == "tool" and m[3]["tool_call_id"] == "u1"
    assert json.loads(m[4]["content"])["status"] == "invalid_arguments"
    assert m[5] == {"role": "assistant", "content": "Done."}


def test_openai_response_parsing():
    response = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "calendar", "arguments": '{"day": "x"}'}},
        {"id": "c2", "type": "function", "function": {"name": "calendar", "arguments": "not json"}},
        {"id": "c3", "type": "function", "function": {"name": "calendar", "arguments": ""}},
        {"id": "c4", "type": "custom", "custom": {"name": "x"}},
    ]}}]}
    step = from_openai_response(response)
    assert [u.arguments for u in step.tool_uses] == [{"day": "x"}, "not json", {}]
    assert from_openai_response({"choices": [{"message": {"content": "bye"}}]}) == FinalAnswer("bye")
    assert from_openai_response({"choices": []}) == FinalAnswer("")


# --------------------------------- end to end through the real SDKs, offline


@dataclass
class DayArgs:
    day: str


@dataclass
class EmailArgs:
    to: str
    body: str


def personal_agent(desk):
    sent = []
    security = SAW(
        config=AgentConfig(agent_id="pa", allowed_tools=["calendar", "send_email"], approval_required=["send_email"]),
        logger=RecordingLogger(), approval_desk=desk,
    )
    security.register_tool("calendar", schema=DayArgs, function=lambda day: ["09:00 stand-up"], description="Events.")
    security.register_tool("send_email", schema=EmailArgs, function=lambda to, body: sent.append(to) or "sent",
                           description="Send email.")
    return security, sent


class FakeProvider:
    """Serves canned replies over a mock HTTP transport and records requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def handler(self, request):
        self.requests.append(json.loads(request.content))
        return http_module().Response(200, json=self.replies.pop(0))


def anthropic_reply(*blocks, stop="tool_use"):
    return {"id": "msg", "type": "message", "role": "assistant", "model": "test-model",
            "content": list(blocks), "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


def openai_reply(content=None, calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {"id": "c", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}]}


def test_anthropic_sdk_end_to_end_offline():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = http_module()
    provider = FakeProvider([
        anthropic_reply({"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "t1", "name": "calendar", "input": {"day": "monday"}},
                        {"type": "tool_use", "id": "t2", "name": "send_email", "input": {"to": "u@example.com", "body": "hi"}}),
        anthropic_reply({"type": "tool_use", "id": "t3", "name": "send_email", "input": {"to": "x@example.com", "body": 7}}),
        anthropic_reply({"type": "text", "text": "All done."}, stop="end_turn"),
    ])
    client = anthropic.Anthropic(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(provider.handler)))
    desk = ApprovalDesk()
    security, sent = personal_agent(desk)

    run = run_agent(security, AnthropicModel(client, "test-model", system="Be careful."), "Check monday, email me",
                    approval_desk=desk, approval_handler=lambda r: (False, "not now"))

    assert run.answer == "All done."
    assert [t.outcome.status for t in run.turns] == [CallStatus.EXECUTED, CallStatus.DENIED, CallStatus.INVALID_ARGUMENTS]
    assert sent == []  # denied by the user; the malformed one never ran
    first, second = provider.requests[0], provider.requests[1]
    assert first["system"] == "Be careful."
    assert [t["name"] for t in first["tools"]] == ["calendar", "send_email"]
    assert first["tools"][0]["input_schema"]["required"] == ["day"]
    results = second["messages"][2]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["t1", "t2"]
    assert json.loads(results["content"][0]["content"]) == {"status": "executed", "result": ["09:00 stand-up"]}
    assert results["content"][1]["is_error"] is True
    assert "approval denied" in json.loads(results["content"][1]["content"])["error"]


def test_openai_sdk_end_to_end_offline():
    openai = pytest.importorskip("openai")
    httpx2 = http_module()
    provider = FakeProvider([
        openai_reply(calls=[
            {"id": "c1", "type": "function", "function": {"name": "calendar", "arguments": '{"day": "monday"}'}},
            {"id": "c2", "type": "function", "function": {"name": "send_email", "arguments": '{"to": "u@example.com", "body": "hi"}'}},
            {"id": "c3", "type": "function", "function": {"name": "delete_all", "arguments": "{}"}},
        ]),
        openai_reply(calls=[{"id": "c4", "type": "function", "function": {"name": "calendar", "arguments": "{not json"}}]),
        openai_reply(content="Sent after approval."),
    ])
    client = openai.OpenAI(api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(provider.handler)))
    desk = ApprovalDesk()
    security, sent = personal_agent(desk)

    run = run_agent(security, OpenAIChatModel(client, "test-model"), "Check monday, email me",
                    approval_desk=desk, approval_handler=lambda r: (True, "ok"), approver="alice")

    assert run.answer == "Sent after approval."
    assert [t.outcome.status for t in run.turns] == [
        CallStatus.EXECUTED, CallStatus.EXECUTED, CallStatus.UNKNOWN_TOOL, CallStatus.INVALID_ARGUMENTS,
    ]
    assert sent == ["u@example.com"]
    first, second = provider.requests[0], provider.requests[1]
    assert [t["function"]["name"] for t in first["tools"]] == ["calendar", "send_email"]
    tool_messages = [m for m in second["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2", "c3"]
    assert json.loads(tool_messages[2]["content"])["status"] == "unknown_tool"


def test_openai_compatible_base_url_offline():
    """OpenAI-compatible services (Ollama, Groq, ...) use the same adapter."""
    openai = pytest.importorskip("openai")
    httpx2 = http_module()
    provider = FakeProvider([openai_reply(content="hello")])
    seen_urls = []

    def handler(request):
        seen_urls.append(str(request.url))
        return provider.handler(request)

    client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1",
                           http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    security, _ = personal_agent(ApprovalDesk())
    run = run_agent(security, OpenAIChatModel(client, "local-model"), "hi")
    assert run.answer == "hello"
    assert seen_urls == ["http://localhost:11434/v1/chat/completions"]

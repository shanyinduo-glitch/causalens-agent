import json
from types import SimpleNamespace

from causalens.agent import tool_loop


def make_tool_call(
    tool_name: str,
    arguments: str,
    tool_call_id: str = "call_test_001",
):
    """创建一个模拟的模型工具调用对象。"""

    return SimpleNamespace(
        id=tool_call_id,
        type="function",
        function=SimpleNamespace(
            name=tool_name,
            arguments=arguments,
        ),
    )


def make_response(content: str = "", tool_calls=None):
    """创建一个模拟的千问响应对象。"""

    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
            )
        ]
    )


def test_tool_loop_returns_direct_answer_when_model_calls_no_tool(monkeypatch):
    """模型没有请求工具时，应直接返回模型文本。"""

    def fake_create_chat_completion(**kwargs):
        return make_response(content="这是一个不需要工具的普通回答。")

    monkeypatch.setattr(
        tool_loop,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    result = tool_loop.run_tool_agent(
        question="你好",
        dataset_path="data/sample.csv",
    )

    assert result.answer == "这是一个不需要工具的普通回答。"
    assert result.tool_calls == []
    assert result.tool_results == []
    assert result.confidence == "low"


def test_tool_loop_handles_unknown_tool(monkeypatch):
    """模型请求未注册工具时，应返回失败 ToolResult。"""

    calls = []

    def fake_create_chat_completion(**kwargs):
        calls.append(kwargs)

        if len(calls) == 1:
            unknown_tool_call = make_tool_call(
                tool_name="unknown_tool",
                arguments='{"dataset_path": "data/sample.csv"}',
            )

            return make_response(
                tool_calls=[unknown_tool_call],
            )

        return make_response(
            content="工具调用失败，因为该工具不存在。",
        )

    monkeypatch.setattr(
        tool_loop,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    result = tool_loop.run_tool_agent(
        question="请分析数据集。",
        dataset_path="data/sample.csv",
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "unknown_tool"

    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is False
    assert "未注册的工具" in result.tool_results[0].error

    # 确认失败结果也被作为 role="tool" 消息发回给模型。
    tool_message = calls[1]["messages"][-1]

    assert tool_message["role"] == "tool"

    returned_result = json.loads(tool_message["content"])

    assert returned_result["success"] is False
    assert "未注册的工具" in returned_result["error"]


def test_tool_loop_handles_invalid_json_arguments(monkeypatch):
    """模型返回无法解析的 JSON 参数时，应返回失败 ToolResult。"""

    calls = []

    def fake_create_chat_completion(**kwargs):
        calls.append(kwargs)

        if len(calls) == 1:
            invalid_json_tool_call = make_tool_call(
                tool_name="profile_dataset",
                arguments="{this is not valid json}",
            )

            return make_response(
                tool_calls=[invalid_json_tool_call],
            )

        return make_response(
            content="工具参数格式错误。",
        )

    monkeypatch.setattr(
        tool_loop,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    result = tool_loop.run_tool_agent(
        question="请分析数据集。",
        dataset_path="data/sample.csv",
    )

    # 参数无法解析时，不记录合法 ToolCall。
    assert result.tool_calls == []

    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is False
    assert "无法解析工具参数" in result.tool_results[0].error

    tool_message = calls[1]["messages"][-1]
    returned_result = json.loads(tool_message["content"])

    assert returned_result["success"] is False
    assert "无法解析工具参数" in returned_result["error"]


def test_tool_loop_handles_tool_execution_failure(monkeypatch):
    """真实工具运行出错时，Agent 循环应返回失败 ToolResult。"""

    def failing_profile_dataset(dataset_path: str):
        raise RuntimeError(f"模拟读取失败：{dataset_path}")

    calls = []

    def fake_create_chat_completion(**kwargs):
        calls.append(kwargs)

        if len(calls) == 1:
            profile_tool_call = make_tool_call(
                tool_name="profile_dataset",
                arguments='{"dataset_path": "data/sample.csv"}',
            )

            return make_response(
                tool_calls=[profile_tool_call],
            )

        return make_response(
            content="读取数据集时发生错误。",
        )

    monkeypatch.setattr(
        tool_loop,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    monkeypatch.setitem(
        tool_loop.TOOL_REGISTRY,
        "profile_dataset",
        failing_profile_dataset,
    )

    result = tool_loop.run_tool_agent(
        question="请分析数据集。",
        dataset_path="data/sample.csv",
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "profile_dataset"

    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is False
    assert "工具执行异常" in result.tool_results[0].error
    assert "模拟读取失败" in result.tool_results[0].error

    tool_message = calls[1]["messages"][-1]
    returned_result = json.loads(tool_message["content"])

    assert returned_result["success"] is False
    assert "工具执行异常" in returned_result["error"]
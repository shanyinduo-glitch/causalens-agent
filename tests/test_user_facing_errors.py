from causalens.agent.nodes import (
    _format_user_facing_error,
    reporter_node,
)


def test_format_access_denied_error():
    error = (
        "工具决策模型调用失败："
        "Error code: 403 - "
        "Workspace.AccessDenied"
    )

    message = _format_user_facing_error(error)

    assert "模型服务当前无法访问" in message

    # 底层服务信息不能直接暴露给用户。
    assert "403" not in message
    assert "Workspace.AccessDenied" not in message


def test_format_connection_error():
    error = (
        "Cannot send a request: "
        "connection timeout"
    )

    message = _format_user_facing_error(error)

    assert "外部服务连接失败" in message

    assert "Cannot send a request" not in message
    assert "timeout" not in message


def test_format_missing_column_error():
    error = (
        "Column does not exist: abc. "
        "Available columns: ['score', 'major']"
    )

    message = _format_user_facing_error(error)

    assert "不存在的字段" in message

    # 不直接把工具的英文技术报错原样抛给用户。
    assert "Column does not exist" not in message


def test_format_unknown_tool_error():
    error = "未知工具：dangerous_tool。"

    message = _format_user_facing_error(error)

    assert "无法为当前分析选择有效工具" in message
    assert "dangerous_tool" not in message


def test_format_max_steps_error():
    error = "已达到最大步骤限制，停止执行。"

    message = _format_user_facing_error(error)

    assert "最大执行步骤限制" in message
    assert "自动停止" in message


def test_format_unknown_internal_error():
    error = (
        "ValueError: some unexpected "
        "internal implementation detail"
    )

    message = _format_user_facing_error(error)

    assert "分析过程中出现错误" in message

    assert "ValueError" not in message
    assert "implementation detail" not in message


def test_format_empty_error():
    message = _format_user_facing_error(None)

    assert "分析未能完成" in message


def test_reporter_hides_internal_error_but_preserves_it():
    """
    验证 Reporter 的完整错误出口：

    1. 内部 state 中保留原始技术错误；
    2. final_answer 只展示用户可读信息；
    3. workflow 最终标记为 failed。
    """

    raw_error = (
        "工具决策模型调用失败："
        "Error code: 403 - "
        "Workspace.AccessDenied"
    )

    state = {
        "question": "请分析这份数据。",
        "task_type": "data_profile",
        "tool_results": [],
        "workflow_status": "reporting",
        "error": raw_error,
    }

    updates = reporter_node(state)

    # 模拟 LangGraph 将 node updates 合并回原 state。
    final_state = {
        **state,
        **updates,
    }

    assert final_state["workflow_status"] == "failed"

    # 内部诊断信息仍然存在，方便日志、调试和 Eval。
    assert final_state["error"] == raw_error

    final_answer = final_state["final_answer"]

    # 用户看到的是友好的错误说明。
    assert "模型服务当前无法访问" in final_answer

    # 但不能暴露内部 API / Provider 细节。
    assert "403" not in final_answer
    assert "Workspace.AccessDenied" not in final_answer
    assert "工具决策模型调用失败" not in final_answer


def test_format_non_numeric_field_error():
    error = (
        "相关分析字段 'major' 必须是数值字段。"
        "相关分析字段 'city' 必须是数值字段。"
    )

    message = _format_user_facing_error(error)

    assert "要求使用数值字段" in message
    assert "检查变量类型" in message

    assert "'major'" not in message
    assert "'city'" not in message
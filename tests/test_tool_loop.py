import json
import copy
from types import SimpleNamespace

import causalens.agent.tool_loop as tool_loop
from causalens.schemas import ToolResult


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
        calls.append(copy.deepcopy(kwargs))

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
    assert "未注册" in result.tool_results[0].error

    tool_message = next(
        message
        for message in reversed(calls[1]["messages"])
        if message["role"] == "tool"
    )

    returned_result = json.loads(tool_message["content"])

    assert tool_message["role"] == "tool"
    assert returned_result["success"] is False
    assert "未注册" in returned_result["error"]


def test_tool_loop_handles_invalid_json_arguments(monkeypatch):
    """模型返回无法解析的 JSON 参数时，应返回失败 ToolResult。"""

    calls = []

    def fake_create_chat_completion(**kwargs):
        calls.append(copy.deepcopy(kwargs))

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

    # 参数不合法时，仍记录模型请求，
    # 但把原始字符串放在 raw_arguments 中，方便排查。
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "profile_dataset"
    assert result.tool_calls[0].arguments["raw_arguments"] == (
        "{this is not valid json}"
    )

    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is False
    assert "JSON" in result.tool_results[0].error

    tool_message = next(
        message
        for message in reversed(calls[1]["messages"])
        if message["role"] == "tool"
    )

    returned_result = json.loads(tool_message["content"])

    assert returned_result["success"] is False
    assert "JSON" in returned_result["error"]


def test_tool_loop_handles_tool_execution_failure(monkeypatch):
    """真实工具运行出错时，Agent 循环应返回失败 ToolResult。"""

    def failing_profile_dataset(dataset_path: str):
        raise RuntimeError(f"模拟读取失败：{dataset_path}")

    calls = []

    def fake_create_chat_completion(**kwargs):
        calls.append(copy.deepcopy(kwargs))

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


def make_tool_call(
    tool_name: str,
    arguments: str | dict,
    tool_call_id: str = "call_test",
):
    """
    构造假的模型工具调用对象。

    兼容两种测试写法：
    1. 旧测试：arguments 直接传 JSON 字符串；
    2. 新测试：arguments 传 Python 字典，由这里转成 JSON 字符串。

    tool_call_id 默认给一个值，
    这样旧测试不需要全部手动补 id。
    """
    if isinstance(arguments, dict):
        arguments = json.dumps(
            arguments,
            ensure_ascii=False,
        )

    return SimpleNamespace(
        id=tool_call_id,
        type="function",
        function=SimpleNamespace(
            name=tool_name,
            arguments=arguments,
        ),
    )


def make_model_response(
    content: str = "",
    tool_calls: list | None = None,
):
    """
    构造一个假的模型响应对象。

    它模仿：
    response.choices[0].message
    """
    assistant_message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=assistant_message,
            )
        ]
    )


def test_agent_can_use_profile_then_sql_then_answer(monkeypatch):
    """
    验证 Agent 支持这样的多轮过程：

    第 1 轮：模型调用 profile_dataset
    第 2 轮：模型调用 run_readonly_sql
    第 3 轮：模型直接给出最终回答
    """
    dataset_path = "data/student_performance.csv"

    model_responses = iter(
        [
            # 第 1 次模型调用：先请求 profile_dataset
            make_model_response(
                tool_calls=[
                    make_tool_call(
                        tool_call_id="call_profile",
                        tool_name="profile_dataset",
                        arguments={
                            "dataset_path": dataset_path,
                        },
                    )
                ]
            ),

            # 第 2 次模型调用：拿到字段信息后，请求 SQL 工具。
            # 故意写 sql_query，验证我们的参数兼容逻辑是否生效。
            make_model_response(
                tool_calls=[
                    make_tool_call(
                        tool_call_id="call_sql",
                        tool_name="run_readonly_sql",
                        arguments={
                            "dataset_path": dataset_path,
                            "sql_query": (
                                "SELECT major, AVG(score) AS avg_score "
                                "FROM dataset "
                                "GROUP BY major "
                                "ORDER BY avg_score DESC "
                                "LIMIT 1"
                            ),
                        },
                    )
                ]
            ),

            # 第 3 次模型调用：不再请求工具，直接给最终回答。
            make_model_response(
                content=(
                    "平均 score 最高的是 cs 专业，"
                    "平均分为 81.23 分。"
                )
            ),
        ]
    )

    received_requests = []

    def fake_create_chat_completion(**kwargs):
        """
        替代真实模型 API。

        每次 Agent 请求模型时，
        依次返回上面准备好的三个假响应。
        """
        received_requests.append(copy.deepcopy(kwargs))
        return next(model_responses)

    def fake_profile_dataset(dataset_path: str) -> ToolResult:
        assert dataset_path == "data/student_performance.csv"

        return ToolResult(
            tool_name="profile_dataset",
            success=True,
            result={
                "column_names": [
                    "student_id",
                    "major",
                    "score",
                ],
            },
        )

    def fake_run_readonly_sql(
        dataset_path: str,
        query: str,
    ) -> ToolResult:
        assert dataset_path == "data/student_performance.csv"

        # 这条断言验证：
        # 模型传入的 sql_query 已被标准化为真实函数需要的 query。
        assert "GROUP BY major" in query
        assert "AVG(score)" in query

        return ToolResult(
            tool_name="run_readonly_sql",
            success=True,
            result={
                "row_count": 1,
                "columns": [
                    "major",
                    "avg_score",
                ],
                "rows": [
                    {
                        "major": "cs",
                        "avg_score": 81.23,
                    }
                ],
            },
        )

    # 把真实模型调用替换成 fake，不联网、不花 Token。
    monkeypatch.setattr(
        tool_loop,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    # 把真实工具注册表替换成测试专用工具。
    monkeypatch.setattr(
        tool_loop,
        "TOOL_REGISTRY",
        {
            "profile_dataset": fake_profile_dataset,
            "run_readonly_sql": fake_run_readonly_sql,
        },
    )

    response = tool_loop.run_tool_agent(
        question="哪个专业的平均 score 最高？",
        dataset_path=dataset_path,
    )

    # 1. 最终答案来自第 3 轮模型。
    assert "cs 专业" in response.answer
    assert "81.23" in response.answer

    # 2. Agent 确实连续执行了两个工具。
    assert [tool_call.tool_name for tool_call in response.tool_calls] == [
        "profile_dataset",
        "run_readonly_sql",
    ]

    # 3. 两个工具都成功执行。
    assert len(response.tool_results) == 2
    assert all(result.success for result in response.tool_results)

    # 4. sql_query 已被内部标准化成 query。
    assert response.tool_calls[1].arguments["query"].startswith(
        "SELECT major"
    )
    assert "sql_query" not in response.tool_calls[1].arguments

    # 5. Agent 一共请求模型 3 次。
    assert len(received_requests) == 3

    # 6. 第二次模型调用时，历史中已包含：
    # system、user、assistant 的工具请求、tool 的执行结果。
    second_round_messages = received_requests[1]["messages"]

    assert [message["role"] for message in second_round_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]

    # 7. 第三次模型调用时，历史中已包含两轮工具调用及结果。
    third_round_messages = received_requests[2]["messages"]

    assert [message["role"] for message in third_round_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


def test_agent_can_use_profile_then_chart_then_answer(monkeypatch):
    """
    验证 Agent 支持以下多轮图表流程：

    第 1 轮：模型调用 profile_dataset
    第 2 轮：模型调用 create_chart
    第 3 轮：模型根据真实图表汇总数据生成最终回答
    """
    dataset_path = "data/student_performance.csv"

    model_responses = iter(
        [
            # 第 1 轮：模型先确认字段。
            make_model_response(
                tool_calls=[
                    make_tool_call(
                        tool_call_id="call_profile",
                        tool_name="profile_dataset",
                        arguments={
                            "dataset_path": dataset_path,
                        },
                    )
                ]
            ),

            # 第 2 轮：模型确认字段后请求柱状图。
            make_model_response(
                tool_calls=[
                    make_tool_call(
                        tool_call_id="call_chart",
                        tool_name="create_chart",
                        arguments={
                            "dataset_path": dataset_path,
                            "chart_type": "bar",
                            "x_column": "major",
                            "y_column": "score",
                            "aggregation": "mean",
                            "title": "Average Score by Major",
                        },
                    )
                ]
            ),

            # 第 3 轮：模型不再调用工具，而是根据图表数据回答。
            make_model_response(
                content=(
                    "已生成不同专业平均 score 的柱状图。"
                    "其中 cs 专业平均分最高，为 81.23 分。"
                )
            ),
        ]
    )

    received_requests = []

    def fake_create_chat_completion(**kwargs):
        received_requests.append(copy.deepcopy(kwargs))
        return next(model_responses)

    def fake_profile_dataset(dataset_path: str) -> ToolResult:
        assert dataset_path == "data/student_performance.csv"

        return ToolResult(
            tool_name="profile_dataset",
            success=True,
            result={
                "column_names": [
                    "student_id",
                    "major",
                    "score",
                    "study_hours",
                    "exam_date",
                ],
            },
        )

    def fake_create_chart(
        dataset_path: str,
        chart_type: str,
        x_column: str,
        y_column: str | None = None,
        aggregation: str | None = None,
        title: str | None = None,
    ) -> ToolResult:
        assert dataset_path == "data/student_performance.csv"
        assert chart_type == "bar"
        assert x_column == "major"
        assert y_column == "score"
        assert aggregation == "mean"
        assert title == "Average Score by Major"

        return ToolResult(
            tool_name="create_chart",
            success=True,
            result={
                "chart_path": (
                    "outputs/charts/"
                    "average_score_by_major_test.png"
                ),
                "chart_type": "bar",
                "x_column": "major",
                "y_column": "score",
                "aggregation": "mean",
                "value_column": "mean_score",
                "plotted_row_count": 3,
                "data_preview": [
                    {
                        "major": "cs",
                        "mean_score": 81.23,
                    },
                    {
                        "major": "math",
                        "mean_score": 79.57,
                    },
                    {
                        "major": "stats",
                        "mean_score": 74.69,
                    },
                ],
            },
        )

    monkeypatch.setattr(
        tool_loop,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    monkeypatch.setattr(
        tool_loop,
        "TOOL_REGISTRY",
        {
            "profile_dataset": fake_profile_dataset,
            "create_chart": fake_create_chart,
        },
    )

    response = tool_loop.run_tool_agent(
        question="请画出不同专业平均 score 的柱状图，并说明哪个专业最高。",
        dataset_path=dataset_path,
    )

    # 最终文本来自第 3 轮模型。
    assert "cs 专业" in response.answer
    assert "81.23" in response.answer

    # Agent 实际执行了两个工具。
    assert [tool_call.tool_name for tool_call in response.tool_calls] == [
        "profile_dataset",
        "create_chart",
    ]

    assert len(response.tool_results) == 2
    assert all(result.success for result in response.tool_results)

    # 图表调用参数正确传到了真实工具层。
    chart_arguments = response.tool_calls[1].arguments

    assert chart_arguments["chart_type"] == "bar"
    assert chart_arguments["x_column"] == "major"
    assert chart_arguments["y_column"] == "score"
    assert chart_arguments["aggregation"] == "mean"

    # 多轮过程一共请求模型 3 次。
    assert len(received_requests) == 3

    # 第二轮模型调用前，已经拥有第一轮工具结果。
    second_round_roles = [
        message["role"]
        for message in received_requests[1]["messages"]
    ]

    assert second_round_roles == [
        "system",
        "user",
        "assistant",
        "tool",
    ]

    # 第三轮模型调用前，已经拥有图表工具的结果。
    third_round_roles = [
        message["role"]
        for message in received_requests[2]["messages"]
    ]

    assert third_round_roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]

def test_system_prompt_defines_scatter_evidence_boundaries():
    """
    验证系统提示词明确限制：
    模型不能把散点图预览或图片路径，
    扩写成没有被工具计算过的全量统计结论。
    """
    assert "relationship_summary" in tool_loop.SYSTEM_PROMPT
    assert "data_preview 只是一部分散点预览" in tool_loop.SYSTEM_PROMPT
    assert "不要根据 data_preview 自行创造分段范围" in (
        tool_loop.SYSTEM_PROMPT
    )
    assert "不要把 PNG 图片路径当作已经完成视觉识别的证据" in (
        tool_loop.SYSTEM_PROMPT
    )
    assert "非常紧密的线性趋势" in tool_loop.SYSTEM_PROMPT
    assert "不要把样本关联表述成适用于所有学生的一般规律" in (
        tool_loop.SYSTEM_PROMPT
    )


def test_agent_logs_each_tool_execution(monkeypatch):
    """
    验证 Agent 每执行一次工具，
    都会调用 append_tool_call_log 记录工具名称、参数、
    成功状态和耗时。
    """
    dataset_path = "data/student_performance.csv"

    model_responses = iter(
        [
            make_model_response(
                tool_calls=[
                    make_tool_call(
                        tool_call_id="call_profile",
                        tool_name="profile_dataset",
                        arguments={
                            "dataset_path": dataset_path,
                        },
                    )
                ]
            ),
            make_model_response(
                content="数据集包含 student_id、major 和 score 等字段。",
            ),
        ]
    )

    logged_records = []

    def fake_create_chat_completion(**kwargs):
        return next(model_responses)

    def fake_profile_dataset(
        dataset_path: str,
    ) -> ToolResult:
        return ToolResult(
            tool_name="profile_dataset",
            success=True,
            result={
                "rows": 40,
                "columns": 8,
                "column_names": [
                    "student_id",
                    "major",
                    "score",
                ],
            },
        )

    def fake_append_tool_call_log(**kwargs):
        logged_records.append(kwargs)

    monkeypatch.setattr(
        tool_loop,
        "create_chat_completion",
        fake_create_chat_completion,
    )

    monkeypatch.setattr(
        tool_loop,
        "TOOL_REGISTRY",
        {
            "profile_dataset": fake_profile_dataset,
        },
    )

    monkeypatch.setattr(
        tool_loop,
        "append_tool_call_log",
        fake_append_tool_call_log,
    )

    response = tool_loop.run_tool_agent(
        question="这个数据集有哪些字段？",
        dataset_path=dataset_path,
    )

    assert response.answer == (
        "数据集包含 student_id、major 和 score 等字段。"
    )

    assert len(logged_records) == 1

    log_record = logged_records[0]

    assert log_record["question"] == "这个数据集有哪些字段？"
    assert log_record["dataset_path"] == dataset_path
    assert log_record["tool_name"] == "profile_dataset"
    assert log_record["arguments"] == {
        "dataset_path": dataset_path,
    }
    assert log_record["success"] is True
    assert log_record["error"] is None
    assert log_record["duration_ms"] >= 0


def test_system_prompt_defines_general_chart_evidence_boundaries():
    """
    验证系统提示词对所有图表都规定了统一的证据边界，
    防止模型将有限预览或基础统计扩写成未经计算的结论。
    """
    assert "图表解释的通用证据边界" in tool_loop.SYSTEM_PROMPT

    assert "data_preview 只是一部分数据预览" in (
        tool_loop.SYSTEM_PROMPT
    )

    assert "numeric_summary 只支持描述其中实际返回的统计指标" in (
        tool_loop.SYSTEM_PROMPT
    )

    assert "relationship_summary 只支持描述其中实际返回的点数、范围和" in (
        tool_loop.SYSTEM_PROMPT
    )

    assert "chart_path 只表示 PNG 已生成" in (
        tool_loop.SYSTEM_PROMPT
    )

    assert "当前工具结果不足以支持该判断" in (
        tool_loop.SYSTEM_PROMPT
    )

def test_system_prompt_rejects_unsupported_chart_requests():
    """
    验证系统提示词明确规定：
    对当前不支持的图表类型，Agent 不得调用工具或擅自替代图表。
    """
    assert "当前 create_chart 只支持 bar、scatter、histogram 和 line" in (
        tool_loop.SYSTEM_PROMPT
    )

    assert "不要调用任何数据工具" in tool_loop.SYSTEM_PROMPT

    assert "不要生成替代图表" in tool_loop.SYSTEM_PROMPT

    assert "只有用户明确同意替代方案后" in tool_loop.SYSTEM_PROMPT


def test_system_prompt_mentions_retrieve_docs() -> None:
    from causalens.agent.tool_loop import SYSTEM_PROMPT

    assert "retrieve_docs" in SYSTEM_PROMPT
    assert "字段含义" in SYSTEM_PROMPT
    assert "当前项目文档未提供该信息" in SYSTEM_PROMPT


def test_system_prompt_distinguishes_documented_range_and_observed_range() -> None:
    from causalens.agent.tool_loop import SYSTEM_PROMPT

    assert "字段范围问题" in SYSTEM_PROMPT
    assert "文档定义范围" in SYSTEM_PROMPT
    assert "当前样本实际范围" in SYSTEM_PROMPT
    assert "只调用 retrieve_docs" in SYSTEM_PROMPT
    assert "不要主动追加 profile_dataset" in SYSTEM_PROMPT
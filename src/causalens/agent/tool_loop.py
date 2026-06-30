import json
from typing import Any

from causalens.agent.qwen_client import create_chat_completion
from causalens.agent.tool_registry import TOOL_REGISTRY, TOOLS
from causalens.schemas import AgentResponse, ToolCall, ToolResult


def build_assistant_tool_message(assistant_message: Any) -> dict[str, Any]:
    """
    将 SDK 返回的 assistant 消息，转换成后续请求可使用的普通字典。
    只保留 Function Calling 所需的字段。
    """

    return {
        "role": "assistant",
        "content": assistant_message.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in assistant_message.tool_calls
        ],
    }


SYSTEM_PROMPT = """
你是 CausaLens，一个面向 CSV 数据集的数据分析 Agent。

你的职责是：
1. 理解用户关于 CSV 数据集的问题；
2. 对涉及数据集事实的问题，优先调用工具获取真实结果；
3. 根据工具返回的真实结果，用清晰、简洁的中文回答；
4. 不要编造数据、字段、统计结果或分析结论。

你有两个数据工具：

【profile_dataset】
适合回答数据集整体概况和单列描述性统计，例如：
- 数据集有多少行、多少列；
- 有哪些字段；
- 每列的数据类型；
- 每列缺失值数量；
- 数值列的平均值、中位数、最大值、最小值、标准差；
- passed 等 0/1 数值列的整体平均值或通过率。

【run_readonly_sql】
适合回答需要筛选、分组、聚合、排序、排名、Top-N 或查看具体记录的问题，例如：
- 每个类别有多少条记录；
- 每个专业的平均 score；
- 哪个城市的平均学习时长最高；
- 分数最高或最低的学生是谁；
- 满足某个条件的记录有哪些；
- 按日期、城市、专业等字段进行分组统计。

使用 run_readonly_sql 时，必须遵守以下规则：
1. CSV 在 SQL 中永远叫做 dataset；
2. 只写一条只读 SQL；
3. SQL 必须以 SELECT、WITH 或 EXPLAIN 开头；
4. 不要使用 INSERT、UPDATE、DELETE、DROP、CREATE 等修改语句；
5. 不要使用多个 SQL 语句；
6. 对“最高、最低、Top-N、排名”问题，使用 ORDER BY 和 LIMIT；
7. 对分组统计问题，使用 GROUP BY；
8. 对平均值、总数、最大值、最小值等统计问题，使用 AVG、COUNT、MAX、MIN 等 SQL 聚合函数；
9. 尽量为计算结果设置清晰的别名，例如 AVG(score) AS avg_score；
10. 除非用户明确要求查看原始记录，否则不要无目的地使用 SELECT *。

工具选择规则：
- 整体概况、缺失值、单个数值列的整体描述统计：优先使用 profile_dataset；
- 分组、筛选、排序、Top-N、具体记录查询：必须使用 run_readonly_sql；
- 当用户问题中明确给出了字段名时，可以直接基于这些字段生成 SQL；
- 当你不确定 CSV 中是否存在某个字段时，不要猜测字段名或编造 SQL；应优先调用 profile_dataset；
- 工具失败时，基于错误信息诚实说明无法完成的原因，不要猜测答案。

注意：
- 相关性不等于因果关系；
- 不要仅凭观察性 CSV 数据声称“某变量导致另一个变量变化”；
- 当用户问因果问题时，可以说明当前数据只能支持描述性分析或相关性分析，除非具备额外的研究设计、处理变量、结果变量、混杂变量和可检验假设。
""".strip()


MAX_TOOL_ROUNDS = 4


def _build_assistant_message_for_history(assistant_message):
    """
    把模型返回的 assistant message 转成下一轮请求可使用的字典。

    当模型请求调用工具时，必须把这条 assistant 消息也放回 messages。
    否则下一轮模型不知道某个 tool_call_id 对应哪一次工具请求。
    """
    message_for_history = {
        "role": "assistant",
        "content": assistant_message.content or "",
    }

    if assistant_message.tool_calls:
        message_for_history["tool_calls"] = []

        for tool_call in assistant_message.tool_calls:
            message_for_history["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )

    return message_for_history


def _normalize_tool_arguments(
    tool_name: str,
    arguments: dict,
) -> dict:
    """
    对模型偶尔写错、但可以安全兼容的参数名做标准化处理。

    当前 run_readonly_sql 的正式参数名是 query。
    但模型刚才返回了 sql_query，因此这里兼容一次。
    """
    normalized_arguments = dict(arguments)

    if tool_name == "run_readonly_sql":
        if "query" not in normalized_arguments and "sql_query" in normalized_arguments:
            normalized_arguments["query"] = normalized_arguments.pop("sql_query")

    return normalized_arguments


def _execute_requested_tool(tool_call):
    """
    执行模型请求的一个工具调用。

    无论发生什么错误，都返回：
    - ToolCall：记录模型请求了什么
    - ToolResult：记录执行成功或失败
    """
    tool_name = tool_call.function.name
    raw_arguments = tool_call.function.arguments

    try:
        arguments = json.loads(raw_arguments)

    except (json.JSONDecodeError, TypeError):
        tool_record = ToolCall(
            tool_name=tool_name,
            arguments={
                "raw_arguments": raw_arguments,
            },
        )

        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"模型返回的工具参数不是合法 JSON：{raw_arguments}",
        )

        return tool_record, tool_result

    if not isinstance(arguments, dict):
        tool_record = ToolCall(
            tool_name=tool_name,
            arguments={
                "raw_arguments": raw_arguments,
            },
        )

        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error="模型返回的工具参数必须是 JSON 对象。",
        )

        return tool_record, tool_result

    arguments = _normalize_tool_arguments(tool_name, arguments)

    tool_record = ToolCall(
        tool_name=tool_name,
        arguments=arguments,
    )

    tool_function = TOOL_REGISTRY.get(tool_name)

    if tool_function is None:
        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"模型请求了未注册工具：{tool_name}",
        )

        return tool_record, tool_result

    try:
        tool_result = tool_function(**arguments)

    except Exception as exc:
        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"工具执行异常：{exc}",
        )

    if not isinstance(tool_result, ToolResult):
        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error="工具没有返回 ToolResult，违反了项目的统一工具返回规范。",
        )

    return tool_record, tool_result


def run_tool_agent(question: str, dataset_path: str) -> AgentResponse:
    """
    运行一个支持多轮工具调用的 Agent。

    可能出现的过程：

    用户问题
    ↓
    模型调用 profile_dataset
    ↓
    模型根据字段信息调用 run_readonly_sql
    ↓
    模型拿到 SQL 查询结果，生成最终答案
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"数据集路径：{dataset_path}\n"
                f"用户问题：{question}"
            ),
        },
    ]

    executed_tool_calls = []
    tool_results = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = create_chat_completion(
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # 无论模型是直接回答还是请求工具，都要把 assistant 消息加入历史。
        messages.append(
            _build_assistant_message_for_history(assistant_message)
        )

        requested_tool_calls = assistant_message.tool_calls or []

        # 没有请求工具，说明模型认为已有信息足够，可以直接回答。
        if not requested_tool_calls:
            final_answer = (assistant_message.content or "").strip()

            if not final_answer:
                final_answer = "模型没有返回可读的最终答案。"

            return AgentResponse(
                answer=final_answer,
                tool_calls=executed_tool_calls,
                tool_results=tool_results,
                confidence="low" if not executed_tool_calls else "medium",
            )

        # 模型可能一次请求一个或多个工具。
        for tool_call in requested_tool_calls:
            tool_record, tool_result = _execute_requested_tool(tool_call)

            executed_tool_calls.append(tool_record)
            tool_results.append(tool_result)

            # Tool Message 的 content 必须是字符串。
            # ensure_ascii=False 保证中文不会变成 \uXXXX。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        tool_result.model_dump(),
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

    # 防止模型反复调用工具，导致无限循环或过多 Token 消耗。
    return AgentResponse(
        answer=(
            f"已达到最多 {MAX_TOOL_ROUNDS} 轮工具调用限制，"
            "模型仍未生成最终回答。"
        ),
        tool_calls=executed_tool_calls,
        tool_results=tool_results,
        confidence="low",
    )
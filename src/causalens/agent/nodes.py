from __future__ import annotations

import json
from time import perf_counter
from typing import Any
import re

import pandas as pd
import duckdb
from causalens.agent.qwen_client import create_chat_completion
from causalens.agent.state import AgentState, TaskType
from causalens.agent.tool_registry import TOOL_REGISTRY, TOOLS
from causalens.observability import append_tool_call_log
from causalens.schemas import ToolCall, ToolResult


DESTRUCTIVE_KEYWORDS = [
    "删除",
    "修改数据",
    "覆盖数据",
    "清空",
    "drop ",
    "delete ",
    "update ",
    "insert ",
]

DOC_INTENT_KEYWORDS = [
    "字段是什么意思",
    "字段含义",
    "什么意思",
    "单位是什么",
    "的单位",
    "单位是",
    "单位为",
    "合法取值",
    "编码含义",
    "数据背景",
    "真实数据",
    "数据来源",
    "因果关系",
    "标准化",
    "归一化",
    "缺失值如何处理",
    "缺失值处理",
]

PROFILE_INTENT_KEYWORDS = [
    "有哪些字段",
    "多少行",
    "多少列",
    "数据概览",
    "字段类型",
    "缺失值数量",
    "缺失值有多少",
]

SQL_INTENT_KEYWORDS = [
    "平均",
    "分组",
    "排名",
    "最高",
    "最低",
    "多少个",
    "统计",
    "筛选",
    "总数",
    "求和",
    "最大值",
    "最小值",
]

CHART_INTENT_KEYWORDS = [
    "画图",
    "图表",
    "柱状图",
    "散点图",
    "直方图",
    "分布图",
    "趋势图",
    "折线图",
    "可视化",
]


def _contains_any(
    text: str,
    keywords: list[str],
) -> bool:
    """
    判断文本是否包含关键词列表中的任意一项。
    """

    return any(keyword in text for keyword in keywords)

def intent_router_node(state: AgentState) -> AgentState:
    """
    判断用户问题属于哪一种任务。

    当前使用通用意图规则完成第一层确定性路由。
    不对 major、score、passed 等具体字段进行硬编码。
    """

    question = state.get("question", "").strip().lower()

    print("\n[intent_router_node] 开始判断任务类型")
    print(f"[intent_router_node] 用户问题：{question}")

    has_doc_intent = _contains_any(
        question,
        DOC_INTENT_KEYWORDS,
    )
    has_profile_intent = _contains_any(
        question,
        PROFILE_INTENT_KEYWORDS,
    )
    has_sql_intent = _contains_any(
        question,
        SQL_INTENT_KEYWORDS,
    )
    has_chart_intent = _contains_any(
        question,
        CHART_INTENT_KEYWORDS,
    )

    intent_count = sum(
        [
            has_doc_intent,
            has_profile_intent,
            has_sql_intent,
            has_chart_intent,
        ]
    )

    task_type: TaskType

    if _contains_any(question, DESTRUCTIVE_KEYWORDS):
        task_type = "unsupported"

    elif intent_count >= 2:
        task_type = "mixed_analysis"

    elif has_doc_intent:
        task_type = "rag_question"

    elif has_chart_intent:
        task_type = "chart_analysis"

    elif has_sql_intent:
        task_type = "sql_analysis"

    elif has_profile_intent:
        task_type = "data_profile"

    else:
        task_type = "unsupported"

    print(f"[intent_router_node] 任务类型：{task_type}")

    return {
        "task_type": task_type,
        "workflow_status": "planning",
    }


SINGLE_TOOL_BY_TASK_TYPE: dict[TaskType, str | None] = {
    "data_profile": "profile_dataset",
    "sql_analysis": "run_readonly_sql",
    "chart_analysis": "create_chart",
    "rag_question": "retrieve_docs",
    "mixed_analysis": None,
    "unsupported": None,
}


def _infer_required_tool_names(
    task_type: TaskType,
    question: str,
) -> list[str]:
    """
    根据任务类型和用户问题，推断完成任务所需的工具。

    工具顺序：
    1. 文档证据
    2. 数据概览
    3. SQL 统计
    4. 图表生成

    当前 Week 4 版本默认每一种工具最多执行一次。
    """

    single_tool = SINGLE_TOOL_BY_TASK_TYPE[task_type]

    if single_tool is not None:
        return [single_tool]

    if task_type != "mixed_analysis":
        return []

    normalized_question = question.strip().lower()
    required_tools: list[str] = []

    if _contains_any(
        normalized_question,
        DOC_INTENT_KEYWORDS,
    ):
        required_tools.append("retrieve_docs")

    if _contains_any(
        normalized_question,
        PROFILE_INTENT_KEYWORDS,
    ):
        required_tools.append("profile_dataset")

    if _contains_any(
        normalized_question,
        SQL_INTENT_KEYWORDS,
    ):
        required_tools.append("run_readonly_sql")

    if _contains_any(
        normalized_question,
        CHART_INTENT_KEYWORDS,
    ):
        required_tools.append("create_chart")

    return required_tools


def _get_tool_schemas(
    tool_names: list[str],
) -> list[dict[str, Any]]:
    """
    从 Function Calling schemas 中筛选指定工具。
    """

    allowed_names = set(tool_names)

    return [
        tool_schema
        for tool_schema in TOOLS
        if tool_schema["function"]["name"] in allowed_names
    ]


def _get_successful_tool_names(
    tool_results: list[dict[str, Any]],
) -> list[str]:
    """
    从工具结果历史中提取成功执行过的工具名称。
    """

    successful_names: list[str] = []

    for tool_result in tool_results:
        if tool_result.get("success") is True:
            tool_name = tool_result.get("tool_name")

            if isinstance(tool_name, str):
                successful_names.append(tool_name)

    return successful_names


def _get_tool_guidance(
    tool_name: str,
) -> str:
    """
    返回当前工具对应的参数生成规则。
    """

    guidance_by_tool = {
        "profile_dataset": (
            "本轮调用 profile_dataset。\n"
            "使用应用程序提供的 dataset_path。\n"
            "不要添加不存在的参数。"
        ),
        "retrieve_docs": (
            "本轮调用 retrieve_docs。\n"
            "query 应保留用户真正想查询的字段或规则。\n"
            "不要把数据统计问题伪装成文档问题。\n"
            "top_k 默认使用 3 即可。"
        ),
        "run_readonly_sql": (
            "本轮调用 run_readonly_sql。\n"
            "必须生成一条能够直接完成用户统计需求的只读 SQL，"
            "不能只预览数据或检查表结构。\n"
            "CSV 在 SQL 中的表名始终是 dataset。\n\n"
        
            "必须遵守以下规则：\n"
            "1. 如果用户要求平均值，必须使用 AVG。\n"
            "2. 如果用户要求按类别、分组或不同取值比较，"
            "必须使用 GROUP BY。\n"
            "3. 如果用户要求最高、最低或排名，"
            "应使用 ORDER BY，必要时使用 LIMIT。\n"
            "4. 如果用户问题已经明确给出字段名，"
            "直接使用这些字段完成查询，不要查询表结构。\n"
            "5. 除非用户明确要求查看原始记录，"
            "否则禁止使用 SELECT *。\n"
            "6. 禁止使用仅用于预览数据的 LIMIT 5、LIMIT 10 查询，"
            "除非用户确实要求查看前几条记录。\n"
            "7. 为聚合结果设置清晰别名，例如 avg_value、"
            "record_count、max_value。\n"
            "8. 对于混合任务，本轮 SQL 只需要完成其中的统计需求，"
            "不要尝试通过 SQL 生成图表。\n"
            "9. 不得使用 INSERT、UPDATE、DELETE、DROP、"
            "CREATE 或其他修改数据的语句。"
        ),
        "create_chart": (
            "本轮调用 create_chart。\n"
            "create_chart 会直接重新读取 dataset_path 指向的原始 CSV，"
            "不会读取或绘制上一条 SQL 查询产生的结果表。\n"
            "因此 x_column 和 y_column 必须是原始 CSV 中真实存在的字段，"
            "不得使用 SQL 查询产生的别名、聚合列或计算列，"
            "例如 avg_score、total_count 等。\n\n"
        
            "绘图规则：\n"
            "1. 散点图使用 chart_type=scatter，"
            "并提供两个原始数值字段 x_column 和 y_column。\n"
            "2. 直方图使用 chart_type=histogram，"
            "只提供一个原始数值字段 x_column。\n"
            "3. 柱状图使用 chart_type=bar。\n"
            "4. 当用户要求按类别比较某个数值字段的平均值时，"
            "x_column 使用原始类别字段，"
            "y_column 使用原始数值字段，"
            "aggregation 使用 mean。\n"
            "5. 当用户要求按类别统计记录数时，"
            "aggregation 使用 count，并且不要提供 y_column。\n"
            "6. 折线图使用 chart_type=line，"
            "x_column 应为原始 CSV 中的日期字段。\n"
            "7. 上一条 SQL 返回的列名和别名只能作为报告依据，"
            "不能直接作为 create_chart 的原始 CSV 字段参数。\n"
            "8. 不要编造用户未提供、且没有证据确认的字段。"
        ),
    }

    return guidance_by_tool.get(
        tool_name,
        "请严格按照当前工具 schema 生成参数。",
    )

AVERAGE_KEYWORDS = [
    "平均",
    "均值",
    "average",
    "mean",
]

COUNT_KEYWORDS = [
    "多少个",
    "数量",
    "总数",
    "记录数",
    "计数",
]

GROUP_KEYWORDS = [
    "不同",
    "各个",
    "每个",
    "分别",
    "分组",
    "按",
]

PREVIEW_KEYWORDS = [
    "预览",
    "前几行",
    "前几条",
    "原始记录",
    "样例数据",
    "查看数据",
]


def _normalize_sql_text(query: str) -> str:
    """
    将 SQL 转成便于规则检查的格式。
    """

    return re.sub(
        r"\s+",
        " ",
        query.strip().lower(),
    )


def _validate_sql_against_dataset(
    query: str,
    dataset_path: str,
) -> list[str]:
    """
    使用 DuckDB EXPLAIN 检查 SQL 中的字段和表能否
    绑定到当前 CSV 数据集。

    EXPLAIN 只分析查询计划，不真正返回查询结果。
    """

    errors: list[str] = []
    connection = None

    try:
        # 只读取少量样本，用于获得字段名和基础数据类型。
        dataframe_sample = pd.read_csv(
            dataset_path,
            nrows=100,
        )

        available_columns = (
            dataframe_sample.columns.tolist()
        )

        connection = duckdb.connect(
            database=":memory:",
        )

        connection.register(
            "dataset",
            dataframe_sample,
        )

        # 检查字段名、函数参数和 SQL 结构能否绑定。
        connection.execute(
            f"EXPLAIN {query}"
        )

    except Exception as exc:
        available_columns_text = (
            available_columns
            if "available_columns" in locals()
            else []
        )

        errors.append(
            "SQL 无法绑定到当前数据集字段。"
            f"可用字段：{available_columns_text}。"
            f"具体原因：{exc}"
        )

    finally:
        if connection is not None:
            connection.close()

    return errors


def _validate_sql_tool_call(
    question: str,
    arguments: dict[str, Any],
) -> list[str]:
    """
    检查 SQL 候选调用是否真正满足用户的统计需求。

    返回空列表表示验证通过；
    返回非空列表表示存在问题。
    """

    errors: list[str] = []

    query = arguments.get("query")

    if not isinstance(query, str) or not query.strip():
        return ["SQL 参数 query 不能为空。"]

    normalized_question = question.strip().lower()
    normalized_sql = _normalize_sql_text(query)

    if not normalized_sql.startswith(
        ("select ", "with ")
    ):
        errors.append(
            "SQL 必须是 SELECT 或 WITH 开头的只读查询。"
        )

    forbidden_keywords = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "create ",
        "alter ",
        "truncate ",
    ]

    if any(
        keyword in normalized_sql
        for keyword in forbidden_keywords
    ):
        errors.append(
            "SQL 中包含修改数据或表结构的语句。"
        )

    has_average_intent = _contains_any(
        normalized_question,
        AVERAGE_KEYWORDS,
    )
    has_count_intent = _contains_any(
        normalized_question,
        COUNT_KEYWORDS,
    )
    has_group_intent = _contains_any(
        normalized_question,
        GROUP_KEYWORDS,
    )
    asks_for_preview = _contains_any(
        normalized_question,
        PREVIEW_KEYWORDS,
    )

    if (
        "select *" in normalized_sql
        and not asks_for_preview
    ):
        errors.append(
            "用户没有要求查看原始记录，"
            "不能使用 SELECT * 代替统计分析。"
        )

    if (
        has_average_intent
        and "avg(" not in normalized_sql
    ):
        errors.append(
            "用户要求计算平均值，SQL 必须使用 AVG。"
        )

    if (
        has_count_intent
        and "count(" not in normalized_sql
    ):
        errors.append(
            "用户要求统计数量，SQL 必须使用 COUNT。"
        )

    has_aggregation_intent = (
        has_average_intent
        or has_count_intent
    )

    if (
        has_group_intent
        and has_aggregation_intent
        and "group by" not in normalized_sql
    ):
        errors.append(
            "用户要求比较不同类别，"
            "聚合 SQL 必须使用 GROUP BY。"
        )

    if (
        "最高" in normalized_question
        and "max(" not in normalized_sql
        and not (
            "order by" in normalized_sql
            and "desc" in normalized_sql
        )
    ):
        errors.append(
            "用户要求最高值，SQL 应使用 MAX，"
            "或使用 ORDER BY ... DESC。"
        )

    if (
        "最低" in normalized_question
        and "min(" not in normalized_sql
        and not (
            "order by" in normalized_sql
            and "asc" in normalized_sql
        )
    ):
        errors.append(
            "用户要求最低值，SQL 应使用 MIN，"
            "或使用 ORDER BY ... ASC。"
        )

    dataset_path = arguments.get(
        "dataset_path"
    )

    if (
        isinstance(dataset_path, str)
        and dataset_path.strip()
    ):
        dataset_errors = (
            _validate_sql_against_dataset(
                query=query,
                dataset_path=dataset_path,
            )
        )

        errors.extend(dataset_errors)

    else:
        errors.append(
            "run_readonly_sql 缺少 dataset_path。"
        )

    return errors

    return errors


def _validate_chart_tool_call(
    question: str,
    arguments: dict[str, Any],
) -> list[str]:
    """
    检查图表参数是否能够用于原始 CSV。

    重点防止把 SQL 查询生成的 avg_score 等临时别名，
    当作原始 CSV 字段传给 create_chart。
    """

    errors: list[str] = []

    dataset_path = arguments.get("dataset_path")
    chart_type = arguments.get("chart_type")

    if (
        not isinstance(dataset_path, str)
        or not dataset_path.strip()
    ):
        return ["create_chart 缺少 dataset_path。"]

    if (
        not isinstance(chart_type, str)
        or not chart_type.strip()
    ):
        errors.append(
            "create_chart 缺少 chart_type。"
        )
        return errors

    try:
        dataframe_header = pd.read_csv(
            dataset_path,
            nrows=0,
        )
        available_columns = set(
            dataframe_header.columns.tolist()
        )

    except Exception as exc:
        return [
            f"无法读取数据集字段：{exc}"
        ]

    x_column = arguments.get("x_column")
    y_column = arguments.get("y_column")

    for argument_name, column_name in [
        ("x_column", x_column),
        ("y_column", y_column),
    ]:
        if column_name is None:
            continue

        if not isinstance(column_name, str):
            errors.append(
                f"{argument_name} 必须是字符串。"
            )
            continue

        if column_name not in available_columns:
            errors.append(
                f"{argument_name}={column_name!r} "
                "不是原始 CSV 字段。"
                f"可用字段：{sorted(available_columns)}"
            )

    normalized_chart_type = chart_type.strip().lower()

    if normalized_chart_type == "scatter":
        if not x_column or not y_column:
            errors.append(
                "散点图必须同时提供 x_column 和 y_column。"
            )

    elif normalized_chart_type in {
        "hist",
        "histogram",
    }:
        if not x_column:
            errors.append(
                "直方图必须提供 x_column。"
            )

    elif normalized_chart_type in {
        "bar",
        "line",
    }:
        if not x_column:
            errors.append(
                f"{normalized_chart_type} 图必须提供 x_column。"
            )

    normalized_question = question.strip().lower()
    aggregation = arguments.get("aggregation")

    if (
        normalized_chart_type == "bar"
        and _contains_any(
            normalized_question,
            AVERAGE_KEYWORDS,
        )
        and aggregation != "mean"
    ):
        errors.append(
            "用户要求绘制平均值柱状图，"
            "aggregation 必须为 mean。"
        )

    if (
        normalized_chart_type == "bar"
        and _contains_any(
            normalized_question,
            COUNT_KEYWORDS,
        )
        and aggregation != "count"
    ):
        errors.append(
            "用户要求绘制数量柱状图，"
            "aggregation 必须为 count。"
        )

    return errors


def _validate_pending_tool_call(
    state: AgentState,
) -> list[str]:
    """
    根据工具类型调用相应的确定性验证器。
    """

    pending_tool_call = state.get(
        "pending_tool_call"
    )

    if not pending_tool_call:
        return ["当前没有待验证的工具调用。"]

    tool_name = pending_tool_call.get("name", "")
    arguments = pending_tool_call.get(
        "arguments",
        {},
    )

    if not isinstance(arguments, dict):
        return ["工具 arguments 必须是字典。"]

    question = state.get("question", "")

    if tool_name == "run_readonly_sql":
        return _validate_sql_tool_call(
            question=question,
            arguments=arguments,
        )

    if tool_name == "create_chart":
        return _validate_chart_tool_call(
            question=question,
            arguments=arguments,
        )

    # profile_dataset 与 retrieve_docs 当前没有复杂参数依赖，
    # 暂时只做基础验证。
    if tool_name == "profile_dataset":
        if not arguments.get("dataset_path"):
            return [
                "profile_dataset 缺少 dataset_path。"
            ]

    if tool_name == "retrieve_docs":
        query = arguments.get("query")

        if not isinstance(query, str) or not query.strip():
            return [
                "retrieve_docs 的 query 不能为空。"
            ]

    return []


RAG_SUBQUESTION_SPLIT_MARKERS = [
    "，并统计",
    "并统计",
    "，并计算",
    "并计算",
    "，并画出",
    "并画出",
    "，并绘制",
    "并绘制",
    "，同时统计",
    "同时统计",
    "，然后统计",
    "然后统计",
]


def _extract_rag_subquestion(question: str) -> str:
    """
    从简单混合问题中提取文档检索部分。

    例如：
    “请解释 passed 字段含义，并统计不同 passed 值的平均 score。”
    →
    “请解释 passed 字段含义”
    """

    rag_query = question.strip()

    for marker in RAG_SUBQUESTION_SPLIT_MARKERS:
        if marker in rag_query:
            rag_query = rag_query.split(
                marker,
                maxsplit=1,
            )[0]
            break

    return rag_query.strip(" ，。；;")


def _normalize_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    兼容模型偶尔生成的其他参数名称。
    """

    normalized_arguments = dict(arguments)

    if tool_name == "run_readonly_sql":
        if (
            "query" not in normalized_arguments
            and "sql_query" in normalized_arguments
        ):
            normalized_arguments["query"] = normalized_arguments.pop(
                "sql_query"
            )

    return normalized_arguments


def planner_node(state: AgentState) -> AgentState:
    """
    根据 task_type 生成初步分析计划。

    当前先使用确定性模板，暂时不调用大模型。
    """

    task_type = state.get("task_type", "unsupported")

    print("\n[planner_node] 开始生成分析计划")
    print(f"[planner_node] 当前任务类型：{task_type}")

    plan_by_task_type: dict[TaskType, list[str]] = {
        "data_profile": [
            "读取 CSV 数据集概况",
            "检查字段、数据类型和缺失值",
            "总结数据集基本信息",
        ],
        "sql_analysis": [
            "理解用户需要的统计指标和分组条件",
            "生成只读 SQL 查询",
            "执行查询并检查返回结果",
            "总结关键统计结论",
        ],
        "chart_analysis": [
            "识别图表类型和所需字段",
            "检查字段是否适合绘图",
            "生成图表",
            "总结图表反映的主要信息",
        ],
        "rag_question": [
            "识别需要查询的字段定义或项目规则",
            "从项目知识库检索相关文档",
            "检查检索证据是否足够",
            "基于文档证据回答问题",
        ],
        "mixed_analysis": [
            "拆分用户问题中的多个分析需求",
            "确定各步骤需要使用的工具",
            "按照依赖顺序执行文档检索、数据分析或图表任务",
            "汇总多个工具的结果",
        ],
        "unsupported": [
            "判断请求是否超出当前项目能力或安全边界",
            "说明不能执行的原因",
            "提供当前支持的只读分析方式",
        ],
    }

    plan = plan_by_task_type[task_type]

    for index, step in enumerate(plan, start=1):
        print(f"[planner_node] {index}. {step}")

    return {
        "plan": plan,
        "workflow_status": "tool_decision",
    }


def tool_decision_node(state: AgentState) -> AgentState:
    """
    根据问题和历史工具结果，生成下一次工具调用。

    对混合任务，每轮只提供当前尚未完成的下一个工具，
    防止模型重复调用已经成功的工具。
    """

    dataset_path = state.get("dataset_path", "")
    question = state.get("question", "")
    task_type = state.get("task_type", "unsupported")
    plan = state.get("plan", [])
    tool_results = state.get("tool_results", [])
    validation_error = state.get(
        "validation_error",
        "",
    )

    print("\n[tool_decision_node] 开始决定工具")
    print(f"[tool_decision_node] 当前任务类型：{task_type}")

    if task_type == "unsupported":
        return {
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
        }

    required_tools = (
        state.get("required_tools")
        or _infer_required_tool_names(
            task_type=task_type,
            question=question,
        )
    )

    successful_tools = set(
        _get_successful_tool_names(tool_results)
    )

    remaining_tools = [
        tool_name
        for tool_name in required_tools
        if tool_name not in successful_tools
    ]

    print(
        f"[tool_decision_node] 所需工具：{required_tools}"
    )
    print(
        f"[tool_decision_node] 已完成工具：{sorted(successful_tools)}"
    )
    print(
        f"[tool_decision_node] 剩余工具：{remaining_tools}"
    )

    if not required_tools:
        return {
            "required_tools": [],
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
            "error": "无法根据当前问题确定所需工具。",
        }

    if not remaining_tools:
        return {
            "required_tools": required_tools,
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
            "error": None,
        }

    # 按既定顺序，每轮只处理下一个工具。
    next_tool_name = remaining_tools[0]

    available_tools = _get_tool_schemas(
        [next_tool_name]
    )

    if not available_tools:
        return {
            "required_tools": required_tools,
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
            "error": f"找不到工具 schema：{next_tool_name}",
        }

    plan_text = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(plan, start=1)
    )

    previous_results_text = json.dumps(
        tool_results,
        ensure_ascii=False,
        default=str,
        indent=2,
    )

    tool_guidance = _get_tool_guidance(
        next_tool_name
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是 CausaLens 的工具决策节点。\n"
                "请生成恰好一次工具调用，不要直接回答用户问题。\n"
                f"本轮必须调用的工具是：{next_tool_name}。\n"
                "只能使用当前请求中提供的工具。\n"
                "应用程序提供的数据集路径是唯一可信路径。\n"
                "不得重复执行已经成功完成的相同工具任务。\n\n"
                f"{tool_guidance}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"数据集路径：{dataset_path}\n"
                f"任务类型：{task_type}\n"
                f"用户问题：{question}\n\n"
                f"分析计划：\n{plan_text}\n\n"
                f"此前工具结果：\n{previous_results_text}\n\n"
                f"上一次候选调用的验证反馈：\n"
                f"{validation_error or '无'}\n\n"
                "如果存在验证反馈，请修正对应问题，"
                "不要重复生成相同的错误参数。"
            ),
        },
    ]

    try:
        response = create_chat_completion(
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        requested_tool_calls = (
            assistant_message.tool_calls or []
        )

    except Exception as exc:
        return {
            "required_tools": required_tools,
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
            "error": f"工具决策模型调用失败：{exc}",
        }

    if not requested_tool_calls:
        return {
            "required_tools": required_tools,
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
            "error": (
                "工具决策模型没有生成工具调用。"
                f"模型返回内容：{assistant_message.content or ''}"
            ),
        }

    raw_tool_call = requested_tool_calls[0]

    tool_name = raw_tool_call.function.name
    raw_arguments = raw_tool_call.function.arguments

    if tool_name != next_tool_name:
        return {
            "required_tools": required_tools,
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
            "error": (
                f"模型选择了错误工具 {tool_name}，"
                f"本轮预期工具为 {next_tool_name}。"
            ),
        }

    try:
        arguments = json.loads(raw_arguments)

    except (json.JSONDecodeError, TypeError):
        return {
            "required_tools": required_tools,
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
            "error": (
                "模型生成的工具参数不是合法 JSON："
                f"{raw_arguments}"
            ),
        }

    if not isinstance(arguments, dict):
        return {
            "required_tools": required_tools,
            "pending_tool_call": None,
            "needs_more_tools": False,
            "workflow_status": "reporting",
            "error": "模型生成的工具参数必须是 JSON 对象。",
        }

    arguments = _normalize_tool_arguments(
        tool_name=tool_name,
        arguments=arguments,
    )

    if tool_name in {
        "profile_dataset",
        "run_readonly_sql",
        "create_chart",
    }:
        arguments["dataset_path"] = dataset_path

    # 对单一文档问答任务，直接保留用户原始问题作为检索 query。
    # 避免模型添加数据集名称或堆叠关键词，反而降低向量检索准确性。
    if tool_name == "retrieve_docs":
        if task_type == "rag_question":
            # 单一文档问题直接保留用户原问题。
            arguments["query"] = question

        elif task_type == "mixed_analysis":
            # 混合任务只检索其中的字段定义或规则部分，
            # 避免统计、绘图需求干扰向量检索。
            arguments["query"] = _extract_rag_subquestion(
                question
            )

        arguments.setdefault("top_k", 3)


    pending_tool_call = {
        "id": raw_tool_call.id,
        "name": tool_name,
        "arguments": arguments,
    }

    print(f"[tool_decision_node] 选择工具：{tool_name}")
    print(f"[tool_decision_node] 工具参数：{arguments}")

    return {
        "required_tools": required_tools,
        "pending_tool_call": pending_tool_call,
        "needs_more_tools": True,
        "workflow_status": "tool_validation",
        "error": None,
    }


def tool_validator_node(
    state: AgentState,
) -> AgentState:
    """
    在真正执行工具前，对候选工具调用进行确定性验证。

    验证通过：
        记录正式 tool_call，进入 tool_executor。

    验证失败：
        将错误反馈给 tool_decision，重新生成参数。

    达到最大重试次数：
        停止循环，进入 reporter。
    """

    pending_tool_call = state.get(
        "pending_tool_call"
    )
    retry_count = state.get(
        "validation_retry_count",
        0,
    )
    max_retries = state.get(
        "max_validation_retries",
        2,
    )

    print("\n[tool_validator_node] 开始验证工具调用")

    validation_errors = (
        _validate_pending_tool_call(state)
    )

    if validation_errors:
        next_retry_count = retry_count + 1
        validation_error = "；".join(
            validation_errors
        )

        print(
            "[tool_validator_node] 验证失败："
            f"{validation_error}"
        )
        print(
            "[tool_validator_node] 重试次数："
            f"{next_retry_count}/{max_retries}"
        )

        if next_retry_count >= max_retries:
            final_error = (
                f"工具调用连续 {next_retry_count} 次"
                f"未通过验证：{validation_error}"
            )

            return {
                "pending_tool_call": None,
                "validation_error": validation_error,
                "validation_retry_count": (
                    next_retry_count
                ),
                "workflow_status": "reporting",
                "error": final_error,
            }

        return {
            "pending_tool_call": None,
            "validation_error": validation_error,
            "validation_retry_count": (
                next_retry_count
            ),
            "workflow_status": "tool_decision",
            "error": None,
        }

    if not pending_tool_call:
        return {
            "workflow_status": "reporting",
            "error": "验证通过后却找不到待执行工具。",
        }

    tool_name = pending_tool_call.get(
        "name",
        "",
    )
    arguments = pending_tool_call.get(
        "arguments",
        {},
    )

    tool_record = ToolCall(
        tool_name=tool_name,
        arguments=arguments,
    )

    print(
        "[tool_validator_node] 验证通过："
        f"{tool_name}"
    )

    return {
        # 只有验证通过的调用，才记录为正式 tool_call。
        "tool_calls": [
            tool_record.model_dump()
        ],
        "validation_error": "",
        "validation_retry_count": 0,
        "workflow_status": "tool_execution",
        "error": None,
    }


def tool_executor_node(state: AgentState) -> AgentState:
    """
    执行 pending_tool_call 中记录的工具。

    工具执行完成后不直接进入 Reporter，
    而是进入 result_checker_node 检查是否还需要其他工具。
    """

    pending_tool_call = state.get("pending_tool_call")
    question = state.get("question", "")
    dataset_path = state.get("dataset_path", "")
    step_count = state.get("step_count", 0)

    print("\n[tool_executor_node] 开始执行工具")

    # 情况一：State 中没有待执行工具。
    if not pending_tool_call:
        return {
            "pending_tool_call": None,
            "workflow_status": "checking_result",
            "error": "当前状态中没有待执行的工具调用。",
        }

    tool_name = pending_tool_call.get("name", "")
    arguments = pending_tool_call.get("arguments", {})

    # 情况二：工具参数格式不正确。
    if not isinstance(arguments, dict):
        return {
            "pending_tool_call": None,
            "workflow_status": "checking_result",
            "error": "待执行工具的 arguments 必须是字典。",
        }

    tool_function = TOOL_REGISTRY.get(tool_name)

    # 情况三：模型请求了一个没有注册的工具。
    if tool_function is None:
        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"工具未注册：{tool_name}",
        )

        return {
            "pending_tool_call": None,
            "tool_results": [tool_result.model_dump()],
            "step_count": step_count + 1,
            "workflow_status": "checking_result",
            "error": tool_result.error,
        }

    tool_started_at = perf_counter()

    try:
        tool_result = tool_function(**arguments)

    except Exception as exc:
        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"工具执行异常：{exc}",
        )

    duration_ms = (
        perf_counter() - tool_started_at
    ) * 1000

    # 所有工具都必须返回统一的 ToolResult。
    if not isinstance(tool_result, ToolResult):
        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error=(
                "工具没有返回 ToolResult，"
                "违反了项目统一工具返回规范。"
            ),
        )

    append_tool_call_log(
        question=question,
        dataset_path=dataset_path,
        tool_name=tool_name,
        arguments=arguments,
        success=tool_result.success,
        result=tool_result.result,
        error=tool_result.error,
        duration_ms=duration_ms,
    )

    print(f"[tool_executor_node] 工具：{tool_name}")
    print(f"[tool_executor_node] success={tool_result.success}")

    # 情况四：工具正常执行结束。
    # 无论成功还是失败，都先进入 result_checker_node。
    updates: AgentState = {
        "pending_tool_call": None,
        "tool_results": [tool_result.model_dump()],
        "step_count": step_count + 1,
        "workflow_status": "checking_result",
    }

    if tool_result.success:
        updates["error"] = None
    else:
        updates["error"] = (
            tool_result.error
            or f"工具 {tool_name} 执行失败。"
        )

    return updates


def result_checker_node(state: AgentState) -> AgentState:
    """
    检查当前任务是否还需要调用其他工具。

    判断依据：
    1. 是否发生错误；
    2. 是否达到最大步数；
    3. required_tools 是否全部成功完成。
    """

    question = state.get("question", "")
    task_type = state.get("task_type", "unsupported")
    tool_results = state.get("tool_results", [])
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 5)
    error = state.get("error")

    required_tools = (
        state.get("required_tools")
        or _infer_required_tool_names(
            task_type=task_type,
            question=question,
        )
    )

    successful_tools = set(
        _get_successful_tool_names(tool_results)
    )

    remaining_tools = [
        tool_name
        for tool_name in required_tools
        if tool_name not in successful_tools
    ]

    print("\n[result_checker_node] 开始检查工具结果")
    print(
        f"[result_checker_node] 所需工具：{required_tools}"
    )
    print(
        f"[result_checker_node] 已成功工具：{sorted(successful_tools)}"
    )
    print(
        f"[result_checker_node] 剩余工具：{remaining_tools}"
    )
    print(
        f"[result_checker_node] 当前步数："
        f"{step_count}/{max_steps}"
    )

    if error:
        reason = f"工具执行失败：{error}"

        print(f"[result_checker_node] {reason}")

        return {
            "required_tools": required_tools,
            "needs_more_tools": False,
            "result_check_reason": reason,
            "workflow_status": "reporting",
        }

    if not remaining_tools:
        reason = "完成当前问题所需的工具均已成功执行。"

        print(f"[result_checker_node] {reason}")

        return {
            "required_tools": required_tools,
            "needs_more_tools": False,
            "result_check_reason": reason,
            "workflow_status": "reporting",
            "error": None,
        }

    if step_count >= max_steps:
        reason = (
            f"已达到最大工具步数 {max_steps}，"
            f"仍未完成工具：{remaining_tools}"
        )

        print(f"[result_checker_node] {reason}")

        return {
            "required_tools": required_tools,
            "needs_more_tools": False,
            "result_check_reason": reason,
            "workflow_status": "reporting",
            "error": reason,
        }

    reason = (
        f"仍需执行工具：{remaining_tools}"
    )

    print(f"[result_checker_node] {reason}")

    return {
        "required_tools": required_tools,
        "needs_more_tools": True,
        "result_check_reason": reason,
        "workflow_status": "tool_decision",
        "error": None,
    }


def reporter_node(state: AgentState) -> AgentState:
    """
    根据真实工具结果生成最终回答。
    """

    question = state.get("question", "")
    task_type = state.get("task_type", "unsupported")
    plan = state.get("plan", [])
    tool_results = state.get("tool_results", [])
    error = state.get("error")

    print("\n[reporter_node] 开始生成最终回答")

    if task_type == "unsupported":
        final_answer = (
            f"问题：{question}\n\n"
            "当前请求不属于 CausaLens 支持的只读数据分析范围。\n\n"
            "CausaLens 不会删除、修改、覆盖或清空原始 CSV 数据。"
            "当前项目只支持数据概览、只读 SQL 查询、图表生成"
            "以及项目文档检索。"
        )

        return {
            "final_answer": final_answer,
            "workflow_status": "completed",
        }

    if error:
        return {
            "final_answer": (
                "无法完成当前分析。\n\n"
                f"失败原因：{error}"
            ),
            "workflow_status": "failed",
        }

    if not tool_results:
        return {
            "final_answer": (
                "当前工作流没有获得可用的工具结果，"
                "因此无法可靠回答用户问题。"
            ),
            "workflow_status": "failed",
            "error": "没有可用工具结果。",
        }

    plan_text = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(plan, start=1)
    )

    serialized_tool_results = json.dumps(
        tool_results,
        ensure_ascii=False,
        default=str,
        indent=2,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是 CausaLens 的最终报告节点。\n"
                "请使用清晰、简洁的中文回答用户问题。\n"
                "工具结果是回答事实的唯一依据，不得补充工具没有返回的信息。\n\n"
                "必须遵守以下证据规则：\n"
                "1. profile_dataset 只能证明行数、列数、字段名、数据类型、"
                "缺失值数量和它实际返回的描述性统计。\n"
                "2. 仅凭字段名不能推测字段含义、中文名称、单位、编码规则"
                "或业务含义。\n"
                "3. 字段含义、单位、合法取值和编码含义，只有在 "
                "retrieve_docs 返回相关文档证据时才能说明。\n"
                "4. 用户只询问字段列表时，直接列出字段名即可，不要自行"
                "增加字段解释。\n"
                "5. SQL 结果只能支持其中实际返回的字段和数值。\n"
                "6. 图表结果只能根据 data_preview、numeric_summary、"
                "relationship_summary 等实际返回内容解释。\n"
                "7. 如果使用了 retrieve_docs，应标注可用的来源文件或 section。\n"
                "8. 如果生成了图表，应提供 chart_path。\n"
                "9. 相关性不能证明因果关系。\n"
                "10. 文档表述为未说明或无法确认时，必须保留不确定性。\n"
                "不得编造数字、字段、来源、单位或结论。\n"
                "11. 描述图表坐标轴时，应使用工具实际返回的字段名和聚合方式，"
                "不得自行创造 mean_score、avg_score 等未明确返回的字段名称。\n"
                "12. 对直方图或数值分布，如果工具只返回 count、mean、median、"
                "min 和 max，只能陈述这些统计量及实际取值范围。"
                "除非工具明确返回 skewness、标准差、四分位数、区间频数或"
                "其他分布形状证据，否则不得判断偏态、对称性、集中程度、"
                "峰值区间或长尾特征。\n"
                "13. 如果 create_chart 返回 relationship_summary，最终回答必须总结"
                "其中实际提供的数据点数量、x/y 取值范围和 Pearson 相关系数。"
                "当 Pearson 相关系数为正时，只能表述为当前样本中的正向线性关联；"
                "为负时，只能表述为负向线性关联。"
                "不得把相关关系解释为因果效应，并应明确提醒相关性不能证明因果关系。\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                f"Task type:\n{task_type}\n\n"
                f"Analysis plan:\n{plan_text}\n\n"
                f"Tool results:\n{serialized_tool_results}"
            ),
        },
    ]

    try:
        response = create_chat_completion(
            messages=messages,
        )

        final_answer = (
            response.choices[0].message.content or ""
        ).strip()

    except Exception as exc:
        final_answer = (
            "工具已经执行成功，但最终报告模型调用失败。\n\n"
            f"失败原因：{exc}\n\n"
            f"原始工具结果：\n{serialized_tool_results}"
        )

    if not final_answer:
        final_answer = (
            "工具已经执行成功，但模型没有生成可读回答。\n\n"
            f"原始工具结果：\n{serialized_tool_results}"
        )

    print("[reporter_node] 最终回答已生成")

    return {
        "final_answer": final_answer,
        "workflow_status": "completed",
    }
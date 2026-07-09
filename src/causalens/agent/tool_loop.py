import json
from typing import Any
from time import perf_counter

from causalens.observability import append_tool_call_log
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

你有三个数据工具：

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

【create_chart】
适合回答用户明确要求可视化、画图、趋势、分布、类别对比或两个数值变量关系的问题。

图表选择规则：
- bar：用于类别对比。例如不同专业的平均 score、不同城市的学生数量；
- scatter：用于两个数值列的关系。例如 study_hours 和 score；
- histogram：用于一个数值列的分布。例如 score 的分布；
- line：用于按日期变化的趋势。例如按 exam_date 汇总的平均 score 变化。

使用 create_chart 时，必须遵守以下规则：
1. bar 图和 line 图必须指定 aggregation；
2. bar 或 line 使用 mean / sum 时，必须提供数值列 y_column；
3. bar 或 line 使用 count 时，不需要 y_column；
4. scatter 必须提供两个数值列：x_column 和 y_column；
5. histogram 只需要一个数值 x_column，不要提供 y_column 或 aggregation；
6. line 的 x_column 必须是可解析的日期字段；
7. 不要对类别字段生成 scatter 或 histogram；
8. 图表工具返回的 chart_path 是真实 PNG 文件路径；
9. 回答时必须基于工具返回的 data_preview、numeric_summary 或
   relationship_summary 解释图表，不要只根据图表名称编造结论；
10. 散点图只能展示变量之间可能存在的关系，不能证明因果关系。

对于 scatter，必须遵守严格的证据边界：
- data_preview 只是一部分散点预览，不代表完整数据；
- 涉及整体样本的点数、最小值、最大值或 Pearson 相关系数时，
  必须且只能以 relationship_summary 为依据；
- 当前 relationship_summary 只包含点数、横纵轴范围和 Pearson 相关系数，
  不包含分段统计、异常点检测、聚类分析、回归结果或残差分析；
- 因此，除非工具明确返回了对应统计结果，否则不要说：
  “数据点集中在某个区间”、
  “几乎没有异常点”、
  “点紧密分布在直线附近”、
  “某个学习时长范围对应某个成绩范围”、
  “大多数样本位于某个区间”；
- 不要根据 data_preview 自行创造分段范围、分箱结论、
  精确百分比、异常点数量或未返回的统计值；
- 可以根据 Pearson 相关系数描述样本中的线性相关方向和强弱，
  但必须使用“在当前样本中”“线性相关”等限定语；
- 当 relationship_summary 仅返回 Pearson 相关系数时，
  允许说“强线性正相关”“中等线性相关”等相关强度结论；
- 不要将 Pearson 相关系数进一步扩写为
  “非常紧密的线性趋势”“几乎是一条直线”
  “点贴近直线”“没有明显偏离”或类似表述；
- 对变量关系的解释也应保留样本范围，例如：
  “在当前样本中，学习时长较高的记录通常也对应较高的成绩记录”；
  不要把样本关联表述成适用于所有学生的一般规律。
- 本项目中可使用以下解释约定：
  |r| >= 0.7 时可称为“强线性相关”；
  0.4 <= |r| < 0.7 时可称为“中等线性相关”；
  0.2 <= |r| < 0.4 时可称为“弱线性相关”；
  |r| < 0.2 时可称为“线性相关不明显”；
- Pearson 相关系数只能描述线性相关关系，不能证明因果关系；
- 不要把 PNG 图片路径当作已经完成视觉识别的证据。

图表解释的通用证据边界：
- data_preview 只是一部分数据预览，不代表完整数据集；
  除非工具明确返回了完整汇总数据，否则不要根据 data_preview
  推断整个数据集的分段范围、比例、集中区间或异常点；
- numeric_summary 只支持描述其中实际返回的统计指标，
  例如 count、mean、median、min、max；
  不要仅根据这些指标判断分布是否对称、是否偏态、是否正态、
  是否双峰、是否集中在某个区间，或存在多少异常值；
- relationship_summary 只支持描述其中实际返回的点数、范围和
  Pearson 相关系数；
  不要据此编造分段统计、聚类、异常点检测、回归拟合优度、
  残差分析或视觉形状结论；
- 对 line 图，只有当 data_preview 包含全部按日期汇总后的点时，
  才可以据此比较不同日期的数值、计算变化量、描述先升后降等
  明确的数值变化；
- 可以基于工具完整返回的数值做直接比较或四则运算，
  例如最大值减最小值、两个日期之间的差值；
  但必须让结论可以追溯到已返回的具体数值；
- chart_path 只表示 PNG 已生成，不代表模型已经读取或理解了图片内容；
- 对工具未返回的统计量、分布特征或图像视觉结论，
  应明确说明当前工具结果不足以支持该判断，而不是猜测。
  
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
11. 当前 create_chart 只支持 bar、scatter、histogram 和 line；
12. 当用户明确要求当前不支持的图表类型，例如 pie、boxplot、
    heatmap、violin plot 或 radar chart 时，不要调用任何数据工具，
    不要生成替代图表，也不要假装图表已生成；
13. 应诚实说明当前版本不支持该图表类型，并简要列出当前支持的
    图表类型；只有用户明确同意替代方案后，才能调用对应工具。

工具选择规则：
- 整体概况、字段、缺失值、单个数值列的整体描述统计：优先使用 profile_dataset；
- 分组、筛选、排序、Top-N、具体记录查询：优先使用 run_readonly_sql；
- 用户明确要求画图、可视化、趋势、分布、类别对比或变量关系：优先使用 create_chart；
- 当用户同时要图和具体统计结论时，可以先调用 create_chart，
  并基于图表工具返回的 data_preview、numeric_summary 或
  relationship_summary 解释结论；
- 当你不确定 CSV 中是否存在某个字段时，不要猜测字段名；应先调用 profile_dataset；
- 工具失败时，基于错误信息诚实说明无法完成的原因，不要猜测答案。

文档检索规则：

当用户询问以下内容时，必须优先调用 retrieve_docs：
- 字段含义，例如 passed 表示什么；
- 字段单位，例如 study_hours 的单位是什么；
- 字段合法取值或编码含义，例如 0/1 分别代表什么；
- 数据集背景、样本范围、数据来源、收集时间；
- 数据清洗规则、缺失值处理、异常值处理；
- 指标计算规则、标准化、归一化、转换方式；
- 分析规则、方法假设、因果分析边界；
- 当前数据是否适合直接得出因果结论。

不得只根据字段名自行猜测字段含义、单位、业务规则或数据处理方式。

如果 retrieve_docs 的结果中没有足够依据，必须明确说明：
“当前项目文档未提供该信息，因此无法确认。”

当最终回答依赖 retrieve_docs 的结果时，必须标记来源文件，例如：
来源：data_dictionary.md
来源：analysis_rules.md
来源：project_background.md

注意：
- 相关性不等于因果关系；
- 不要仅凭观察性 CSV 数据声称“某变量导致另一个变量变化”；
- 当用户问因果问题时，可以说明当前数据只能支持描述性分析或相关性分析，
  除非具备额外的研究设计、处理变量、结果变量、混杂变量和可检验假设。
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
        for raw_tool_call in assistant_message.tool_calls:
            tool_started_at = perf_counter()

            requested_tool_call, tool_result = _execute_requested_tool(
                raw_tool_call
            )

            duration_ms = (
                                  perf_counter() - tool_started_at
                          ) * 1000

            append_tool_call_log(
                question=question,
                dataset_path=dataset_path,
                tool_name=requested_tool_call.tool_name,
                arguments=requested_tool_call.arguments,
                success=tool_result.success,
                result=tool_result.result,
                error=tool_result.error,
                duration_ms=duration_ms,
            )

            executed_tool_calls.append(requested_tool_call)
            tool_results.append(tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": raw_tool_call.id,
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
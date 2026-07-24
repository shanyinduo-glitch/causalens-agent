# Week 4：使用 LangGraph 重构 CausaLens Agent 工作流

## 1. 本周目标

本周的主要目标是将 CausaLens Agent 原有的手写工具调用循环重构为基于 LangGraph 的可维护工作流。

在重构之前，项目主要通过普通 Python `while` 循环完成以下过程：

1. 调用大语言模型；
2. 判断模型是否请求工具；
3. 执行工具；
4. 将工具结果重新发送给模型；
5. 重复上述过程，直到生成最终回答。

这种方式可以工作，但随着工具数量和任务复杂度增加，会出现以下问题：

* 工作流结构不够直观；
* 状态分散在普通变量和消息列表中；
* 多工具任务难以追踪；
* 错误处理和停止条件容易混在一起；
* 不方便对每一个步骤进行单元测试；
* 后续难以增加验证、重试、人工审批等能力。

因此，本周使用 LangGraph 将 Agent 拆分为多个职责明确的节点，并通过条件边控制节点之间的跳转。

---

## 2. 本周完成的工作

本周完成了以下核心功能：

* 学习并运行最小 LangGraph 示例；
* 理解 `StateGraph`、node、edge、`START` 和 `END`；
* 设计统一的 `AgentState`；
* 实现任务意图路由节点；
* 实现分析计划节点；
* 实现工具决策节点；
* 实现工具参数验证节点；
* 实现工具执行节点；
* 实现结果检查节点；
* 实现最终报告节点；
* 接入原有的 Profile、SQL、Chart 和 RAG 工具；
* 支持单工具任务；
* 支持多工具循环；
* 支持工具失败处理；
* 支持验证失败后的参数重新生成；
* 支持最大工具步数限制；
* 支持最大验证重试次数限制；
* 更新节点和图的自动化测试；
* 完成全部项目测试。

---

## 3. 当前工作流架构

当前 CausaLens Agent 的主要工作流如下：

```text
START
  ↓
intent_router
  ├── 不支持的请求 ─────────────────────→ reporter
  ↓
planner
  ↓
tool_decision
  ↓
tool_validator
  ├── 验证失败且未达到上限 ─────────────→ tool_decision
  ├── 验证失败且达到上限 ───────────────→ reporter
  └── 验证通过
          ↓
    tool_executor
          ↓
    result_checker
      ├── 仍需要其他工具 ───────────────→ tool_decision
      └── 已完成、失败或达到步数上限 ───→ reporter
                                               ↓
                                              END
```

这个工作流形成了一个可控的循环：

```text
工具决策
→ 参数验证
→ 工具执行
→ 结果检查
→ 决定继续或结束
```

与原来的手写循环相比，LangGraph 版本能够更清晰地展示每一个阶段的职责和状态变化。

---

## 4. AgentState 设计

`AgentState` 是 LangGraph 中所有节点共享的状态。

主要字段包括：

### 4.1 用户输入

```text
dataset_path
question
```

* `dataset_path`：用户当前需要分析的 CSV 文件路径；
* `question`：用户提出的自然语言问题。

### 4.2 任务理解

```text
task_type
plan
required_tools
```

* `task_type`：任务类型；
* `plan`：Planner 生成的高层分析计划；
* `required_tools`：完成当前任务所需的工具列表。

当前任务类型包括：

```text
data_profile
sql_analysis
chart_analysis
rag_question
mixed_analysis
unsupported
```

### 4.3 工作流控制

```text
workflow_status
needs_more_tools
result_check_reason
```

* `workflow_status`：当前工作流阶段；
* `needs_more_tools`：是否还需要继续调用工具；
* `result_check_reason`：Result Checker 的判断原因。

### 4.4 工具调用状态

```text
pending_tool_call
tool_calls
tool_results
```

* `pending_tool_call`：当前等待验证或执行的候选工具调用；
* `tool_calls`：已经通过验证的正式工具调用历史；
* `tool_results`：工具实际执行结果历史。

`tool_calls` 和 `tool_results` 使用 reducer 进行追加，因此多轮工具调用不会覆盖之前的历史。

被 Validator 拦截的错误候选调用不会写入正式的 `tool_calls`。

### 4.5 重试与停止条件

```text
validation_error
validation_retry_count
max_validation_retries
step_count
max_steps
```

* `validation_error`：上一次候选工具调用未通过验证的原因；
* `validation_retry_count`：当前工具连续验证失败次数；
* `max_validation_retries`：允许的最大验证重试次数；
* `step_count`：已经真正执行的工具次数；
* `max_steps`：允许的最大工具执行次数。

验证失败只增加 `validation_retry_count`，不会增加 `step_count`。

只有工具真正进入 `tool_executor_node` 后，`step_count` 才会增加。

### 4.6 最终输出

```text
final_answer
error
```

* `final_answer`：Reporter 生成的最终回答；
* `error`：工作流中出现的错误信息。

---

## 5. 各节点职责

## 5.1 Intent Router

`intent_router_node` 根据用户问题判断任务类型。

示例：

```text
这个数据集有哪些字段？
→ data_profile
```

```text
passed 字段是什么意思？
→ rag_question
```

```text
不同专业的平均 score 是多少？
→ sql_analysis
```

```text
请画出 study_hours 和 score 的散点图。
→ chart_analysis
```

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
→ mixed_analysis
```

```text
请删除所有 score 小于 60 的学生。
→ unsupported
```

当前 Router 使用通用意图关键词，不对 `score`、`major` 或 `passed` 等具体字段进行硬编码。

当问题同时包含两种以上分析意图时，任务会被识别为 `mixed_analysis`。

---

## 5.2 Planner

`planner_node` 根据任务类型生成高层执行计划。

Planner 只描述任务步骤，不直接执行工具。

例如，SQL 分析任务的计划可能是：

```text
1. 理解用户需要的统计指标和分组条件
2. 生成只读 SQL 查询
3. 执行查询并检查返回结果
4. 总结关键统计结论
```

混合任务的计划可能是：

```text
1. 拆分用户问题中的多个分析需求
2. 确定各步骤需要使用的工具
3. 按照依赖顺序执行文档检索、数据分析或图表任务
4. 汇总多个工具的结果
```

---

## 5.3 Tool Decision

`tool_decision_node` 根据以下信息生成下一次候选工具调用：

* 用户问题；
* 任务类型；
* 分析计划；
* 已经成功执行的工具；
* 历史工具结果；
* 上一次验证失败原因。

对于单工具任务，它会选择对应工具。

对于混合任务，它会根据 `required_tools` 的顺序，每轮只选择一个尚未完成的工具。

当前工具对应关系为：

```text
data_profile
→ profile_dataset

sql_analysis
→ run_readonly_sql

chart_analysis
→ create_chart

rag_question
→ retrieve_docs
```

混合任务可以包含多个工具，例如：

```text
retrieve_docs
→ run_readonly_sql
```

或者：

```text
run_readonly_sql
→ create_chart
```

---

## 5.4 Tool Validator

`tool_validator_node` 位于 Tool Decision 和 Tool Executor 之间。

模型生成的工具调用不会直接执行，而是先经过 Python 确定性验证。

```text
模型生成候选调用
→ Python 验证
→ 合格后执行
```

这样可以减少对 Prompt 的过度依赖。

### SQL 验证

SQL 验证器目前会检查：

* SQL 是否以 `SELECT` 或 `WITH` 开头；
* 是否包含写操作；
* 用户要求平均值时是否使用 `AVG`；
* 用户要求计数时是否使用 `COUNT`；
* 分组统计时是否使用 `GROUP BY`；
* 用户没有要求预览时是否错误使用 `SELECT *`；
* 最高或最低值任务是否使用正确的聚合或排序；
* `dataset_path` 是否存在；
* SQL 中引用的字段是否真实存在；
* SQL 是否可以在当前数据集上完成绑定。

例如，下面的 SQL 会被拦截：

```sql
SELECT * FROM dataset LIMIT 5
```

因为它没有完成“不同专业平均 score”的统计需求。

下面的 SQL也会被拦截：

```sql
SELECT 专业, AVG(score)
FROM dataset
GROUP BY 专业
```

因为原始 CSV 中不存在名为 `专业` 的字段。

正确查询应为：

```sql
SELECT
    major,
    AVG(score) AS avg_score
FROM dataset
GROUP BY major
ORDER BY major
```

### 图表参数验证

图表验证器目前会检查：

* `dataset_path` 是否存在；
* `chart_type` 是否提供；
* `x_column` 是否存在于原始 CSV；
* `y_column` 是否存在于原始 CSV；
* 散点图是否同时提供两个字段；
* 直方图是否提供数值字段；
* 柱状图的聚合方式是否符合用户需求；
* 是否错误地将 SQL 临时别名当成原始 CSV 字段。

例如：

```text
{
    "x_column": "major",
    "y_column": "avg_score",
    "aggregation": "mean",
}
```

会被拦截，因为 `avg_score` 只是 SQL 查询产生的临时别名，不是原始 CSV 字段。

正确参数应为：

```text
{
    "x_column": "major",
    "y_column": "score",
    "aggregation": "mean",
}
```

由 `create_chart` 自己读取原始 CSV 并计算平均值。

### 验证重试

验证失败后，错误原因会写入：

```text
validation_error
```

下一次 Tool Decision 会读取这条反馈，并要求模型重新生成候选调用。

如果连续验证失败次数达到：

```text
max_validation_retries
```

工作流会停止重试并进入 Reporter，避免形成死循环。

---

## 5.5 Tool Executor

`tool_executor_node` 只执行已经通过 Validator 的工具调用。

它通过 `TOOL_REGISTRY` 获取真实 Python 工具函数，并记录：

* 工具名称；
* 工具参数；
* 是否执行成功；
* 返回结果；
* 错误信息；
* 执行耗时。

所有工具必须统一返回 Pydantic `ToolResult`。

如果工具：

* 未注册；
* 抛出异常；
* 没有返回 `ToolResult`；
* 返回 `success=False`；

Tool Executor 都会将其转换为统一的失败结果。

工具执行结束后不会直接进入 Reporter，而是先进入 Result Checker。

---

## 5.6 Result Checker

`result_checker_node` 检查当前任务是否已经完成。

它主要比较：

```text
required_tools
```

和：

```text
已经成功执行的工具
```

如果还有工具没有成功执行：

```text
result_checker
→ tool_decision
```

如果全部完成：

```text
result_checker
→ reporter
```

如果出现工具错误：

```text
result_checker
→ reporter
```

如果达到最大工具步数：

```text
result_checker
→ reporter
```

例如，在下面的任务中：

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

`required_tools` 为：

```text
[
    "retrieve_docs",
    "run_readonly_sql",
]
```

执行完 `retrieve_docs` 后，Result Checker 会发现还缺少：

```text
run_readonly_sql
```

因此再次进入 Tool Decision。

---

## 5.7 Reporter

`reporter_node` 根据真实工具结果生成最终回答。

Reporter 必须遵守证据边界：

* Profile 结果只能说明数据集实际字段、行列数、类型、缺失和统计信息；
* 字段含义、单位、编码和合法取值必须来自 RAG 文档；
* SQL 结果只能说明查询实际返回的行和数值；
* 图表结论只能基于工具返回的图表路径、数值摘要和关系摘要；
* 相关性不能被解释为因果关系；
* 文档未明确说明的信息必须保留不确定性；
* 工具失败时不能编造不存在的分析结果。

例如，文档只定义：

```text
passed=1 表示通过
passed=0 表示未通过
```

但没有定义具体分数线。

Reporter 可以说明编码含义，但不能自行推断：

```text
score 大于等于 60 就是通过
```

---

## 6. 单工具任务测试

本周完成了四类单工具任务。

### 6.1 数据概览

问题：

```text
这个数据集有哪些字段？
```

流程：

```text
data_profile
→ profile_dataset
→ reporter
```

结果返回：

* 数据集行数；
* 数据集列数；
* 字段名称；
* 数据类型；
* 缺失值；
* 数值字段摘要。

---

### 6.2 RAG 问答

问题：

```text
passed 字段是什么意思？
```

流程：

```text
rag_question
→ retrieve_docs
→ reporter
```

回答基于 `data_dictionary.md`：

* `passed` 是二元变量；
* `1` 表示通过考试；
* `0` 表示未通过考试；
* 文档没有定义具体及格分数线。

---

### 6.3 SQL 分析

问题：

```text
不同专业的平均 score 是多少？
```

流程：

```text
sql_analysis
→ run_readonly_sql
→ reporter
```

SQL：

```sql
SELECT
    major,
    AVG(score) AS avg_score
FROM dataset
GROUP BY major
ORDER BY major
```

查询结果：

| major | avg_score |
| ----- | --------: |
| cs    |     81.23 |
| math  |     79.57 |
| stats |     74.69 |

---

### 6.4 图表分析

问题：

```text
请画出 study_hours 和 score 的散点图。
```

流程：

```text
chart_analysis
→ create_chart
→ reporter
```

工具返回：

* 图表文件路径；
* 40 个数据点；
* `study_hours` 范围；
* `score` 范围；
* Pearson 相关系数约为 0.9874。

最终回答会说明两者在当前样本中存在很强的线性正相关，但不会将相关性解释为因果关系。

---

## 7. 多工具任务测试

## 7.1 RAG + SQL

问题：

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

执行流程：

```text
retrieve_docs
→ result_checker
→ run_readonly_sql
→ result_checker
→ reporter
```

`required_tools`：

```text
[
    "retrieve_docs",
    "run_readonly_sql",
]
```

SQL 查询结果：

| passed | 含义    | average_score |
| -----: | ----- | ------------: |
|      0 | 未通过考试 |         62.09 |
|      1 | 通过考试  |         84.76 |

最终回答会同时说明：

* `passed` 的文档定义；
* 两个编码的平均成绩；
* 文档没有定义具体及格线。

---

## 7.2 SQL + Chart

问题：

```text
请统计不同专业的平均 score，并画出柱状图。
```

执行流程：

```text
run_readonly_sql
→ result_checker
→ create_chart
→ result_checker
→ reporter
```

SQL 负责返回统计结果：

```sql
SELECT
    major,
    AVG(score) AS avg_score
FROM dataset
GROUP BY major
ORDER BY major
```

图表工具使用原始 CSV 字段：

```text
{
    "chart_type": "bar",
    "x_column": "major",
    "y_column": "score",
    "aggregation": "mean",
}
```

最终回答同时包含：

* 各专业平均成绩；
* 柱状图文件路径。

---

## 8. 本周遇到的问题与解决方法

## 8.1 Tool Executor 同时连接两个节点

### 问题

在 Graph 中同时保留了：

```text
tool_executor → reporter
```

和：

```text
tool_executor → result_checker
```

导致两个节点在同一轮同时修改 `workflow_status`，LangGraph 报错：

```text
InvalidUpdateError:
At key 'workflow_status':
Can receive only one value per step
```

### 原因

普通 State 字段只能在同一个执行步骤接收一个更新值。

### 解决方法

删除旧边：

```text
tool_executor → reporter
```

只保留：

```text
tool_executor → result_checker
```

由 Result Checker 决定是否进入 Reporter。

---

## 8.2 重复注册条件边

### 问题

对 `tool_decision` 重复调用了两次：

```text
add_conditional_edges(...)
```

LangGraph 报错：

```text
Branch with name route_after_tool_decision
already exists for node tool_decision
```

### 解决方法

删除旧的：

```text
tool_decision → tool_executor
```

只保留新的：

```text
tool_decision → tool_validator
```

---

## 8.3 SQL 只预览数据，没有完成统计

### 问题

模型曾生成：

```sql
SELECT * FROM dataset LIMIT 5
```

虽然工具执行成功，但没有完成“不同专业平均 score”的任务。

### 解决方法

最开始通过 Prompt 强调必须使用 `AVG` 和 `GROUP BY`。

随后进一步新增 Python SQL Validator：

* 检查平均值意图；
* 检查分组意图；
* 检查 `SELECT *`；
* 不再只依赖 Prompt。

---

## 8.4 SQL 临时别名被当成图表字段

### 问题

SQL 返回：

```text
avg_score
```

模型随后生成：

```text
y_column="avg_score"
```

但 `create_chart` 会重新读取原始 CSV，原始 CSV 不存在 `avg_score`。

### 解决方法

图表 Validator 会读取 CSV 表头，检查字段是否真实存在。

正确调用为：

```text
x_column="major"
y_column="score"
aggregation="mean"
```

---

## 8.5 模型将中文含义当成实际字段名

### 问题

模型曾生成：

```sql
SELECT 专业, AVG(score)
FROM dataset
GROUP BY 专业
```

但实际字段是：

```text
major
```

### 解决方法

增加 DuckDB `EXPLAIN` 验证。

在真正执行 SQL 前，将当前 CSV 注册为 `dataset`，通过 `EXPLAIN` 检查字段能否绑定。

如果字段不存在，Validator 会返回：

* 可用字段列表；
* DuckDB 的具体绑定错误；
* 让 Tool Decision 重新生成 SQL。

---

## 8.6 自动测试受实时模型随机性影响

### 问题

同一个测试问题，有时模型生成正确 SQL，有时生成：

```sql
SELECT * FROM dataset LIMIT 5
```

自动测试因此偶尔失败。

### 原因

自动测试原本实际调用了在线模型，测试结果依赖模型当次输出。

### 解决方法

在 Graph 测试中使用 `monkeypatch` 替换模型调用，返回固定的工具调用和最终回答。

自动测试现在主要验证：

* Graph 路由；
* State 更新；
* Validator；
* Tool Executor；
* Result Checker；
* 多工具循环。

真实模型表现则通过手动端到端测试验证。

---

## 9. 错误处理与 Human-in-the-loop

当前项目没有单独设置 `error_handler_node`。

错误处理分散在各阶段：

* Tool Decision 处理模型调用和 JSON 解析错误；
* Tool Validator 处理参数和语义验证错误；
* Tool Executor 处理工具未注册和工具执行异常；
* Result Checker 处理工具失败和步数上限；
* Reporter 负责向用户输出可理解的失败信息。

这种分层处理方式能够保留错误发生的具体阶段和原因。

当前 CausaLens 是只读数据分析 Agent。

以下请求会被识别为 `unsupported`：

* 删除数据；
* 更新数据；
* 覆盖文件；
* 清空记录；
* 执行写入 SQL。

由于当前版本不执行写操作，因此暂时不需要在数据修改前增加人工审批节点。

未来如果开放以下能力：

* 删除记录；
* 修改 CSV；
* 覆盖原始文件；
* 执行数据库写操作；

则必须加入 Human-in-the-loop 审批节点，在真正执行写操作前获得用户确认。

---

## 10. 自动化测试

本周更新了：

```text
tests/test_nodes.py
tests/test_graph.py
```

测试覆盖：

* 六种任务类型的 Router；
* 不同任务类型的 Planner；
* SQL 与 RAG 计划内容；
* Reporter 成功和失败路径；
* 支持任务的完整 Graph；
* 不支持请求跳过 Planner；
* 条件路由；
* 初始 State 字段保留；
* SQL 工具真实执行；
* 单工具任务；
* 混合任务；
* Validator；
* 多工具循环。

本周相关测试结果：

```text
30 passed
```

随后运行项目完整测试套件，全部测试通过。

---

## 11. 本周架构改进总结

本周最重要的改进不是简单地“使用了 LangGraph”，而是完成了以下架构升级：

### 改进前

```text
模型
→ 直接调用工具
→ 工具成功就认为任务完成
```

### 改进后

```text
模型生成候选调用
→ Python 确定性验证
→ 执行工具
→ 检查是否真正完成全部工具步骤
→ 必要时继续循环
```

模型负责：

* 理解自然语言；
* 生成 SQL；
* 生成图表参数；
* 组织最终回答。

Python 代码负责：

* 安全边界；
* SQL 结构检查；
* 字段存在性检查；
* 图表参数检查；
* 重试上限；
* 工具步数限制；
* 工作流状态；
* 工具执行记录。

这种分工比单纯依赖 Prompt 更稳定。

---

## 12. 当前局限

当前版本仍存在以下局限：

### 12.1 Router 仍然基于关键词

虽然没有硬编码具体字段，但意图识别仍然依赖关键词。

一些复杂或模糊问题可能被错误分类。

未来可以考虑：

* 模型路由；
* 规则与模型结合；
* 结构化意图分类结果。

### 12.2 Required Tools 使用预定义顺序

当前混合任务按照：

```text
RAG
→ Profile
→ SQL
→ Chart
```

的顺序执行。

这个顺序适合目前的任务，但还不是完整的动态依赖规划。

### 12.3 Result Checker 主要检查工具完成情况

当前 Result Checker 主要判断：

* 所需工具是否都成功执行；
* 是否出现错误；
* 是否达到步数上限。

虽然 SQL 和图表在执行前经过语义验证，但 Result Checker 本身还没有使用模型检查最终答案是否完整满足用户需求。

### 12.4 Chart 不直接绘制 SQL 结果表

`create_chart` 会重新读取原始 CSV。

因此它不能直接绘制任意 SQL 查询产生的临时结果表。

当前通过：

```text
原始字段 + aggregation
```

完成常见聚合图表。

未来可以考虑：

* SQL 结果保存为临时 DataFrame；
* 将中间结果注册为命名数据集；
* 图表工具接受结构化查询结果。

### 12.5 自动测试仍会执行部分真实工具

模型调用已经被 mock，但部分 Graph 测试仍会真实执行：

* CSV 读取；
* SQL；
* 图表创建；
* 文档检索。

因此测试速度还有进一步优化空间。

---

## 13. 面试中可能被问到的问题

### 13.1 为什么从手写循环改为 LangGraph？

回答要点：

* 手写循环适合简单原型；
* 随着节点、错误处理和多工具任务增加，流程难以维护；
* LangGraph 可以显式定义节点和条件边；
* State 可以集中保存工作流上下文；
* 更容易测试每个节点；
* 更方便增加重试、审批和持久化。

### 13.2 StateGraph、Node 和 Edge 分别是什么？

回答要点：

* `StateGraph` 定义整个有状态工作流；
* Node 是读取 State 并返回部分 State 更新的函数；
* Edge 定义节点执行顺序；
* Conditional Edge 根据 State 动态选择下一节点；
* `START` 和 `END` 分别表示工作流入口和出口。

### 13.3 为什么 Tool Decision 后面要增加 Validator？

回答要点：

* 大语言模型输出具有随机性；
* Prompt 不能保证 SQL 或参数永远正确；
* Validator 将关键约束写成确定性 Python 规则；
* 可以在工具执行前阻止无效或危险调用；
* 减少工具异常；
* 方便独立测试。

### 13.4 为什么不能只判断工具是否 `success=True`？

回答要点：

* 工具执行成功不代表完成了用户需求；
* `SELECT * LIMIT 5` 可以成功，但不能完成平均值统计；
* 需要检查工具参数是否符合任务语义；
* 因此在执行前进行 SQL 和图表参数验证。

### 13.5 为什么自动测试要 mock 模型？

回答要点：

* 在线模型输出不完全确定；
* 自动测试应稳定、快速、可重复；
* 测试 Graph 时应该固定模型输出；
* 模型实际效果通过单独的端到端测试评估；
* 单元测试和模型效果评估不应混为一体。

### 13.6 多工具循环如何避免死循环？

回答要点：

* 记录已经成功执行的工具；
* 每轮只选择尚未完成的工具；
* 设置 `max_steps`；
* 设置 `max_validation_retries`；
* 错误或达到上限后进入 Reporter。

### 13.7 为什么 `tool_calls` 只记录验证通过的调用？

回答要点：

* 被拦截的候选调用没有真正执行；
* 正式调用历史应该反映真实执行轨迹；
* 验证错误单独保存在 `validation_error`；
* 可以避免审计日志中混入未执行调用。

---

## 14. 本周最终成果

本周完成了 CausaLens Agent 的核心工作流重构。

项目已经从：

```text
手写的单轮或多轮工具循环
```

升级为：

```text
带状态管理、条件路由、确定性验证、
失败重试、结果检查和多工具循环的 LangGraph Agent
```

当前系统可以：

* 理解用户的 CSV 数据分析问题；
* 判断任务类型；
* 生成分析计划；
* 选择 Profile、SQL、Chart 或 RAG 工具；
* 验证模型生成的工具参数；
* 执行一个或多个工具；
* 根据执行历史判断是否继续；
* 汇总多个工具结果；
* 遵守只读和证据边界；
* 在失败时停止并说明原因。

本周相关测试和项目完整测试均已通过。

---

## 15. 下一周计划

下一阶段可以重点提升 Agent 的分析能力和项目展示效果：

1. 改进复杂任务的动态规划；
2. 增强 Result Checker 的任务完成度判断；
3. 支持 SQL 中间结果继续传递给图表工具；
4. 优化 Router，减少关键词依赖；
5. 增加更复杂的分析工具；
6. 增加回归、假设检验或因果分析能力边界；
7. 整理 README、项目结构图和演示案例；
8. 为求职面试准备项目介绍和架构讲解。

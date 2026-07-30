# CausaLens Architecture

## 1. Architecture Overview

CausaLens is a read-only data research agent for CSV datasets.

The system combines:

* a large language model for natural-language understanding and tool argument generation;
* deterministic Python tools for dataset inspection, SQL analysis, chart generation, and document retrieval;
* a LangGraph workflow for multi-step execution and state management;
* deterministic validation for safety and argument correctness;
* evidence-aware reporting for grounded final answers;
* automated end-to-end evaluation for regression testing.

The current system is designed around one central principle:

> Model outputs are treated as candidate actions rather than automatically trusted commands.

The language model may propose a tool call, SQL query, chart configuration, or final explanation. Before any tool is executed, deterministic Python logic checks whether the proposed action is safe and consistent with the user’s request.

The primary execution path is:

```text
User Question
    ↓
Intent Routing
    ↓
Analysis Planning
    ↓
Tool Selection
    ↓
Deterministic Validation
    ↓
Tool Execution
    ↓
Result Checking
    ↓
Evidence-Grounded Reporting
```

CausaLens currently supports the following task types:

```text
data_profile
sql_analysis
chart_analysis
rag_question
mixed_analysis
unsupported
```

A request containing more than one supported analytical requirement is classified as `mixed_analysis`.

For example:

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

requires both:

```text
retrieve_docs
run_readonly_sql
```

---

## 2. LangGraph Workflow

The main Agent workflow is implemented with LangGraph.

```text
START
  ↓
intent_router
  ├── unsupported request ───────────────→ reporter
  ↓
planner
  ↓
tool_decision
  ↓
tool_validator
  ├── validation failed, retry allowed ─→ tool_decision
  ├── retry limit reached ──────────────→ reporter
  └── validation passed
          ↓
    tool_executor
          ↓
    result_checker
      ├── more tools required ──────────→ tool_decision
      └── completed, failed, or limited → reporter
                                               ↓
                                              END
```

The workflow separates natural-language reasoning from deterministic execution control.

The model is not allowed to directly execute arbitrary actions. Instead, it proposes one tool call at a time. The graph then validates, executes, records, and checks the result before deciding whether another tool is needed.

This design makes multi-step analysis explicit and observable.

A mixed workflow may look like:

```text
intent_router
→ planner
→ tool_decision
→ retrieve_docs
→ result_checker
→ tool_decision
→ run_readonly_sql
→ result_checker
→ reporter
```

Each successfully executed tool increases `step_count` by one.

Validation retries do not increase `step_count`, because no tool has been executed yet.

---

## 3. Node Responsibilities

### 3.1 `intent_router_node`

The intent router classifies the user request into one of the supported task types.

It currently uses deterministic keyword-based rules.

Examples:

```text
这个数据集有哪些字段？
→ data_profile
```

```text
passed 字段是什么意思？
→ rag_question
```

```text
不同专业的平均 score 分别是多少？
→ sql_analysis
```

```text
请画出 study_hours 和 score 的散点图。
→ chart_analysis
```

```text
请说明 study_hours 的单位，并画出散点图。
→ mixed_analysis
```

Destructive requests are routed to:

```text
unsupported
```

For example:

```text
请删除 score 小于 60 的学生记录。
```

The router does not directly execute tools.

---

### 3.2 `planner_node`

The planner creates a high-level analysis plan based on the detected task type.

For a SQL analysis task, the plan may include:

```text
1. 识别分组字段和数值字段
2. 生成只读 SQL
3. 执行查询
4. 总结实际结果
```

For a mixed task, the plan may include:

```text
1. 拆分用户问题中的多个分析需求
2. 确定各步骤需要使用的工具
3. 按依赖顺序执行工具
4. 汇总多个工具的结果
```

The plan is descriptive. It does not itself execute any action.

---

### 3.3 `tool_decision_node`

The tool decision node determines which tool should be called next and asks the model to produce candidate arguments.

It considers:

* the user question;
* the current task type;
* the analysis plan;
* the required tools;
* tools already completed;
* validation feedback from previous attempts.

For mixed tasks, the required tools are inferred in a predefined order:

```text
1. retrieve_docs
2. profile_dataset
3. run_readonly_sql
4. create_chart
```

Only the tools actually required by the current question are included.

The node generates one candidate tool call per iteration.

For selected RAG tasks, deterministic normalization is applied to prevent query drift.

For example:

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

is reduced to the document subquestion:

```text
请解释 passed 字段含义
```

before calling `retrieve_docs`.

---

### 3.4 `tool_validator_node`

The validator checks the candidate tool call before execution.

Its purpose is to prevent:

* unsafe SQL;
* invalid columns;
* unsupported chart types;
* missing required arguments;
* tool arguments that do not satisfy the user request;
* temporary SQL aliases being treated as original CSV columns.

A tool call that fails validation is not added to the official `tool_calls` history.

The validator returns feedback to `tool_decision_node`, allowing the model to generate a corrected call.

Validation retries are bounded by a maximum retry count.

---

### 3.5 `tool_executor_node`

The executor runs only validated tool calls.

It invokes the registered Python implementation and converts the output into a structured tool result.

A tool result contains information such as:

```text
tool_name
success
result
error
```

After execution:

* the official tool call is stored;
* the tool result is stored;
* `step_count` increases;
* control passes to `result_checker_node`.

---

### 3.6 `result_checker_node`

The result checker determines whether the current task has enough successful tool results.

It compares:

```text
required_tools
```

with:

```text
successfully completed tools
```

If tools remain, the graph returns to `tool_decision_node`.

If all required tools have succeeded, the graph proceeds to `reporter_node`.

The result checker also stops the workflow when:

* the maximum number of steps is reached;
* a required tool fails;
* validation retries are exhausted;
* the task can no longer continue safely.

The current completion check mainly verifies whether the required tools executed successfully.

---

### 3.7 `reporter_node`

The reporter creates the final user-facing answer from actual tool evidence.

It receives:

* the original question;
* task type;
* analysis plan;
* tool calls;
* tool results;
* workflow errors;
* evidence rules.

The reporter must not invent information that is absent from the tool results.

Examples of enforced reporting boundaries include:

* field definitions and units require document evidence;
* SQL numbers must come from actual query results;
* chart descriptions must use returned fields and summaries;
* correlation must not be presented as causation;
* missing documentation must remain explicitly uncertain;
* histogram shape must not be inferred without supporting statistics.

The reporter always runs for supported and unsupported requests, allowing the workflow to produce a readable explanation even when analysis fails.

---

## 4. Shared Agent State

LangGraph nodes communicate through a shared `AgentState`.

Important fields include:

```text
dataset_path
question
task_type
plan
required_tools
workflow_status
pending_tool_call
tool_calls
tool_results
validation_error
validation_retry_count
max_validation_retries
step_count
max_steps
needs_more_tools
result_check_reason
final_answer
error
```

### 4.1 Core Input Fields

```text
dataset_path
question
```

These identify the CSV file and the user request.

---

### 4.2 Planning Fields

```text
task_type
plan
required_tools
```

These describe what kind of task is being performed and which tools are required.

---

### 4.3 Tool Execution Fields

```text
pending_tool_call
tool_calls
tool_results
```

`pending_tool_call` contains the current candidate call before execution.

`tool_calls` contains only calls that passed validation and were executed.

`tool_results` stores structured success or failure results.

---

### 4.4 Validation Fields

```text
validation_error
validation_retry_count
max_validation_retries
```

These fields support bounded correction attempts when model-generated arguments are invalid.

---

### 4.5 Workflow Control Fields

```text
workflow_status
step_count
max_steps
needs_more_tools
result_check_reason
```

These fields allow conditional graph routing and prevent infinite execution loops.

---

### 4.6 Final Output Fields

```text
final_answer
error
```

`final_answer` contains the user-facing response.

`error` stores workflow-level failure information when the analysis cannot be completed.

---

## 5. Tool System

CausaLens currently exposes the following main tools.

### 5.1 `profile_dataset`

The profiling tool inspects a CSV dataset and returns:

* row count;
* column count;
* column names;
* data types;
* missing values;
* numeric summaries.

It is used for questions such as:

```text
这个数据集有哪些字段？
```

The profiling tool describes observed dataset structure. It does not define business meanings or units.

---

### 5.2 `run_readonly_sql`

The SQL tool uses DuckDB to execute read-only queries over a CSV dataset.

The CSV is exposed to SQL using the table name:

```sql
dataset
```

Example:

```sql
SELECT
    major,
    AVG(score) AS avg_score
FROM dataset
GROUP BY major
ORDER BY avg_score DESC;
```

The tool returns structured result fields and rows.

Write operations are not supported.

---

### 5.3 `create_chart`

The chart tool currently supports:

```text
bar
scatter
histogram
line
```

Depending on the chart type, it may return:

* chart path;
* plotted row count;
* aggregated data;
* numeric summary;
* variable ranges;
* Pearson correlation.

For example, a scatter plot may return:

```text
point_count
x_min
x_max
y_min
y_max
pearson_correlation
```

The chart tool reads the original CSV dataset.

It does not currently consume arbitrary SQL result tables.

---

### 5.4 `retrieve_docs`

The RAG tool retrieves relevant chunks from project knowledge documents.

Current knowledge files include:

```text
knowledge/data_dictionary.md
knowledge/project_background.md
knowledge/analysis_rules.md
```

It is used for:

* field definitions;
* units;
* valid values;
* coding meanings;
* dataset background;
* analysis rules;
* causal-analysis boundaries.

For example:

```text
passed 字段是什么意思？
```

should be answered using the documented `passed` section rather than guessing from the field name.

---

## 6. Deterministic Tool Validation

The language model generates candidate tool arguments, but deterministic Python logic decides whether those arguments may be executed.

This creates a separation between:

```text
semantic proposal
```

and:

```text
safe execution
```

Validation is applied before tool execution.

---

### 6.1 SQL Validation

The SQL validator checks that:

* the query is read-only;
* the query begins with `SELECT` or `WITH`;
* write and schema-changing statements are rejected;
* average-value tasks use `AVG`;
* count tasks use `COUNT`;
* grouped analysis uses `GROUP BY`;
* `SELECT *` is not used as a substitute for the requested analysis;
* referenced columns exist in the current dataset;
* the query can be bound successfully using DuckDB;
* the CSV table name is `dataset`.

For example, this query is not sufficient for an average-by-group request:

```sql
SELECT *
FROM dataset
LIMIT 5;
```

A suitable query would be:

```sql
SELECT
    major,
    AVG(score) AS avg_score
FROM dataset
GROUP BY major;
```

A query that references an invented column is rejected.

For example, if the dataset contains `major`, the following should not be executed:

```sql
SELECT
    专业,
    AVG(score)
FROM dataset
GROUP BY 专业;
```

The SQL validator also uses DuckDB binding to detect invalid columns before official execution.

---

### 6.2 Chart Validation

The chart validator checks that:

* the dataset path is present;
* the chart type is supported;
* referenced columns exist in the original CSV;
* required fields for the chart type are present;
* scatter plots include both x and y columns;
* histogram parameters match a numeric distribution request;
* bar-chart aggregation matches the requested calculation;
* temporary SQL aliases are not treated as original CSV fields.

For example, `avg_score` may be created by SQL:

```sql
AVG(score) AS avg_score
```

but it is not an original CSV column.

Therefore, the chart tool should receive:

```python
{
    "chart_type": "bar",
    "x_column": "major",
    "y_column": "score",
    "aggregation": "mean",
}
```

rather than:

```python
{
    "chart_type": "bar",
    "x_column": "major",
    "y_column": "avg_score",
}
```

---

## 7. Multi-Tool Execution

Mixed tasks are completed through repeated graph iterations.

Example:

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

The required tool sequence is:

```text
retrieve_docs
→ run_readonly_sql
```

The workflow proceeds as follows:

```text
1. retrieve_docs is selected and validated.
2. Documentation evidence is retrieved.
3. result_checker detects that SQL is still required.
4. run_readonly_sql is selected and validated.
5. The grouped average is calculated.
6. result_checker confirms that all required tools succeeded.
7. reporter combines document and SQL evidence.
```

Another mixed task is:

```text
请说明 study_hours 的单位，并画出 study_hours 和 score 的散点图。
```

Its tool sequence is:

```text
retrieve_docs
→ create_chart
```

A SQL and chart task may use:

```text
run_readonly_sql
→ create_chart
```

The current implementation uses a predefined tool dependency order rather than a fully dynamic planner.

Each tool type is normally executed at most once per workflow.

---

## 8. Evidence Boundaries

CausaLens distinguishes among different evidence sources.

### Dataset Profile Evidence

Dataset profiling may establish:

* field names;
* data types;
* row counts;
* missing-value counts;
* observed numeric ranges.

It cannot establish undocumented business meanings or units.

---

### SQL Evidence

SQL may establish:

* grouped averages;
* counts;
* rankings;
* filtered observations;
* actual values calculated from the dataset.

SQL results describe the current dataset only.

They do not automatically establish causal relationships.

---

### Chart Evidence

Chart tools may establish:

* plotted variables;
* plotted row count;
* numeric ranges;
* chart file path;
* calculated summaries;
* Pearson correlation.

A correlation coefficient describes an observed statistical relationship.

It does not prove that one variable causes another.

---

### RAG Evidence

RAG documents may establish:

* field definitions;
* coding meanings;
* units;
* valid value ranges;
* project background;
* analysis restrictions.

If the documentation does not define a fact, the system should report that it cannot be confirmed.

---

### Reporting Constraints

The reporter must not:

* invent field definitions;
* invent measurement units;
* invent school or organization names;
* infer undocumented thresholds;
* describe correlation as causation;
* claim a histogram is skewed or symmetric without supporting evidence;
* create fields or aliases that were not returned by tools;
* hide uncertainty stated in the documentation.

For a histogram that returns only:

```text
count
mean
median
min
max
```

the reporter may describe those statistics and the observed range.

It must not infer:

```text
left skew
right skew
symmetry
concentration
long tail
peak interval
```

unless the tool returns additional evidence supporting those claims.

---

## 9. Error Handling and Retry Limits

CausaLens uses bounded execution controls.

### Validation Retry Limit

When a candidate tool call fails validation:

```text
tool_validator
→ validation feedback
→ tool_decision
```

The model may propose corrected arguments.

The number of retries is limited by:

```text
max_validation_retries
```

When the limit is reached, the workflow moves to the reporter.

---

### Maximum Tool Steps

The workflow also tracks:

```text
step_count
max_steps
```

Only actual tool executions increase `step_count`.

This prevents infinite multi-tool loops.

---

### Tool Failure Handling

When a tool returns:

```text
success = False
```

the failure is recorded in `tool_results`.

The reporter receives the error and produces a user-facing explanation.

The CLI displays the final explanation before returning a nonzero exit code.

---

### Unsupported Requests

Destructive requests do not enter the tool loop.

For example:

```text
请删除 score 小于 60 的学生记录。
```

is routed directly to:

```text
unsupported
→ reporter
```

No SQL or data modification tool is executed.

---

### Model Client Lifecycle

Each model request creates an independent OpenAI-compatible client.

The client is managed within the request function and closed after the synchronous response is returned.

This avoids reusing a closed client across multiple LangGraph iterations.

---

## 10. Design Trade-Offs

### Keyword-Based Routing

The current router is deterministic and easy to test.

Advantages:

* transparent behavior;
* low latency;
* no extra model request;
* easy regression testing.

Limitations:

* wording coverage depends on keyword design;
* complex paraphrases may not be recognized;
* overlapping intentions require explicit rules.

---

### Predefined Tool Order

Mixed tasks currently use a predefined tool order.

Advantages:

* predictable execution;
* simple dependency handling;
* easier debugging;
* reduced risk of unnecessary calls.

Limitations:

* dependencies are not fully dynamic;
* some future tasks may need a different order;
* repeated use of the same tool is limited.

---

### Model for Semantics, Python for Control

The language model handles:

* understanding natural-language questions;
* proposing tool arguments;
* generating read-only SQL;
* choosing chart parameters;
* writing final answers.

Deterministic Python handles:

* routing rules;
* destructive-request rejection;
* SQL validation;
* chart validation;
* column checking;
* retry limits;
* maximum steps;
* actual tool execution;
* state transitions;
* structured evidence;
* report formatting.

This division prevents core safety behavior from depending entirely on prompt compliance.

---

### Original CSV as Chart Input

The chart tool currently reads the original CSV directly.

This makes chart generation simple and reliable.

However, it means SQL result tables cannot yet be passed directly into the chart tool.

For grouped charts, the chart tool performs its own aggregation over the original fields.

---

### Deterministic Markdown Reporting

Markdown reports are generated from the completed `AgentState`.

The report generator:

* does not call the language model again;
* does not rerun tools;
* does not infer new conclusions;
* preserves raw tool arguments and results;
* adds a deterministic evidence summary.

This makes reports reproducible and auditable.

---

## 11. Current Limitations

The current MVP has the following limitations:

* intent routing still relies mainly on keywords;
* mixed-task tool dependencies use a predefined order;
* each tool type is normally executed at most once;
* result checking mainly verifies successful tool completion;
* semantic completeness checking is limited;
* chart generation reads the original CSV rather than SQL result tables;
* the project focuses on CSV datasets;
* the current RAG knowledge base contains project-specific Markdown documents;
* complex causal analysis is not yet implemented;
* regression and hypothesis-testing tools are not yet included;
* the MVP evaluation uses a fixed demonstration dataset and a limited set of core cases;
* some real end-to-end runs depend on external model availability;
* local embedding initialization can make RAG runs slower than pure unit tests.

The current system should therefore be described as:

> A validated read-only CSV research Agent MVP with profiling, SQL, charting, RAG, multi-tool workflows, structured reporting, and automated core-case evaluation.

It should not be described as a universal data-analysis system for every CSV dataset or every statistical question.

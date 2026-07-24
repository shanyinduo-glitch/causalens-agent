# CausaLens Agent

CausaLens is a read-only data research agent for CSV datasets.

It combines large language models, deterministic Python tools, retrieval-augmented generation, and a LangGraph workflow to inspect datasets, run SQL analysis, create charts, retrieve project documentation, and complete multi-step data analysis tasks.

## Current Capabilities

CausaLens currently supports:

* CSV dataset profiling
* Read-only SQL analysis
* Bar charts, scatter plots, histograms, and line charts
* Retrieval-augmented generation over project knowledge documents
* Natural-language task routing
* Multi-tool analysis workflows
* Deterministic tool-call validation
* Tool failure handling and retry limits
* Evidence-aware final reports
* Structured tool results and execution logs

The current version focuses on safe, read-only analysis. Data deletion, modification, and write SQL operations are not supported.

## Example Questions

### Dataset Profiling

```text
这个数据集有哪些字段？
```

The agent uses `profile_dataset` to inspect:

* row and column counts
* column names
* data types
* missing values
* numeric summaries

### SQL Analysis

```text
不同专业的平均 score 是多少？
```

The agent generates and executes a read-only SQL query such as:

```sql
SELECT
    major,
    AVG(score) AS avg_score
FROM dataset
GROUP BY major
ORDER BY major;
```

### Chart Analysis

```text
请画出 study_hours 和 score 的散点图。
```

The agent creates a chart and returns:

* chart file path
* plotted row count
* numeric ranges
* relationship summary
* Pearson correlation when applicable

Correlation is reported as an observed relationship and is not interpreted as causation.

### Knowledge-Base Question

```text
passed 字段是什么意思？
```

The agent retrieves evidence from project documents such as `data_dictionary.md` and answers using the documented field definition.

### Multi-Tool Analysis

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

This task uses:

```text
retrieve_docs
→ run_readonly_sql
→ reporter
```

Another example:

```text
请统计不同专业的平均 score，并画出柱状图。
```

This task uses:

```text
run_readonly_sql
→ create_chart
→ reporter
```

## LangGraph Workflow

The current Agent workflow is implemented with LangGraph.

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

### Node Responsibilities

* `intent_router_node`: classifies the user request
* `planner_node`: creates a high-level analysis plan
* `tool_decision_node`: generates the next candidate tool call
* `tool_validator_node`: validates tool arguments before execution
* `tool_executor_node`: executes validated Python tools
* `result_checker_node`: decides whether another tool is required
* `reporter_node`: produces the final evidence-grounded answer

## Supported Task Types

```text
data_profile
sql_analysis
chart_analysis
rag_question
mixed_analysis
unsupported
```

When a question contains multiple analysis requirements, it is classified as `mixed_analysis`.

## Tool Validation

Large language model outputs are treated as candidate actions rather than automatically trusted commands.

Before execution, CausaLens applies deterministic Python validation.

### SQL Validation

The SQL validator checks:

* the query is read-only
* the query begins with `SELECT` or `WITH`
* write and schema-changing statements are rejected
* average-value tasks use `AVG`
* count tasks use `COUNT`
* grouped analysis uses `GROUP BY`
* `SELECT *` is not used as a substitute for statistical analysis
* referenced columns exist in the current dataset
* the query can be bound to the dataset using DuckDB

For example, this query is rejected for an average-by-group task:

```sql
SELECT *
FROM dataset
LIMIT 5;
```

This query is also rejected when the real column is named `major`:

```sql
SELECT
    专业,
    AVG(score)
FROM dataset
GROUP BY 专业;
```

### Chart Validation

The chart validator checks:

* the dataset path is present
* the chart type is supported
* referenced columns exist in the original CSV
* scatter plots contain both x and y columns
* histogram and bar-chart parameters match the request
* SQL aliases are not incorrectly treated as original CSV columns

For example, `avg_score` may be a temporary SQL alias but not an original CSV field.

The chart tool should therefore receive:

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

## Shared Agent State

LangGraph nodes communicate through a shared `AgentState`.

Important fields include:

* `dataset_path`
* `question`
* `task_type`
* `plan`
* `required_tools`
* `workflow_status`
* `pending_tool_call`
* `tool_calls`
* `tool_results`
* `validation_error`
* `validation_retry_count`
* `max_validation_retries`
* `step_count`
* `max_steps`
* `needs_more_tools`
* `result_check_reason`
* `final_answer`
* `error`

Only tool calls that pass validation are added to the official `tool_calls` history.

Validation retries do not increase `step_count`. The step count increases only when a tool is actually executed.

## Project Structure

```text
causalens-agent/
├── data/
│   └── student_performance.csv
├── docs/
│   └── week_04.md
├── knowledge/
│   ├── analysis_rules.md
│   ├── data_dictionary.md
│   └── project_background.md
├── outputs/
│   └── charts/
├── scripts/
├── src/
│   └── causalens/
│       ├── agent/
│       │   ├── graph.py
│       │   ├── nodes.py
│       │   ├── qwen_client.py
│       │   ├── state.py
│       │   ├── tool_loop.py
│       │   └── tool_registry.py
│       ├── rag/
│       ├── tools/
│       ├── observability.py
│       └── schemas.py
└── tests/
    ├── test_graph.py
    └── test_nodes.py
```

The project contains additional scripts and tests beyond the abbreviated structure shown above.

## Running the LangGraph Workflow

In PowerShell, make the source directory importable:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Example execution:

```powershell
python -c "from causalens.agent.graph import run_causalens_graph; result = run_causalens_graph('data/student_performance.csv', '请统计不同专业的平均 score，并画出柱状图。'); print(result.get('final_answer'))"
```

The returned final state can also contain:

* selected task type
* required tools
* validated tool calls
* tool results
* validation feedback
* result-check reason
* final answer
* error information
* executed step count

## Testing

Run the complete test suite:

```powershell
pytest -v
```

Run only the LangGraph node and workflow tests:

```powershell
pytest -v tests/test_nodes.py tests/test_graph.py
```

The LangGraph node and graph suite currently contains 30 passing tests.

Online model calls are mocked in automated graph tests so that the tests remain deterministic and repeatable. Real model behavior is evaluated separately through manual end-to-end runs.

## Design Principles

### Read-Only by Default

CausaLens does not execute destructive data operations.

Requests involving deletion, updates, file overwrites, or write SQL are routed to `unsupported`.

### Evidence Boundaries

The final answer must distinguish between different evidence sources:

* dataset profiling describes observed dataset structure
* SQL describes actual query results
* chart summaries describe generated visualization data
* RAG documents define field meanings and project rules

The Agent must not infer undocumented field meanings, units, labels, or causal relationships.

### Model and Python Responsibilities

The language model is responsible for:

* understanding the question
* proposing tool arguments
* generating read-only SQL
* selecting chart parameters
* writing the final response

Deterministic Python code is responsible for:

* safety rules
* field validation
* SQL validation
* retry limits
* maximum execution steps
* tool execution
* state transitions
* logging and structured results

## Current Limitations

* Intent routing still relies mainly on keywords.
* Tool dependencies use a predefined order.
* Result checking primarily verifies successful tool completion.
* Chart generation currently reads the original CSV rather than arbitrary SQL result tables.
* Some integration tests still execute real local tools and can take longer than pure unit tests.
* The current version focuses on CSV datasets.

## Development Progress

### Week 1

* project initialization
* CSV profiling
* structured tool results
* basic command-line Agent

### Week 2

* read-only SQL analysis
* tool registration
* safety validation
* expanded automated tests

### Week 3

* chart tools
* project knowledge documents
* retrieval-augmented generation
* evidence-aware answers

### Week 4

* LangGraph workflow
* shared Agent state
* conditional routing
* multi-tool execution
* deterministic tool validation
* validation retries
* result checking
* updated graph and node tests

Detailed Week 4 documentation is available in:

```text
docs/week_04.md
```

## Planned Improvements

Potential next steps include:

* model-assisted intent routing
* dynamic tool dependency planning
* stronger semantic completion checking
* passing SQL result tables directly into chart generation
* regression and hypothesis-testing tools
* clearer causal-analysis boundaries
* improved command-line and user-interface experience
* faster fully mocked integration tests

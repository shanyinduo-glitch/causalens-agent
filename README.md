# CausaLens Agent

CausaLens is a validated, read-only data research Agent for CSV datasets.

It combines large language models, deterministic Python tools, retrieval-augmented generation, and a LangGraph workflow to:

* inspect CSV datasets;
* execute read-only SQL analysis;
* generate charts;
* retrieve project documentation;
* complete multi-tool analysis tasks;
* produce evidence-grounded answers;
* save structured Markdown reports.

The current MVP focuses on safe and auditable CSV analysis. It does not modify, delete, overwrite, or update source data.

---

## Key Features

* CSV dataset profiling
* Read-only SQL analysis with DuckDB
* Bar charts, scatter plots, histograms, and line charts
* Retrieval-augmented generation over project knowledge documents
* LangGraph-based multi-step workflows
* Mixed RAG, SQL, and chart tasks
* Deterministic tool-call validation
* Invalid-field and unsafe-SQL rejection
* Bounded retries and maximum execution steps
* Evidence-aware final answers
* Structured Markdown analysis reports
* Automated end-to-end MVP evaluation
* Explicit correlation-versus-causation boundaries

---

## Example Questions

### Dataset Profiling

```text
这个数据集有哪些字段？
```

### SQL Analysis

```text
不同专业的平均 score 分别是多少？
```

### Chart Analysis

```text
请画出 study_hours 和 score 的散点图。
```

### Documentation Retrieval

```text
passed 字段是什么意思？
```

### RAG and SQL

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

### RAG and Chart

```text
请说明 study_hours 的单位，并画出 study_hours 和 score 的散点图。
```

### SQL and Chart

```text
请统计不同专业的平均 score，并画出柱状图。
```

### Unsupported Request

```text
请删除 score 小于 60 的学生记录。
```

Destructive requests are rejected before tool execution.

---

## Architecture Overview

CausaLens uses the following LangGraph workflow:

```text
START
  ↓
Intent Router
  ↓
Planner
  ↓
Tool Decision
  ↓
Tool Validator
  ↓
Tool Executor
  ↓
Result Checker
  ├── more tools required → Tool Decision
  └── completed or failed → Reporter
                                  ↓
                                 END
```

Unsupported or destructive requests are routed directly to the Reporter without executing tools.

The workflow currently supports:

```text
data_profile
sql_analysis
chart_analysis
rag_question
mixed_analysis
unsupported
```

The language model proposes semantic actions such as SQL queries and chart arguments.

Deterministic Python code controls:

* read-only safety;
* field validation;
* SQL validation;
* chart validation;
* retry limits;
* execution limits;
* tool execution;
* graph transitions;
* evidence formatting;
* Markdown report generation.

Detailed architecture documentation is available in [docs/architecture.md](docs/architecture.md).

---

## Main Tools

### `profile_dataset`

Returns information such as:

* row and column counts;
* column names;
* data types;
* missing-value counts;
* numeric summaries.

### `run_readonly_sql`

Executes read-only SQL over the CSV dataset through DuckDB.

The table name used in SQL is:

```sql
dataset
```

### `create_chart`

Supports:

```text
bar
scatter
histogram
line
```

Depending on the chart type, it may return:

* chart path;
* plotted row count;
* numeric summaries;
* variable ranges;
* Pearson correlation.

### `retrieve_docs`

Retrieves evidence from:

```text
knowledge/data_dictionary.md
knowledge/project_background.md
knowledge/analysis_rules.md
```

It is used for field definitions, units, coding rules, dataset background, and analytical boundaries.

---

## Quick Start

### 1. Activate the Environment

```powershell
conda activate langchain_env
```

Move to the project root:

```powershell
cd E:\Projects\causalens-agent
```

### 2. Configure the Source Path

```powershell
$env:PYTHONPATH = "$PWD\src"
```

This variable applies only to the current PowerShell session.

### 3. Configure the Model API Key

CausaLens reads the Qwen API key from:

```text
DASHSCOPE_API_KEY
```

Optional environment variables include:

```text
DASHSCOPE_BASE_URL
MODEL_NAME
```

Do not store API keys in source code or commit them to Git.

### 4. Run a Basic Analysis

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "不同专业的平均 score 分别是多少？"
```

The CLI displays:

* user question;
* detected task type;
* workflow status;
* analysis plan;
* validated tool calls;
* final answer;
* evidence from actual tool results.

### 5. Run a Mixed Workflow

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "请解释 passed 字段含义，并统计不同 passed 值的平均 score。"
```

Expected tool order:

```text
retrieve_docs
→ run_readonly_sql
```

Another mixed example:

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "请说明 study_hours 的单位，并画出 study_hours 和 score 的散点图。"
```

Expected tool order:

```text
retrieve_docs
→ create_chart
```

---

## Markdown Reports

Add the optional `--report-path` argument to save a structured report:

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "不同专业的平均 score 分别是多少？" `
  --report-path outputs/reports/major_score_report.md
```

The generated report contains:

* execution summary;
* user question;
* analysis plan;
* readable evidence summary;
* tool arguments;
* raw structured tool results;
* final answer;
* workflow errors when applicable.

Report generation is deterministic.

It formats the completed Agent state and does not make another model request or rerun tools.

---

## Safety and Evidence Boundaries

CausaLens is read-only by default.

The system rejects requests involving:

* deletion;
* updates;
* overwriting data;
* schema changes;
* write SQL operations.

Different claims require different evidence sources:

* dataset profiling describes observed CSV structure;
* SQL provides calculated results from the current dataset;
* chart tools provide visualization-related summaries;
* RAG documents define field meanings, units, and project rules.

The Agent must not:

* invent undocumented field definitions;
* invent units or thresholds;
* use invalid dataset columns;
* treat SQL aliases as original CSV fields;
* infer histogram shape without supporting evidence;
* interpret correlation as causation.

For example, a Pearson correlation describes an observed association. It does not prove that one variable causes another.

---

## Automated Tests

Run the complete test suite:

```powershell
pytest
```

The tests cover:

* dataset tools;
* SQL safety;
* chart behavior;
* RAG retrieval;
* intent routing;
* LangGraph nodes and transitions;
* validation retries;
* result checking;
* CLI behavior;
* evidence formatting;
* Markdown reporting.

Model calls are mocked where deterministic and repeatable testing is required.

---

## MVP Evaluation

The end-to-end evaluation cases are stored in:

```text
evals/mvp_cases.json
```

Run the complete MVP evaluation:

```powershell
python scripts/run_mvp_evals.py
```

Run one case:

```powershell
python scripts/run_mvp_evals.py `
  --case-id mvp_07_mixed_rag_sql
```

The runner checks:

* task type;
* workflow status;
* tool execution order;
* required answer facts;
* forbidden conclusions;
* non-empty final answers.

Evaluation reports are written to:

```text
outputs/evals/latest_results.json
outputs/evals/latest_summary.md
```

Current result on the fixed demonstration dataset and ten core MVP cases:

```text
Passed: 10
Failed: 0
Pass rate: 100.0%
```

This result applies only to the current dataset and evaluation suite. It does not imply support for every CSV dataset or analytical question.

---

## Project Structure

```text
causalens-agent/
├── data/
│   └── student_performance.csv
├── docs/
│   ├── architecture.md
│   ├── mvp_demo.md
│   ├── mvp_design.md
│   ├── week_01.md
│   ├── week_02.md
│   ├── week_03.md
│   ├── week_04.md
│   └── week_05.md
├── evals/
│   ├── mvp_cases.json
│   ├── week_02_cases.jsonl
│   └── week_03_rag_cases.jsonl
├── knowledge/
│   ├── analysis_rules.md
│   ├── data_dictionary.md
│   └── project_background.md
├── outputs/
│   ├── charts/
│   ├── evals/
│   └── reports/
├── scripts/
│   └── run_mvp_evals.py
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
│       ├── reporting/
│       │   └── markdown_report.py
│       ├── tools/
│       ├── cli.py
│       ├── response_evidence.py
│       └── schemas.py
└── tests/
```

The structure above is abbreviated. The repository contains additional scripts and tests.

---

## Current Limitations

* Intent routing mainly relies on deterministic keywords.
* Mixed-task tools use a predefined dependency order.
* Each tool type is normally executed at most once per workflow.
* Result checking mainly verifies successful tool completion.
* Chart generation reads the original CSV rather than arbitrary SQL result tables.
* The current project focuses on CSV datasets.
* The RAG knowledge base contains project-specific documents.
* Complex causal inference is not yet implemented.
* Regression and hypothesis-testing tools are not currently included.
* Real end-to-end runs depend on external model availability.
* The MVP evaluation uses a fixed dataset and a limited core-case suite.
* No graphical user interface is currently included.

---

## Documentation

### Product and Architecture

* [MVP Design](docs/mvp_design.md)
* [Architecture](docs/architecture.md)
* [MVP Demo and Evaluation](docs/mvp_demo.md)

### Development Notes

* [Week 1 Development Notes](docs/week_01.md)
* [Week 2 Development Notes](docs/week_02.md)
* [Week 3 Development Notes](docs/week_03.md)
* [Week 4 Development Notes](docs/week_04.md)
* [Week 5 Development Notes](docs/week_05.md)

---

## Development Summary

CausaLens was developed incrementally:

```text
Standalone CSV tools
→ hand-written tool loop
→ RAG and chart integration
→ LangGraph workflow
→ deterministic validation
→ mixed-tool execution
→ unified CLI
→ automated MVP evaluation
→ structured Markdown reports
```

The earlier `tool_loop.py` remains in the repository to preserve the project’s development history and provide a simpler comparison with the formal LangGraph workflow.

---

## Possible Future Work

Potential extensions include:

* hybrid or model-assisted intent routing;
* dynamic tool dependency planning;
* stronger semantic completion checking;
* direct charting from SQL result tables;
* regression and hypothesis-testing tools;
* explicit causal-analysis workflows;
* additional datasets and evaluation cases;
* embedding and model-call caching;
* a lightweight Web interface.

These extensions are outside the current MVP scope.

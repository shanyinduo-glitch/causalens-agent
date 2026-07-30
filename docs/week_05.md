# Week 5: MVP Integration, Evaluation, and Project Finalization

## 1. Week 5 Goals

The main goal of Week 5 was to turn the existing CausaLens components into a complete and demonstrable MVP.

Before Week 5, the project already had:

* CSV profiling;
* read-only SQL analysis;
* chart generation;
* project-document retrieval;
* a LangGraph workflow;
* deterministic tool validation;
* multi-step tool execution.

However, these components still needed to be integrated into a stable product entry point and validated as a complete system.

The Week 5 goals were therefore:

1. replace the old CLI execution path with the formal LangGraph workflow;
2. validate single-tool and mixed-tool tasks through the real CLI;
3. fix evidence and routing problems found during end-to-end runs;
4. build a repeatable MVP evaluation suite;
5. generate structured Markdown analysis reports;
6. organize the repository documentation;
7. prepare the project for demonstrations, interviews, and future extension.

---

## 2. Completed Work

## 2.1 Unified LangGraph CLI

The command-line interface was updated to use:

```text
run_causalens_graph(
    dataset_path=dataset_path,
    question=question,
)
```

instead of the earlier hand-written tool loop.

The formal CLI now displays:

* the user question;
* detected task type;
* workflow status;
* analysis plan;
* validated tool calls;
* final answer;
* evidence extracted from actual tool results;
* workflow errors when applicable.

The primary command is:

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "不同专业的平均 score 分别是多少？"
```

The CLI remains backward-compatible: users who do not request a report receive the same terminal-based workflow.

---

## 2.2 Tool-Call and Tool-Result Compatibility

During CLI integration, the final LangGraph state contained tool calls and results in dictionary form.

The existing CLI display and evidence code had originally expected object or Pydantic-model forms.

Compatibility logic was added so that:

* dictionary-form tool calls can be displayed;
* older object-form tool calls remain supported;
* dictionary-form tool results are converted into `ToolResult` models;
* evidence-table generation continues to use structured results.

This prevented the display layer from depending on only one internal state representation.

---

## 2.3 Single-Tool CLI Validation

The formal CLI was tested with the main single-tool task types.

### Dataset Profiling

Question:

```text
这个数据集有哪些字段？
```

Expected workflow:

```text
data_profile
→ profile_dataset
→ reporter
```

The CLI successfully displayed the dataset structure, including 40 rows and eight fields.

### Documentation Retrieval

Question:

```text
passed 字段是什么意思？
```

Expected workflow:

```text
rag_question
→ retrieve_docs
→ reporter
```

The final answer correctly explained the binary coding of `passed` and did not invent a passing-score threshold.

### SQL Analysis

Question:

```text
不同专业的平均 score 分别是多少？
```

Expected workflow:

```text
sql_analysis
→ run_readonly_sql
→ reporter
```

The resulting averages were approximately:

```text
cs: 81.23
math: 79.57
stats: 74.69
```

### Histogram Generation

Question:

```text
请画出 score 的分布图。
```

Expected workflow:

```text
chart_analysis
→ create_chart
→ reporter
```

The chart tool returned the histogram path and the following summary:

```text
count = 40
mean = 78.525
median = 79.5
min = 52
max = 98
```

### Scatter-Plot Generation

Question:

```text
请画出 study_hours 和 score 的散点图。
```

Expected workflow:

```text
chart_analysis
→ create_chart
→ reporter
```

The tool returned:

```text
point_count = 40
x range = 3 to 18
y range = 52 to 98
Pearson correlation ≈ 0.9874
```

The final answer explicitly distinguished correlation from causation.

---

## 2.4 Mixed-Tool Workflow Validation

Three multi-tool workflows were validated through the formal CLI.

### RAG and SQL

Question:

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

Tool order:

```text
retrieve_docs
→ run_readonly_sql
```

The final answer combined:

* the documented meaning of `passed`;
* the absence of a documented passing-score threshold;
* the average score for each `passed` group.

Results:

```text
passed = 0: average score ≈ 62.09
passed = 1: average score ≈ 84.76
```

The answer did not interpret the group difference as a causal effect.

---

### RAG and Chart

Question:

```text
请说明 study_hours 的单位，并画出 study_hours 和 score 的散点图。
```

Tool order:

```text
retrieve_docs
→ create_chart
```

The final answer included:

* `study_hours` means average weekly study time;
* the unit is hours per week;
* the chart path;
* the plotted row count;
* variable ranges;
* Pearson correlation;
* a correlation-versus-causation warning.

---

### SQL and Chart

Question:

```text
请统计不同专业的平均 score，并画出柱状图。
```

Tool order:

```text
run_readonly_sql
→ create_chart
```

The SQL tool calculated grouped means.

The chart tool used:

```python
{
    "chart_type": "bar",
    "x_column": "major",
    "y_column": "score",
    "aggregation": "mean",
}
```

The temporary SQL alias `avg_score` was not incorrectly passed as an original CSV column.

---

## 2.5 Unsupported Request Handling

The following destructive request was tested:

```text
请删除 score 小于 60 的学生记录。
```

The workflow correctly classified it as:

```text
unsupported
```

Execution path:

```text
intent_router
→ reporter
```

No tools were called.

The system did not generate or execute destructive SQL and clearly explained that CausaLens supports read-only analysis only.

---

## 2.6 End-to-End MVP Evaluation Suite

A formal evaluation suite was created in:

```text
evals/mvp_cases.json
```

The evaluation runner was created in:

```text
scripts/run_mvp_evals.py
```

The runner:

* loads the fixed MVP cases;
* calls the formal LangGraph workflow;
* checks the task type;
* checks workflow completion;
* checks exact tool order;
* checks required answer terms;
* checks forbidden answer terms;
* continues after individual case failures;
* generates JSON and Markdown reports;
* returns a nonzero process exit code when any case fails.

Evaluation outputs are saved to:

```text
outputs/evals/latest_results.json
outputs/evals/latest_summary.md
```

The ten current MVP cases cover:

1. dataset profiling;
2. RAG field definition;
3. RAG unit definition;
4. SQL grouped average;
5. histogram generation;
6. scatter-plot generation;
7. RAG and SQL workflow;
8. RAG and chart workflow;
9. SQL and chart workflow;
10. destructive-request rejection.

Final result:

```text
Passed: 10
Failed: 0
Pass rate: 100.0%
```

This result applies to the current fixed dataset and ten core MVP cases.

It does not imply universal support for every CSV dataset or analytical question.

---

## 2.7 Structured Markdown Reports

A deterministic reporting module was created in:

```text
src/causalens/reporting/
├── __init__.py
└── markdown_report.py
```

The report generator converts the completed `AgentState` into a structured Markdown document.

Reports contain:

* execution summary;
* dataset path;
* user question;
* task type;
* workflow status;
* analysis plan;
* evidence summary;
* tool calls and arguments;
* raw structured tool results;
* final answer;
* workflow error information.

The report generator does not:

* call the language model again;
* execute tools again;
* calculate new statistics;
* infer new conclusions.

This keeps reports deterministic and auditable.

The CLI now supports:

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "不同专业的平均 score 分别是多少？" `
  --report-path outputs/reports/major_score_report.md
```

When `--report-path` is omitted, the original terminal-only behavior remains unchanged.

---

## 2.8 Evidence Summary

The Markdown report contains both:

```text
Evidence Summary
```

and:

```text
Tool Execution
```

The evidence summary is designed for readability.

It extracts facts such as:

* row and column counts;
* retrieved document sources;
* SQL result row counts and fields;
* chart type and output path;
* numeric summaries;
* relationship summaries.

The full tool arguments and raw results remain available under `Tool Execution` for debugging and audit.

The evidence summary is deterministic and does not add semantic interpretations such as:

* strong or weak correlation;
* skewness;
* symmetry;
* causal effects.

---

## 2.9 Documentation Organization

The repository documentation was reorganized into separate responsibilities.

```text
README.md
→ project overview and quick start

docs/mvp_design.md
→ MVP scope, product decisions, and acceptance criteria

docs/architecture.md
→ LangGraph workflow, state, validation, tools, and trade-offs

docs/mvp_demo.md
→ reproducible commands, demonstrations, evaluation, and troubleshooting

docs/week_01.md to docs/week_05.md
→ incremental development history
```

This prevents README from becoming an overly detailed technical manual.

---

## 3. Problems Encountered

## 3.1 RAG Query Drift

### Problem

For the question:

```text
passed 字段是什么意思？
```

the model generated a rewritten query similar to:

```text
student_performance.csv passed 字段 定义 含义
```

The added dataset name and stacked keywords reduced retrieval quality.

The tool itself returned successfully, but the relevant `passed` document chunk was not retrieved.

### Key Lesson

```text
tool success
```

does not necessarily mean:

```text
task success
```

A retrieval tool may execute without errors while still returning insufficient evidence.

### Solution

For single RAG questions, the system now preserves the original user question as the retrieval query.

For mixed questions, a deterministic subquestion extractor removes the statistical or chart request.

Example:

```text
请解释 passed 字段含义，并统计不同 passed 值的平均 score。
```

becomes:

```text
请解释 passed 字段含义
```

for the RAG tool.

A regression test was added to prevent future query drift.

---

## 3.2 Intent Routing for Unit Questions

### Problem

The mixed question:

```text
请说明 study_hours 的单位，并画出 study_hours 和 score 的散点图。
```

was initially classified as:

```text
chart_analysis
```

instead of:

```text
mixed_analysis
```

The documentation-intent keyword list recognized:

```text
单位是什么
```

but not:

```text
的单位
```

### Solution

The documentation keyword coverage was expanded to include common unit expressions such as:

```text
的单位
单位是
单位为
```

The same keyword list is reused by both the intent router and required-tool inference, so the change fixed both behaviors.

Regression tests were added for:

```text
unit question
→ rag_question
```

```text
chart question
→ chart_analysis
```

```text
unit plus chart
→ mixed_analysis
```

---

## 3.3 Unsupported Histogram Interpretation

### Problem

The chart tool returned only:

```text
count
mean
median
min
max
```

but the reporter initially described the histogram as potentially left-skewed or approximately symmetric.

These claims were not supported by the tool evidence.

### Solution

The reporter evidence rules were strengthened.

When the chart result contains only:

```text
count
mean
median
min
max
```

the reporter may describe only those values and the observed range.

It may not infer:

```text
skewness
symmetry
concentration
long-tail behavior
peak interval
```

without additional returned evidence.

A reporter regression test was added.

---

## 3.4 Incomplete Scatter-Plot Reporting

### Problem

For a mixed RAG and chart task, the chart tool returned:

```text
point_count
x_min
x_max
y_min
y_max
pearson_correlation
```

but the initial final answer only mentioned the unit and chart path.

The workflow executed correctly, but the reporter failed to use all available evidence.

### Solution

The reporter prompt was updated so that a returned `relationship_summary` must be summarized in the final answer.

The final answer now includes:

* point count;
* x range;
* y range;
* Pearson correlation;
* association wording;
* correlation-versus-causation warning.

A regression test was added to preserve this behavior.

---

## 3.5 Model Client Lifecycle

### Problem

During an automated mixed-task evaluation, the workflow failed after the first tool with:

```text
Cannot send a request, as the client has been closed.
```

This showed that model-client lifecycle behavior needed to be explicit during repeated graph calls.

### Solution

Each model request now creates an independent OpenAI-compatible client within a managed context.

The client is closed after the synchronous response is returned.

The mixed RAG and SQL evaluation was rerun successfully, followed by the complete ten-case suite.

### Key Lesson

Manual CLI success is not sufficient to prove that repeated end-to-end execution is stable.

Automated evaluation can expose lifecycle and state-management problems that unit tests may miss.

---

## 3.6 CLI and Agent-State Representation

### Problem

LangGraph stored tool calls and tool results as dictionaries, while some earlier CLI and evidence code expected structured objects.

### Solution

The CLI now normalizes these records while preserving backward compatibility.

This keeps the workflow state simple while allowing the display and reporting layers to use validated models when needed.

---

## 4. Solutions and Engineering Decisions

The main Week 5 engineering decisions were:

### Preserve Model Flexibility but Validate Execution

The model can propose:

* SQL;
* chart arguments;
* RAG queries;
* final answers.

Deterministic Python remains responsible for:

* validation;
* destructive-operation rejection;
* field checking;
* retry limits;
* step limits;
* state transitions;
* report formatting.

---

### Use Targeted Rules Based on Observed Failures

Prompt rules were added only after concrete evidence failures, such as unsupported histogram interpretation or incomplete scatter reporting.

Core safety and argument validation remain implemented in Python rather than relying only on prompts.

---

### Evaluate Stable Behaviors Rather Than Exact Wording

The MVP evaluation does not require the final answer to match one fixed paragraph.

It checks stable dimensions:

```text
task type
workflow status
tool order
required facts
forbidden conclusions
```

This allows natural-language variation while detecting real regressions.

---

### Keep Reports Deterministic

The Markdown report is generated from existing state.

No additional model request is made.

This avoids:

* extra cost;
* new hallucinations;
* changes between terminal and report;
* failure when the model service is unavailable after analysis.

---

### Retain the Old Tool Loop

The earlier `tool_loop.py` remains in the repository.

It is useful for:

* showing the project’s development history;
* comparing a manual loop with LangGraph;
* explaining why the workflow was migrated;
* providing a simpler reference implementation.

The formal CLI now uses LangGraph.

---

## 5. Final Evaluation Results

The current project status is:

```text
Formal CLI: completed
Mixed workflows: completed
Unsupported-request handling: completed
MVP evaluation runner: completed
Markdown report generation: completed
Core MVP evaluation: 10/10 passed
```

The final MVP supports:

```text
CSV profiling
read-only SQL
bar charts
scatter plots
histograms
line charts
project-document RAG
mixed-tool LangGraph workflows
deterministic tool validation
evidence-aware final answers
Markdown reports
automated end-to-end evaluation
```

---

## 6. Interview Talking Points

## 6.1 Why Use LangGraph?

A manual tool loop is sufficient for early experiments, but becomes difficult to manage when the Agent needs:

* explicit planning;
* multiple tool calls;
* validation retries;
* maximum-step controls;
* result checking;
* conditional routing;
* observable state.

LangGraph makes these transitions explicit and testable.

---

## 6.2 Why Not Use LangGraph from the Beginning?

The earlier manual tool loop was useful for understanding:

* function calling;
* tool schemas;
* model responses;
* structured results;
* basic execution control.

Starting with the simpler loop made it easier to understand what LangGraph later abstracted.

The migration also demonstrates an engineering progression from a prototype to a maintainable workflow.

---

## 6.3 What Is the Role of the Language Model?

The model handles semantic tasks:

* understanding the user request;
* proposing tool arguments;
* generating SQL;
* selecting chart parameters;
* producing a readable final answer.

It is not trusted to enforce safety by itself.

---

## 6.4 What Is the Role of Deterministic Python?

Python controls:

* read-only safety;
* schema validation;
* chart-column validation;
* retries;
* execution limits;
* tool execution;
* graph transitions;
* structured reports;
* automated evaluation.

This separation is one of the main design strengths of the project.

---

## 6.5 What Was the Most Important Bug Found?

One important issue was that a tool could return:

```text
success = True
```

while the overall task still failed semantically.

The RAG tool successfully returned chunks, but the relevant field definition was missing because the query had drifted.

This led to the distinction between:

```text
technical tool success
```

and:

```text
semantic task completion
```

---

## 6.6 How Is Hallucination Controlled?

CausaLens controls hallucination by:

* separating evidence sources;
* validating fields and SQL;
* requiring document evidence for definitions and units;
* preventing undocumented thresholds;
* restricting chart interpretation to returned summaries;
* requiring correlation-versus-causation warnings;
* preserving uncertainty when evidence is missing.

---

## 6.7 How Is the Project Evaluated?

The project uses:

* deterministic unit tests;
* LangGraph node and transition tests;
* CLI tests;
* report-generation tests;
* a ten-case end-to-end MVP evaluation suite.

The evaluation verifies tool order and evidence boundaries, not just whether the Agent returns some text.

---

## 6.8 What Does the 100% Evaluation Result Mean?

It means:

> The current CausaLens implementation passed all ten defined MVP cases on the fixed demonstration dataset.

It does not mean:

* every possible CSV is supported;
* every statistical method is implemented;
* every wording will be routed correctly;
* the system performs general causal inference.

---

## 7. Remaining Limitations

The current limitations include:

* keyword-based intent routing;
* predefined mixed-tool ordering;
* limited semantic completion checking;
* normally one execution per tool type;
* original CSV used as chart input;
* no direct charting of SQL result tables;
* no regression or hypothesis-testing tools;
* no complete causal-inference workflow;
* a small project-specific RAG knowledge base;
* dependence on an external language-model service for real runs;
* limited MVP evaluation coverage;
* no graphical user interface.

These limitations are intentional boundaries of the current MVP rather than hidden failures.

---

## 8. Possible Future Improvements

Reasonable future directions include:

1. model-assisted or hybrid intent routing;
2. dynamic tool dependency planning;
3. stronger semantic result checking;
4. SQL-result-to-chart pipelines;
5. regression and hypothesis-testing tools;
6. explicit causal-analysis configuration;
7. dataset-aware schema adaptation;
8. additional evaluation datasets;
9. a lightweight Web interface;
10. better caching for embeddings and model calls.

These improvements should be added only when they support a clear user need.

---

## 9. Week 5 Completion Summary

Week 5 transformed CausaLens from a collection of working components into a complete MVP.

The project now has:

* one formal LangGraph execution path;
* a usable CLI;
* stable mixed-tool workflows;
* validated safety boundaries;
* repeatable end-to-end evaluation;
* structured analysis reports;
* organized technical documentation;
* clear product and technical limitations.

The MVP is ready for:

* repository presentation;
* technical interviews;
* live demonstrations;
* further tool expansion;
* future UI integration.

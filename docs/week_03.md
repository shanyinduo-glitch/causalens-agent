# Week 3 Summary：RAG 数据字典检索与可追溯回答

## 1. 本周目标

本周目标是为 CausaLens 增加 RAG（Retrieval-Augmented Generation，检索增强生成）能力。

在 Week 1 和 Week 2 中，CausaLens 已经能够基于 CSV 数据调用工具完成：

- 数据集概览；
- SQL 查询；
- 分组统计；
- 图表生成；
- 简单的数据分析回答。

但这些能力主要依赖 CSV 本身，无法可靠回答以下问题：

- 字段是什么意思；
- 字段单位是什么；
- 字段合法取值是什么；
- 数据集是否是真实数据；
- 是否能直接得出因果结论；
- 缺失值、标准化、异常值处理等规则是否存在。

因此，Week 3 的核心目标是：

> 让 Agent 不仅能分析 CSV 数据本身，还能查询项目文档、数据字典和分析规则，并基于文档给出可追溯回答。

最终希望实现的流程是：

```text
用户问题
↓
模型判断是否需要查询文档
↓
调用 retrieve_docs 工具
↓
从本地知识库中检索相关文档片段
↓
模型基于检索结果生成回答
↓
回答中尽量标注来源文件
```

---

## 2. 新增知识库文档

本周新增了 `knowledge/` 目录，用于存放项目知识库原始文档。

当前包含：

```text
knowledge/
├── data_dictionary.md
├── project_background.md
└── analysis_rules.md
```

三份文档的作用如下。

### 2.1 data_dictionary.md

`data_dictionary.md` 是数据字典，记录 `student_performance.csv` 的字段定义。

主要包含：

- 字段名；
- 中文名称；
- 字段含义；
- 数据类型；
- 单位；
- 文档定义范围；
- 合法取值；
- 缺失值规则；
- 标准化状态；
- 解释边界。

例如：

- `study_hours` 表示每周学习时长，单位为 `hours per week` / 小时每周；
- `score` 表示最终考试成绩，文档定义量表范围为 `0` 到 `100`；
- `passed` 是二元变量，`1` 表示通过考试，`0` 表示未通过考试；
- `exam_date` 是日期字段，格式为 `YYYY-MM-DD`。

### 2.2 project_background.md

`project_background.md` 记录项目背景和数据集限制。

主要说明：

- 当前数据集是模拟生成的演示数据；
- 数据集用于 CausaLens 项目的分析功能展示；
- 数据不代表真实教育数据；
- 数据集中不包含真实学校、真实学生或真实采集流程；
- 不能基于该数据直接做真实教育决策。

### 2.3 analysis_rules.md

`analysis_rules.md` 记录分析规则和方法边界。

主要说明：

- 相关性不能证明因果关系；
- 描述性统计只能描述当前样本；
- 分组差异不能直接解释为因果效应；
- 回归分析如果没有识别策略，也不能自动解释为因果；
- 因果分析需要明确 treatment、outcome、confounders、时间顺序和识别假设。

---

## 3. 新增 RAG 代码结构

本周新增了 RAG 相关模块：

```text
src/causalens/rag/
├── __init__.py
├── document_loader.py
├── chunker.py
├── vector_store.py
├── retriever.py
└── rag_tool.py
```

各文件作用如下。

### 3.1 document_loader.py

`document_loader.py` 负责读取 `knowledge/` 目录下的 Markdown 文档。

核心流程：

```text
knowledge/*.md
↓
读取 Markdown 文件
↓
转换成 Python document list
```

每个文档会被读取成类似结构：

```text
source: data_dictionary.md
path: knowledge/data_dictionary.md
content: 整篇 Markdown 文档内容
```

### 3.2 chunker.py

`chunker.py` 负责将长文档切分成较小的文本块，即 chunk。

chunk 可以理解为：

> 从知识文档中切出来的一张“知识卡片”。

本周最初使用固定字符切分，后来改进为 Markdown 标题切分。

原因是数据字典天然具有字段小节结构，例如：

```text
### score
...
### passed
...
### study_hours
...
```

如果按固定字符数切分，可能把相邻字段混在一起，例如把 `score` 和 `passed` 的内容切进同一个 chunk。

改为 Markdown 标题切分后，每个字段小节可以更稳定地成为独立 chunk。

每个 chunk 当前大致包含：

```text
chunk_id: data_dictionary_0016
source: data_dictionary.md
path: knowledge/data_dictionary.md
section: score
chunk_index: 16
content: ### score ...
```

其中：

- `content` 是实际检索内容；
- `source` 表示来源文件；
- `section` 表示来源小节；
- `chunk_id` 用于追踪具体片段。

### 3.3 vector_store.py

`vector_store.py` 负责将 chunks 转成 embedding，并写入本地 Chroma 向量库。

当前知识库构建流程为：

```text
knowledge/*.md
↓
load_markdown_documents()
↓
create_chunks_from_documents()
↓
embed_texts()
↓
build_vector_store()
↓
storage/chroma_db/
```

`storage/chroma_db/` 是本地向量数据库，不应提交到 GitHub。

### 3.4 retriever.py

`retriever.py` 负责从 Chroma 向量库中检索与用户问题最相关的 chunks。

例如用户问题是：

```text
score 的取值范围是什么？
```

检索结果可能包含：

```text
source: data_dictionary.md
section: score
chunk_id: data_dictionary_0016
content: Documented scale: 0 to 100
distance: 0.7394
```

其中 `distance` 表示向量距离，通常越小表示语义越相似。

### 3.5 rag_tool.py

`rag_tool.py` 负责将检索能力包装成 Agent 工具：

```text
retrieve_docs(query, top_k)
```

`retrieve_docs` 是本周新增的核心工具。

它不是用来分析 CSV 数值，而是用来查询项目文档，例如：

- 字段含义；
- 字段单位；
- 字段合法取值；
- 数据背景；
- 分析规则；
- 缺失值处理规则；
- 因果分析边界。

---

## 4. retrieve_docs 工具说明

`retrieve_docs` 可以理解为 CausaLens 的“查资料工具”。

例如用户问：

```text
passed 字段是什么意思？
```

Agent 应优先调用：

```text
retrieve_docs(query="passed 字段是什么意思？")
```

工具会从知识库中检索相关文档片段，例如：

```text
data_dictionary.md，passed
```

然后模型基于检索结果回答：

```text
passed 字段表示学生是否通过考试：
1 表示通过；
0 表示未通过。
来源：data_dictionary.md
```

这使 Agent 从：

```text
根据字段名猜测
```

变成：

```text
基于项目文档回答
```

这是 Week 3 最重要的能力升级。

---

## 5. 工具路由规则

本周进一步明确了工具选择边界。

### 5.1 文档问题

以下问题应优先使用 `retrieve_docs`：

| 问题类型 | 示例 |
| --- | --- |
| 字段含义 | `passed 字段是什么意思？` |
| 字段单位 | `study_hours 的单位是什么？` |
| 合法取值 | `major 字段有哪些合法取值？` |
| 文档定义范围 | `score 的取值范围是什么？` |
| 日期格式 | `exam_date 是什么格式？` |
| 数据背景 | `这份数据是真实教育数据吗？` |
| 分析边界 | `相关性可以证明因果关系吗？` |
| 缺失值处理 | `缺失值是如何处理的？` |
| 标准化状态 | `score 是否经过标准化处理？` |

### 5.2 当前 CSV 统计问题

以下问题应使用 `profile_dataset` 或 `run_readonly_sql`：

| 问题类型 | 示例 |
| --- | --- |
| 当前实际范围 | `当前 CSV 中 score 实际出现的最小值和最大值是多少？` |
| 分组统计 | `不同专业的平均 score 分别是多少？` |
| 字段列表 | `这个数据集有哪些字段？` |
| 当前缺失值数量 | `当前 CSV 中每列缺失值是多少？` |

### 5.3 图表问题

以下问题应使用 `create_chart`：

| 问题类型 | 示例 |
| --- | --- |
| 散点图 | `请画出 study_hours 与 score 的散点图` |
| 柱状图 | `画出不同专业的平均 score` |
| 直方图 | `画出 score 的分布` |
| 折线图 | `按 exam_date 展示平均 score 变化` |

---

## 6. 文档定义范围与当前样本实际范围

Week 3 中一个重要问题是：

```text
score 的取值范围是什么？
```

这个问题可能有两种含义：

1. 文档定义范围；
2. 当前 CSV 中实际观测范围。

本项目中需要明确区分：

```text
文档定义范围
→ 来自 retrieve_docs
→ 例如 score 的文档定义范围是 0 到 100

当前样本实际范围
→ 来自 profile_dataset 或 run_readonly_sql
→ 例如当前 CSV 中实际出现的 score 范围是 52 到 98
```

因此，回答字段范围类问题时，不能混淆这两类证据。

---

## 7. Chunking 优化

本周早期使用固定字符切分，后来发现对于数据字典不够稳定。

例如原始文档中如果有：

```text
### study_hours

### 每周学习时长
```

简单的 Markdown 标题切分会把它切成两个 section：

```text
section: study_hours
section: 每周学习时长
```

这可能导致 `study_hours` section 里只有标题，而单位信息被切到另一个 section。

为提高检索稳定性，后续将数据字典格式调整为：

```text
### study_hours

**中文名称：** 每周学习时长
```

即每个字段只保留一个 `### 字段名`，中文名放入正文。

这种结构更适合当前 chunker，使字段定义、单位、缺失值规则和解释说明保留在同一个 chunk 中。

---

## 8. 执行依据与可追溯回答

本周还增强了执行依据展示。

对于 `retrieve_docs`，执行依据中会展示：

- 检索问题；
- 返回片段数；
- 来源文件；
- section；
- chunk_id；
- distance。

示例：

```text
retrieve_docs 成功
检索问题：score 的取值范围是什么
返回片段数：3
来源片段：
- data_dictionary.md，score (data_dictionary_0016, distance=0.7394)
```

这使回答更容易追踪到具体文档来源，而不仅仅是笼统地说“来自数据字典”。

---

## 9. 最大工具轮数兜底

在评测过程中发现，模型有时已经拿到了足够工具结果，但仍继续调用工具，最终达到最大工具轮数限制。

原始行为是：

```text
已达到最多 4 轮工具调用限制，模型仍未生成最终回答。
```

这会导致用户明明已经有足够证据，却得不到最终答案。

因此，本周增加了最终回答兜底逻辑：

```text
达到最大工具轮数
↓
不再继续调用工具
↓
再调用一次模型
↓
不传 tools
↓
强制模型只基于已有工具结果生成最终回答
```

该改动属于通用 Agent 稳定性改进，不是针对某个具体数据集的特调。

---

## 10. Week 3 自动评测

本周新增评测文件：

```text
evals/week_03_rag_cases.jsonl
```

新增评测脚本：

```text
scripts/run_week_03_rag_eval.py
```

评测输出保存到：

```text
outputs/evals/week_03_rag_results.jsonl
```

评测覆盖 15 个 case：

| Case | 问题类型 | 示例 |
| --- | --- | --- |
| w3_01 | 字段含义 | `passed 字段是什么意思？` |
| w3_02 | 字段单位 | `study_hours 的单位是什么？` |
| w3_03 | 文档定义范围 | `score 的取值范围是什么？` |
| w3_04 | 日期格式 | `exam_date 是什么格式？` |
| w3_05 | 合法取值 | `major 字段有哪些合法取值？` |
| w3_06 | 当前样本实际范围 | `当前 CSV 中 score 实际出现的最小值和最大值是多少？` |
| w3_07 | SQL 分组统计 | `不同专业的平均 score 分别是多少？` |
| w3_08 | 数据背景 | `这份数据是真实教育数据吗？` |
| w3_09 | 因果边界 | `这份数据是否适合直接得出因果结论？` |
| w3_10 | 相关性与因果 | `相关性可以直接证明因果关系吗？` |
| w3_11 | 标准化状态 | `score 是否经过标准化处理？` |
| w3_12 | 缺失信息 | `学生的具体学校名称是什么？` |
| w3_13 | 缺失值处理 | `缺失值是如何处理的？` |
| w3_14 | 图表路由 | `请画出 study_hours 与 score 的散点图。` |
| w3_15 | 综合文档问题 | `这份数据适合做什么类型的分析？` |

---

## 11. Week 3 评测结果

本轮 Week 3 RAG 自动评测结果为：

```text
Passed: 13 / 15
```

通过的能力包括：

- 字段含义检索；
- 字段单位检索；
- 字段文档定义范围检索；
- 日期格式检索；
- 当前 CSV 实际范围查询；
- SQL 分组统计；
- 数据背景检索；
- 因果边界检索；
- 相关性与因果关系说明；
- 标准化状态说明；
- 图表工具路由；
- 综合分析类型说明。

剩余 2 个 case 未自动通过，失败原因均为：

```text
expected_answer_contains_ok
```

即模型回答与评测文件中预设关键词没有完全匹配。

人工检查后，这两个 case 的工具调用、来源检索和回答方向基本正确，主要问题来自自然语言表达与关键词级自动评测之间的不一致。

例如：

- 模型可能回答“没有记录学校名称字段”，而评测期望中写的是“无法确认”；
- 模型可能回答“没有提供缺失值填补流程证据”，而评测期望中写的是另一个固定表达。

因此，本轮评测结论为：

> Week 3 RAG 主链路基本通过。后续应优化评测脚本的语义匹配能力，而不是继续修改 Agent 主逻辑来迎合固定关键词。

---

## 12. 重要问题与解决过程

### 12.1 Windows 编码问题

初次运行评测时，大量中文输出出现乱码，并导致部分 CLI 调用返回非零状态码。

解决方式是在评测脚本的子进程环境中加入：

```text
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
```

该改动使 CLI 子进程输出统一为 UTF-8，避免 Windows 默认编码导致的乱码和 Rich 输出异常。

### 12.2 检索命中但模型不收口

部分 case 中，模型已经调用工具取得足够信息，但仍不断重复调用工具，最终达到 `MAX_TOOL_ROUNDS`。

解决方式是在最大工具轮数后增加最终回答兜底：

```text
不再传 tools
只基于已有工具结果生成最终回答
```

该改动提高了 Agent 的稳定性。

### 12.3 文档写法影响 chunking

原始数据字典中曾使用连续同级标题，例如：

```text
### study_hours

### 每周学习时长
```

这会让 chunker 将字段名和中文说明切成两个不同 section，导致检索结果不完整。

后续改为：

```text
### study_hours

**中文名称：** 每周学习时长
```

使字段信息保留在同一个 chunk 中。

### 12.4 评测关键词过于严格

当前评测脚本主要使用字符串包含检查。

这会导致语义正确但措辞不同的回答被判失败。

例如：

```text
未提供
没有说明
没有记录
无法确认
```

这些表达在语义上可能相近，但简单字符串匹配无法自动判断。

后续可以改进为：

- 支持 `expected_answer_contains_any`；
- 支持同义表达分组；
- 使用更稳定的结构化字段检查；
- 或引入 LLM-as-judge 做语义评估。

---

## 13. 当前已知局限

当前 Week 3 版本仍有以下局限：

1. RAG 知识库仍以 `student_performance.csv` 为 demo，没有完成多数据集隔离；
2. `retrieve_docs` 有时会重复检索同一类问题；
3. 当前路由主要依赖 prompt，尚未实现确定性意图分类器；
4. 评测脚本主要依赖关键词匹配，不能完全理解语义等价表达；
5. 当前 Chroma 向量库需要手动运行 `build_knowledge_base.py` 重建；
6. 修改 `knowledge/` 文档或 chunking 策略后，必须重新 build；
7. 对缺失信息类问题，模型有时会结合 `profile_dataset` 和 `retrieve_docs` 同时回答，虽然结果更完整，但会增加工具调用次数；
8. 目前还没有实现跨数据集知识库结构，例如按 `dataset_id` 隔离不同数据集的数据字典。

---

## 14. 本周收获

本周最重要的收获包括：

1. RAG 的核心不是让模型“记住更多”，而是让模型能够基于当前项目文档回答；
2. CSV 数据本身和数据字典是两类不同证据；
3. 文档定义范围和当前样本实际范围必须分开；
4. Chunking 策略会显著影响检索质量；
5. Metadata，例如 `source`、`section`、`chunk_id`，对可追溯回答非常重要；
6. Prompt 不能无限打补丁，过度复杂的 prompt 可能导致模型重复调用工具；
7. 最大工具轮数后需要有最终回答兜底，否则用户可能拿不到答案；
8. 自动评测不仅要看通过率，也要分析失败原因；
9. 关键词级评测容易误判自然语言同义表达；
10. 工程上要区分：模型问题、工具问题、检索问题、展示问题、评测问题。

---

## 15. Demo 问题

Week 3 Demo 可以展示以下问题：

```text
passed 字段是什么意思？
```

展示字段含义检索。

```text
study_hours 的单位是什么？
```

展示字段单位检索。

```text
score 的取值范围是什么？
```

展示文档定义范围检索。

```text
当前 CSV 中 score 实际出现的最小值和最大值是多少？
```

展示当前样本实际范围查询。

```text
不同专业的平均 score 分别是多少？
```

展示 SQL 分组统计。

```text
请画出 study_hours 与 score 的散点图。
```

展示图表工具调用。

```text
相关性可以直接证明因果关系吗？
```

展示分析规则和因果边界检索。

```text
学生的具体学校名称是什么？
```

展示缺失信息处理。

---

## 16. 下一周方向

下一周建议进入 Agent 工作流升级，重点方向包括：

1. 引入更明确的工作流控制，例如 LangGraph；
2. 将当前工具循环拆分为显式节点；
3. 增加确定性路由预检查；
4. 减少模型重复调用同一个工具；
5. 增加工具结果后的回答验证；
6. 改进缺失信息类问题的判断；
7. 支持多数据集知识库结构；
8. 优化评测脚本，使其支持同义表达和结构化检查；
9. 将当前 RAG 能力从 `student_performance.csv` demo 扩展到更多 CSV 数据集。

---

## 17. 总结

Week 3 完成了 CausaLens 从“只会分析 CSV”到“能够结合项目文档回答”的关键升级。

当前系统已经能够：

- 查询字段含义；
- 查询字段单位；
- 查询字段合法取值；
- 查询文档定义范围；
- 区分文档定义范围和当前样本实际范围；
- 说明数据集背景；
- 说明分析规则和因果边界；
- 对缺失信息保持谨慎；
- 在回答中展示来源文件和文档 section；
- 通过自动评测验证主要 RAG 能力。

虽然当前版本仍存在评测关键词匹配、重复检索、多数据集隔离等问题，但 Week 3 的 RAG 主链路已经基本跑通。

本周最终结论：

> CausaLens 已经具备基础 RAG 数据字典检索能力，能够基于本地项目文档进行更可靠、更可追溯的数据分析问答。
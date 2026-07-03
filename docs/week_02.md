# Week 2 总结：SQL 分析、图表工具与评测

## 本周目标

本周的目标是将 CausaLens 从一个基础的 CSV 数据概览 Agent，升级为一个具备基础数据分析、图表生成、多轮工具调用、日志记录与评测能力的 CSV 数据分析 Agent。

经过本周开发，CausaLens 已经能够：

* 读取并概览 CSV 数据集；
* 使用只读 SQL 回答筛选、分组、排序、Top-N 等问题；
* 自动生成基础图表；
* 在一次对话中连续调用多个工具；
* 记录每一次真实工具调用；
* 在 CLI 中展示基于真实工具结果生成的执行依据；
* 使用固定评测题集验证工具路由与边界行为。

---

## 新增能力

## 1. 数据集概览

工具名称：

```text
profile_dataset
```

支持能力：

* 获取数据集行数、列数；
* 查看字段名；
* 查看字段数据类型；
* 统计每列缺失值数量；
* 获取数值列的描述性统计；
* 查看均值、中位数、最大值、最小值、标准差；
* 对 `passed` 等 0/1 数值字段进行整体平均值或通过率解释。

示例问题：

```text
这个数据集有哪些字段？
这个数据集有多少行、多少列？
哪些字段存在缺失值？
score 的平均值、中位数、最大值和最小值分别是多少？
passed 字段的整体平均值是多少？
```

---

## 2. 只读 SQL 数据分析

工具名称：

```text
run_readonly_sql
```

CSV 文件在 DuckDB 中统一注册为：

```sql
dataset
```

支持能力：

* 条件筛选；
* 分组统计；
* 聚合计算；
* 排序；
* Top-N 查询；
* 查询具体记录；
* 类别计数；
* 分组平均值；
* 分组通过率；
* 最大值、最小值、总数等统计。

示例问题：

```text
不同专业的平均 score 分别是多少？
每个 city 分别有多少名学生？
score 最高的 3 名学生是谁？
attendance_rate 最高的 5 名学生是谁？
不同 major 的通过率分别是多少？
找出 score 大于等于 90 的学生。
```

### SQL 安全边界

当前 SQL 工具采用基础只读安全限制：

* 一次只允许执行一条 SQL；
* SQL 必须以 `SELECT`、`WITH` 或 `EXPLAIN` 开头；
* 禁止 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`CREATE` 等修改操作；
* 禁止多条 SQL 语句；
* 查询结果会限制最大返回行数，避免结果过大。

这套限制适合当前本地学习项目，但还不是面向不可信生产环境的完整 SQL 沙箱。

---

## 3. 图表生成

工具名称：

```text
create_chart
```

当前支持四类基础图表：

| 图表类型        | 适用场景           |
| ----------- | -------------- |
| `bar`       | 类别对比、分组均值、类别计数 |
| `scatter`   | 两个数值变量之间的关系    |
| `histogram` | 单个数值变量的分布      |
| `line`      | 按日期聚合后的趋势变化    |

示例问题：

```text
请画出不同专业平均 score 的柱状图。
请画出不同 city 的学生数量柱状图。
请画出 study_hours 和 score 的散点图。
请画出 score 的直方图。
请画出按 exam_date 汇总的平均 score 折线图。
```

生成的 PNG 图表保存在：

```text
outputs/charts/
```

运行生成的 PNG 图片已加入 `.gitignore`，不会提交到 GitHub。

---

## 4. Agent 工具路由

Agent 会根据用户问题自动选择工具。

基本路由规则：

```text
字段、缺失值、整体描述统计
→ profile_dataset

分组、筛选、排序、Top-N、具体记录查询
→ run_readonly_sql

明确要求画图、趋势、分布、类别对比、变量关系
→ create_chart
```

例如：

```text
用户问题：
请画出不同专业平均 score 的柱状图，并说明哪个专业最高。

可能的 Agent 工作流：
1. profile_dataset
2. create_chart
3. 根据图表工具返回的真实汇总数据生成回答
```

---

## 5. 多轮工具调用

Agent 已支持在一次用户请求中进行多轮工具调用。

典型流程：

```text
用户提出问题
↓
模型判断需要先确认字段
↓
profile_dataset
↓
模型获得字段信息
↓
run_readonly_sql 或 create_chart
↓
工具执行并返回真实结果
↓
模型根据工具结果生成最终回答
```

系统中设置了最大工具调用轮数，避免模型陷入无限循环。

---

## 6. 工具调用日志

每一次真实工具调用都会记录到：

```text
logs/tool_calls.jsonl
```

日志采用 JSONL 格式，即“一行一个 JSON 对象”。

每条日志包含：

* 时间戳；
* 用户问题；
* 数据集路径；
* 工具名称；
* 工具参数；
* 执行是否成功；
* 执行耗时；
* 工具结果摘要；
* 错误信息。

例如：

```json
{
  "timestamp": "2026-07-03T22:08:02+08:00",
  "question": "不同专业的平均 score 分别是多少？",
  "dataset_path": "data/student_performance.csv",
  "tool_name": "run_readonly_sql",
  "arguments": {
    "dataset_path": "data/student_performance.csv",
    "query": "SELECT major, AVG(score) AS avg_score FROM dataset GROUP BY major"
  },
  "success": true,
  "duration_ms": 18.42,
  "result_summary": {
    "row_count": 3,
    "returned_row_count": 3,
    "truncated": false
  },
  "error": null
}
```

日志文件默认不提交到 GitHub。

查看日志时，PowerShell 需要显式指定 UTF-8 编码：

```powershell
Get-Content -Encoding UTF8 logs/tool_calls.jsonl
```

---

## 7. CLI 执行依据展示

CLI 在输出 Agent 最终回答后，还会展示一张由 Python 根据真实 `ToolResult` 自动生成的“执行依据”表格。

它不是模型自己编写的解释，而是基于真实工具结果确定性生成。

例如，散点图任务会显示：

```text
工具：create_chart
状态：成功
关键事实：
生成 scatter 图
参与绘图数据：40 条
关系摘要：point_count=40，x_min=3，x_max=18，
y_min=52，y_max=98，pearson_correlation=0.9874
图表：outputs/charts/...
```

执行依据可以展示：

* 数据集规模与字段信息；
* SQL 返回行数与结果字段；
* 图表类型；
* 图表路径；
* 实际参与绘图的数据量；
* 直方图统计摘要；
* 散点图关系摘要。

这使最终回答更容易验证，也降低了模型自然语言解释带来的不确定性。

---

## 8. 图表解释的证据边界

为了避免模型根据有限预览夸大结论，系统提示词中加入了图表解释边界。

核心原则如下：

* `data_preview` 只是部分数据预览，不代表完整数据集；
* `numeric_summary` 只支持描述实际返回的统计量，例如 `count`、`mean`、`median`、`min`、`max`；
* `relationship_summary` 只支持描述实际返回的样本点数、横纵轴范围和 Pearson 相关系数；
* `chart_path` 只能证明 PNG 图表已经生成，不能证明模型已经“看懂”图像；
* 工具未返回的统计量、分布特征或视觉结论，不应被当作确定事实陈述。

例如，当前工具结果不足以直接支持以下结论：

```text
数据呈正态分布
数据明显偏态
大多数样本集中在某个区间
存在双峰分布
几乎没有异常点
散点紧密贴近某条回归线
```

### 散点图的证据边界

散点图工具会返回：

* `point_count`
* `x_min`
* `x_max`
* `y_min`
* `y_max`
* `pearson_correlation`

因此可以合理描述：

```text
在当前样本中，study_hours 与 score 存在强正线性相关。
```

但不能仅凭当前返回值直接描述：

```text
某个学习时长区间一定对应某个成绩范围
数据点几乎完全贴近一条直线
没有明显异常点
```

此外，相关关系不代表因果关系。

---

## 9. 不支持的图表请求

当前版本仅支持：

```text
bar
scatter
histogram
line
```

当用户要求当前不支持的图表类型时，例如：

```text
pie
boxplot
heatmap
violin plot
radar chart
```

Agent 应当：

1. 诚实说明当前版本暂不支持该图表类型；
2. 列出当前支持的图表类型；
3. 不调用数据工具；
4. 不伪造图表；
5. 不未经用户同意擅自改成其他图表。

例如，用户要求“画饼图”时，Agent 不应直接改画柱状图。

---

## 自动化测试

本周新增和维护了以下测试覆盖：

* 数据集概览工具；
* SQL 查询工具；
* SQL 安全边界；
* 柱状图、散点图、直方图、折线图；
* 图表参数与字段类型校验；
* 工具注册表；
* 多轮工具调用；
* 工具调用日志；
* CLI 执行依据；
* 评测题读取与评分逻辑；
* 系统提示词中的安全规则与证据边界。

完整测试套件已通过。

---

## Week 2 评测

评测题集位置：

```text
evals/week_02_cases.jsonl
```

评测结果位置：

```text
outputs/evals/week_02_results.jsonl
```

评测题集共包含 20 条案例：

| 类别      | 数量 |
| ------- | -: |
| 数据概览    |  5 |
| SQL 分析  |  6 |
| 图表请求    |  5 |
| 边界与拒绝场景 |  4 |
| 总计      | 20 |

最终评测结果：

| 类别       | 通过数 | 总数 |  通过率 |
| -------- | --: | -: | ---: |
| profile  |   5 |  5 | 100% |
| sql      |   6 |  6 | 100% |
| chart    |   5 |  5 | 100% |
| boundary |   4 |  4 | 100% |
| total    |  20 | 20 | 100% |

评测主要检查：

* 必需工具是否被调用；
* 禁止工具是否未被调用；
* 必需工具是否执行成功；
* 图表类型是否符合预期；
* 应拒绝的边界案例是否没有调用工具。

需要注意：这套评测主要验证工具路由、参数选择、执行结果和边界行为，不会完全自动评估自然语言回答的表达质量与解释严谨性。

---

## Demo 命令

### 1. 数据集概览

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "这个数据集有哪些字段？"
```

### 2. SQL 数据分析

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "不同专业的平均 score 分别是多少？"
```

### 3. 生成柱状图

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "请画出不同专业平均 score 的柱状图，并说明哪个专业最高。"
```

### 4. 生成散点图

```powershell
python -m causalens.cli ask `
  --dataset-path data/student_performance.csv `
  --question "请画出 study_hours 和 score 的散点图，并描述两者可能存在的关系。"
```

### 5. 运行 Week 2 评测

```powershell
python scripts/run_week_02_eval.py
```

---

## 已知局限

1. 当前评测主要衡量工具路由、工具执行、参数选择和边界行为，不能完全自动判断模型自然语言回答是否足够严谨。

2. Prompt 可以减少模型超出证据范围的解释，但无法像 Python 规则一样做到绝对确定。

3. 当前图表系统不支持饼图、箱线图、热力图、小提琴图、雷达图、分面图和复杂分组图表。

4. 当前 SQL 安全限制适用于本地学习项目，不适合直接视为生产级不可信 SQL 沙箱。

5. 当前 Agent 可以进行描述性分析和相关性分析，但不能根据观察性 CSV 数据直接得出因果结论。

6. 图表工具当前只生成本地 PNG 文件，模型不会自动对生成的 PNG 进行真正的视觉识别。

---

## 下一步计划

Week 3 可以重点探索以下方向：

* 在模型调用前增加更确定性的请求预检查；
* 设计更结构化的最终回答格式；
* 建立自动化的“回答结论—工具证据”一致性检查；
* 支持更多图表类型和分组图表；
* 增加相关性、回归等分析工具；
* 增加因果分析能力评估模块；
* 使用 Streamlit 或其他方式制作简单 Web Demo。

# Student Performance Dataset Data Dictionary

# 学生成绩数据集数据字典

## Dataset Overview

## 数据集概览

This dataset records student learning behavior and examination performance.

本数据集记录学生的学习行为与考试表现，用于 CausaLens 项目的数据分析功能演示。

* Unit of observation: one student
  观察单位：每一行代表一名学生。

* Dataset type: simulated demonstration dataset
  数据类型：模拟生成的演示数据集。

* Documentation-level time scope: March 2026
  文档定义的时间范围：2026 年 3 月。

* Exact observed date range: use the `exam_date` column in the current CSV to verify the minimum and maximum dates.
  精确的实际日期范围：应通过当前 CSV 的 `exam_date` 列查询最早和最晚日期。

* Expected record count for the current demo dataset: approximately 40 records
  当前演示数据集的预期记录数：约 40 条。若 CSV 被替换或更新，应以 `profile_dataset` 的实际结果为准。

* Main purpose: exploratory data analysis, statistical summaries, visualization, and Agent tool-calling demonstrations
  主要用途：探索性数据分析、描述性统计、可视化，以及 Agent 工具调用演示。

## Column Definitions

## 字段说明

### student_id

### 学生唯一标识

Unique identifier for each student.

每名学生的唯一标识符。

* Type: integer
  类型：整数。

* Example: `1`, `2`, `3`
  示例：`1`、`2`、`3`。

* Missing values: not allowed by the documented schema
  缺失值：按照当前字段定义不允许缺失。

* Interpretation note: `student_id` is an identifier, not a meaningful numeric measurement. It should not be treated as a continuous analytical feature.
  解释说明：`student_id` 是标识符，不是有业务含义的连续数值变量；不应将其作为普通数值特征分析。

### major

### 专业

Student major.

学生所属专业。

* Type: categorical
  类型：分类变量。

* Documented possible values: `math`, `cs`, `stats`
  文档定义的可能取值：`math`、`cs`、`stats`。

* Missing values: not allowed by the documented schema
  缺失值：按照当前字段定义不允许缺失。

* Interpretation note: use the actual CSV to confirm which categories are present in the current dataset.
  解释说明：当前 CSV 中实际出现的类别，应通过数据概览或 SQL 查询确认。

### city

### 学习城市

City where the student studies.

学生学习所在的城市。

* Type: categorical
  类型：分类变量。

* Documented possible values: `wuhan`, `beijing`, `shanghai`
  文档定义的可能取值：`wuhan`、`beijing`、`shanghai`。

* Missing values: allowed by the documented schema
  缺失值：按照当前字段定义允许缺失。

* Important distinction: “allowed” means the schema permits missing values. It does not mean the current CSV necessarily contains missing values.
  重要区分：“允许缺失”仅表示字段规则允许空值，不代表当前 CSV 一定实际存在缺失。

### study_hours

### 每周学习时长

Average weekly study hours.

学生平均每周学习时长。

* Type: numeric
  类型：数值型。

* Unit: hours per week
  单位：小时/周。

* Missing values: allowed by the documented schema
  缺失值：按照当前字段定义允许缺失。

* Interpretation note: this field describes study time and can be used for descriptive analysis, association analysis, regression, and visualization. It does not by itself establish a causal effect on score.
  解释说明：该字段可用于描述性分析、关联分析、回归和可视化；但仅凭该字段不能证明其对成绩存在因果影响。

### attendance_rate

### 出勤率

Class attendance rate.

学生课程出勤率。

* Type: numeric
  类型：数值型。

* Expected range: `0` to `1`
  预期范围：`0` 到 `1`。

* Example: `0.85` means 85% attendance
  示例：`0.85` 表示出勤率为 85%。

* Missing values: the current documentation does not define a separate imputation rule for missing values.
  缺失值：当前文档没有定义缺失值填补规则。

### score

### 最终考试成绩

Final examination score.

学生最终考试成绩。

* Type: numeric
  类型：数值型。

* Documented scale: `0` to `100`
  文档定义的量表范围：`0` 到 `100`。

* Missing values: not allowed by the documented schema
  缺失值：按照当前字段定义不允许缺失。

* Standardization status: the current documentation does not state that this variable has been standardized, normalized, or transformed.
  标准化状态：当前文档没有说明该变量经过标准化、归一化或其他转换。

### passed

### 是否通过考试

Whether the student passed the examination.

学生是否通过考试。

* Type: binary
  类型：二元变量。

* `1` means passed
  `1` 表示通过考试。

* `0` means failed
  `0` 表示未通过考试。

* Important note: the current documentation defines the binary label but does not specify a score threshold for passing. Do not infer a passing threshold unless another document explicitly provides one.
  重要说明：当前文档定义了二元标签含义，但没有规定“多少分算通过”的分数线。除非其他文档明确提供，不能自行推断通过阈值。

### exam_date

### 考试日期

Date of the examination.

考试发生的日期。

* Type: date
  类型：日期型。

* Format: `YYYY-MM-DD`
  格式：`YYYY-MM-DD`。

* Use: suitable for date-based filtering and time-series aggregation after validating date parsing.
  用途：在日期解析正确后，可用于按日期筛选或进行按日期聚合的趋势分析。

## Data Processing Status

## 数据处理状态

The current project documentation does not provide evidence of a specific missing-value imputation procedure, outlier-removal procedure, score transformation procedure, or feature-engineering pipeline.

当前项目文档没有提供以下处理流程的具体证据：

* missing-value imputation
  缺失值填补；

* outlier removal or winsorization
  异常值删除或缩尾处理；

* score standardization or normalization
  成绩标准化或归一化；

* feature engineering
  特征工程。

Therefore, when users ask whether one of these procedures was applied, the correct response is:

> The current documentation does not provide this information, so it cannot be confirmed.

因此，当用户询问是否进行过上述处理时，正确回答应是：

> 当前文档未提供该信息，因此无法确认。

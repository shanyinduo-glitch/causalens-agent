# Project Background

# 项目背景

## Purpose

## 项目用途

This dataset is a simulated student performance dataset created for the CausaLens project.

本数据集是为 CausaLens 项目创建的模拟学生成绩数据集。

It is designed to demonstrate the following capabilities:

* exploratory data analysis
  探索性数据分析；

* descriptive statistics
  描述性统计；

* SQL-based data querying
  基于 SQL 的数据查询；

* chart generation
  图表生成；

* basic association analysis
  基础关联分析；

* Agent tool selection and evidence-grounded answers
  Agent 工具选择与基于证据的回答。

## Unit of Observation

## 观察单位

Each row represents one student.

每一行代表一名学生。

The dataset includes fields related to student identity, major, city, study hours, attendance, examination score, pass status, and examination date.

数据集包含与学生标识、专业、城市、学习时长、出勤率、考试成绩、是否通过考试以及考试日期相关的字段。

## Intended Analytical Uses

## 适合的分析类型

This dataset is suitable for:

* dataset profiling and data-quality inspection
  数据概览与数据质量检查；

* descriptive statistics, such as averages, medians, counts, and group comparisons
  描述性统计，例如均值、中位数、计数和分组比较；

* SQL queries, filtering, grouping, sorting, and Top-N summaries
  SQL 查询、筛选、分组、排序与 Top-N 汇总；

* visualizations, including bar charts, scatter plots, histograms, and line charts
  可视化，例如柱状图、散点图、直方图和折线图；

* exploratory analysis of associations between variables
  变量之间关联关系的探索性分析；

* demonstrations of tool-calling Agents and evidence-aware responses
  工具调用 Agent 与证据约束回答的演示。

## Important Limitations

## 重要限制

* This dataset is simulated. It should not be treated as real educational evidence or used to make real-world educational decisions.
  本数据集为模拟数据，不应视为真实教育证据，也不应用于现实教育决策。

* Observed relationships in this dataset apply only to the current sample.
  在本数据集中观察到的关系，仅适用于当前样本。

* Correlation, group differences, and ordinary regression results do not establish causality by themselves.
  相关性、分组差异和普通回归结果本身都不能直接证明因果关系。

* The dataset may omit important variables that affect both learning behavior and examination outcomes.
  数据集可能遗漏同时影响学习行为和考试结果的重要变量。

* The current documentation does not identify a real school, real students, a real collection process, or a real sampling frame.
  当前文档没有说明真实学校、真实学生、真实采集流程或真实抽样框。

* If a user asks for information that is not present in the documentation, the system should state that the information cannot be confirmed.
  若用户询问文档未提供的信息，系统应明确说明无法确认，而不是自行猜测。

## Evidence Rule

## 证据规则

When answering questions about dataset background, data collection, business meaning, or limitations, use the project documentation as the source of truth.

回答数据背景、采集方式、业务含义或局限性问题时，应以项目文档为依据。

If the documentation does not provide enough evidence, use wording such as:

> The current project documentation does not provide this information, so I cannot confirm it.

若文档没有提供足够依据，应使用类似表述：

> 当前项目文档未提供该信息，因此我无法确认。

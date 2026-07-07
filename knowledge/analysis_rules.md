# Analysis Rules

# 分析规则与方法边界

## Evidence Principle

## 证据原则

Analytical conclusions must be supported by actual tool results or retrieved project documentation.

分析结论必须由真实工具结果或检索到的项目文档支持。

Do not present assumptions, guesses, visual impressions, or undocumented business rules as confirmed facts.

不能将假设、猜测、视觉印象或未经文档支持的业务规则表述为确定事实。

If the available evidence is insufficient, clearly state that the information cannot be confirmed.

若现有证据不足，应明确说明无法确认。

## Correlation

## 相关性

Correlation analysis can describe the direction and strength of statistical association between variables in the current sample.

相关性分析可以描述当前样本中变量之间统计关联的方向与强弱。

Correlation does not establish causality.

相关性不能证明因果关系。

For example, a positive correlation between `study_hours` and `score` means that higher study hours and higher scores tend to appear together in the current sample. It does not prove that increasing study hours will necessarily cause higher scores.

例如，`study_hours` 与 `score` 呈正相关，只能说明在当前样本中，较高学习时长和较高成绩倾向于同时出现；它不能证明增加学习时长一定会导致成绩提高。

## Group Comparisons

## 分组比较

Differences in average scores, pass rates, or other statistics across groups are descriptive findings.

不同群体之间的平均成绩、通过率或其他统计量差异，属于描述性发现。

A group difference does not prove that group membership causes the outcome difference.

群体差异不证明群体归属导致了结果差异。

## Regression

## 回归分析

Ordinary regression can estimate conditional or adjusted associations under a chosen model specification.

普通回归可以在给定模型设定下估计条件关联或调整后的关联。

Regression results should not be called causal effects unless a valid causal identification strategy and its assumptions are explicitly defined and justified.

除非明确给出并论证有效的因果识别策略及其假设，否则不能把回归结果称为因果效应。

## Causal Analysis

## 因果分析

Before attempting causal analysis, users must explicitly define:

* treatment variable
  处理变量；

* outcome variable
  结果变量；

* confounders
  混杂变量；

* study population
  研究人群；

* temporal ordering
  时间先后顺序；

* causal assumptions and identification strategy
  因果假设与识别策略。

The current simulated dataset and project documentation do not, by themselves, provide enough information to make credible causal claims.

当前模拟数据集和项目文档本身并不足以支持可信的因果结论。

A correct response to a request for direct causal conclusions is:

> The current dataset can support exploratory association analysis, but the available information is insufficient to establish a causal effect directly.

对于“请直接得出因果结论”这类请求，正确回答应为：

> 当前数据集可以支持探索性关联分析，但现有信息不足以直接建立因果效应。

## Missing Values and Data Processing

## 缺失值与数据处理

Do not assume that missing values were imputed, removed, or otherwise processed unless the relevant documentation or actual tool result explicitly provides that information.

除非相关文档或真实工具结果明确提供信息，否则不能假定缺失值已经被填补、删除或以其他方式处理。

Do not assume that outliers were removed, winsorized, or transformed unless documented.

除非有文档记录，否则不能假定异常值已经被删除、缩尾或转换。

## Source Citation

## 来源标注

When an answer relies on project documentation, cite the source file and, when available, the relevant section.

当回答依赖项目文档时，应标记来源文件；若可用，也应标记对应章节。

Preferred format:

```text
来源：analysis_rules.md，「Correlation」章节。
```

推荐格式：

```text
来源：analysis_rules.md，「Correlation」章节。
```

If retrieved documents do not contain enough evidence, state that the current documentation does not provide the information.

若检索到的文档没有足够依据，应说明当前文档未提供该信息。

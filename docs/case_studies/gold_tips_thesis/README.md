# 黄金价格与 TIPS 收益率因果分析案例

本案例使用毕业论文中的真实金融时间序列数据，验证 CausaLens Agent 对连续冲击问题的变量配置、确定性方法路由、诊断、估计和报告能力。

## 1. 研究问题

美国 10 年期通胀保值国债（TIPS）收益率发生变化后，黄金日对数收益率在冲击当期及未来若干交易日如何响应？

## 2. 数据概况

- 样本区间：2006-01-05 至 2025-06-17
- 观测数量：4538 个交易日
- 字段数量：15
- 时间字段：`date`
- 处理变量：`t_diff`，10 年期 TIPS 收益率一阶差分
- 结果变量：`y_ret`，黄金下午定盘价日对数收益率
- 缺失值：0
- 重复整行：0
- 重复日期：0

Agent 使用的数据位于：

```text
data/private/gold_tips_thesis/processed/gold_tips_daily_stationary.csv
```

该目录已被 `.gitignore` 排除。真实数据只保存在本机，不随代码仓库发布。

## 3. 论文基准

论文使用连续处理变量的 `LinearDML`，并分别以 GBM、随机森林、XGBoost、LightGBM 和 ElasticNet 作为 nuisance models。论文基准结果保存在：

```text
docs/case_studies/gold_tips_thesis/reference_results/thesis_dml_model_comparison.csv
```

其中 GBM-DML 的平均处理效应为 `-0.038157`，标准误为 `0.005921`，95% 置信区间为 `[-0.049762, -0.026551]`。

## 4. Agent 当前验证方法

当前 CausaLens 的横截面 `LinearDML` 仅支持 0/1 二元处理，不能直接复现论文中的连续处理 DML。对本数据，确定性路由会选择：

- 数据结构：单条有序时间序列
- 处理类型：连续冲击
- 估计目标：IRF
- 估计器：Jordà-style Local Projection
- 推断：Newey-West HAC 标准误

因此，论文 DML 与 Agent Local Projection 的数值不能视为同一估计量。展示时应比较变量理解、效应方向、稳定性、动态响应和诊断，而不是要求两个系数完全相等。

## 5. 两套演示契约

### 核心控制变量

`examples/case_studies/gold_tips_thesis/lp_core.json`

控制变量：

- `dollar_ret`
- `sp500_ret`
- `fed_funds_diff`
- `vix_level`

这套配置适合现场主演示，变量较少，解释清楚。

### 扩展控制变量

`examples/case_studies/gold_tips_thesis/lp_full_controls.json`

在核心配置上加入 COMEX 库存、ETF 持仓、CPI 同比、五年期通胀预期和 GPR。该配置用于稳健性对照，不代表加入更多变量一定产生更强的识别。

预先生成的 `y_ret_lag1`、`t_diff_lag1` 不放入 confounders，因为 Local Projection 会根据契约自动创建 treatment 和 outcome 的滞后项。

## 6. 验收重点

一次合格的展示应证明：

1. Agent 正确识别 `t_diff` 为 Treatment、`y_ret` 为 Outcome、`date` 为时间字段。
2. 方法路由由明确契约和注册表决定，而不是由 LLM 随意选择。
3. 系统输出 h=0 至 h=5 的动态响应和置信区间。
4. 系统执行 ADF、未来 Treatment 安慰剂和 Ljung-Box 残差诊断。
5. 系统明确说明 HAC 修正不等于冲击外生性已经得到证明。
6. 系统可以导出 JSON 和 Markdown 报告。

## 7. 已验证的核心契约结果

核心契约已经在迁移后的数据上完成端到端运行，模型调用次数为 0：

- 主估计器：`TimeSeriesLocalProjection`
- h=0 累计响应：`-0.022476`
- h=0 的 95% 置信区间：`[-0.038508, -0.006444]`
- h=5 累计响应：`-0.059456`
- 未来 Treatment 安慰剂：通过，`p=0.1545`
- Ljung-Box 残差检验：存在自相关警告，`p=0.00236`
- 证据等级：`exploratory_only`

完整报告位于：

```text
reports/case_studies/gold_tips_thesis/lp_core_report.md
```

这些结果表明该案例可以用于展示，但残差动态仍需改进，不能将统计显著直接表述为已经确认的结构性因果效应。

## 8. 相关文档

- [变量字典](data_dictionary.md)
- [数据来源与迁移记录](source_manifest.md)
- [项目演示手册](DEMO_GUIDE.md)

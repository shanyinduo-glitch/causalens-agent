# 数据来源与迁移记录

## 迁移范围

本次只迁移 Agent 演示所需的最终分析数据和论文基准结果。原始 Excel、论文 PDF、答辩 PPT、中间图片、IDE 配置、缓存文件和重复目录未迁移。

| 用途 | 原位置 | 项目位置 | SHA-256 |
| --- | --- | --- | --- |
| Agent 最终输入数据 | `E:\graduation_project\data\cleaned_data_stationary.csv` | `data/private/gold_tips_thesis/processed/gold_tips_daily_stationary.csv` | `6299D5728160BB024E45E30C3CC94B481F9E5D5197C4D09E85B6C427105EEBB6` |
| 论文 DML 基准结果 | `E:\graduation_project\output\model_comparison_results.csv` | `docs/case_studies/gold_tips_thesis/reference_results/thesis_dml_model_comparison.csv` | `52EDC61367C730866B9D61925B8C2F32FF41703C0AB9FF6288E3614B608934C0` |

复制后的文件与源文件哈希一致，源文件未被移动或修改。

## 重复文件判断

以下两套目录中的关键文件哈希一致：

- `E:\graduation_project\data`
- `E:\graduation_project\代码数据及结果\data`

以及：

- `E:\graduation_project\causal_analysis_scripts`
- `E:\graduation_project\代码数据及结果\causal_analysis_scripts`

因此只选择较短、结构清晰的第一套路径作为迁移来源，避免重复复制。

## 未迁移原始数据的原因

原始数据涉及 FRED、Kaggle、世界黄金协会、CSMAR/校内授权数据和其他来源。部分文件名明确包含“仅供武汉大学使用”。这些数据不是 Agent 现场演示的必要输入，且可能存在再分发限制，所以继续保留在毕业设计目录，不进入主项目。

如需公开项目，只应发布得到授权的数据、字段结构说明或合成示例，不应直接提交本案例的私有 CSV。

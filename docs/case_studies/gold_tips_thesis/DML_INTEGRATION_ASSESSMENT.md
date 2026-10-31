# 黄金案例 DML 接入评估（2026-10-22）

本次为只读代码审查，未执行原论文训练、未验证原结果，也未注册新的估计器。当前网页黄金案例继续使用LP，不能称为论文DML复现。

## 已确认的两条原脚本路径

| 来源 | 数据与目标 | 拟合与分割 | 接入时需要保留的差异 |
| --- | --- | --- | --- |
| E:/graduation_project/causal_analysis_scripts/03_dml_simple_analysis.py | cleaned_data.csv；y、t；vix与dollar作为控制 | DoubleMLPLR；n_folds=5；未在脚本显式提供时间折分；随机森林与Lasso | 水平数据，不是当前平稳化样本；不能称为短期LP同一估计目标 |
| E:/graduation_project/causal_analysis_scripts/04_linearDML_analysis.py | cleaned_data_stationary.csv；y_ret、t_diff；12个候选协变量 | 按date排序，前80%训练；训练段拟合StandardScaler；LinearDML(discrete_treatment=False, cv=TimeSeriesSplit(n_splits=3))；X=X_train | 有效应修饰结构，不等同于只将协变量放入W的常数效应模型；ate_inference在训练X上汇总 |

04脚本还使用GBM、RandomForest、可选XGBoost/LightGBM和ElasticNetCV。后者的内部cv=5不是显式的时间折分。脚本对缺失协变量采取警告后忽略；正式复现应固定字段列表，不能静默改变模型。

## 尚不能直接接入的原因

- 需要明确哪一套脚本、哪一个学习器和哪个结果表对应论文最终结论。仅凭文件名不能认定全部路径都是最终模型。
- 需固定数据哈希、样本期、控制集、训练/评估样本及单位。当前LP基础4项控制与论文12项不是同一设定。
- 时间顺序拆分不等于推断自动有效：需审核交叉拟合覆盖、内层调参、预处理是否跨验证折、边界与时间间隔。StandardScaler在整个训练段上拟合，仍需评估对内层折分的影响。
- 需要核实所用EconML版本对TimeSeriesSplit的实际处理，以及ate_inference的推断口径是否适合序列相关。此次未运行，不能判定其置信区间已满足时间依赖条件。
- 连续处理每单位对比不能不加说明称为二元ATE；更不能与LP第5期累计响应直接作相同参数比较。
- CPI/GPR/ETF发布时间及删行造成的时序风险继续存在，换DML不能修复。

## 接入门槛与顺序

1. 固定论文复现契约：脚本版本、数据校验值、变量、X/W、学习器、种子、折分、目标和单位。
2. 将连续处理DML作为独立执行器与目标类型接入，不复用当前binary-only注册项。
3. 先以已知生成机制的合成数据测试参数恢复、时间泄漏和失败路径，再对接私人数据。
4. 审核时间依赖下的推断方案；默认软件置信区间不能未经说明就解释为稳健因果区间。
5. 分开提供“原论文配置复现”和“修订后探索配置”，所有差异可追溯；不以复现显著性为验收标准。

## 方法来源

- [DoubleML模型与目标定义](https://docs.doubleml.org/stable/guide/models.html)
- [DML原始研究](https://www.nber.org/papers/w23564)
- 方法来源解释理论，项目可执行能力以本地注册表为准。

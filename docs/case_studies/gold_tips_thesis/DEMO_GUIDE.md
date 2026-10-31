# 毕业论文真实数据演示手册

## 1. 演示前检查

在项目根目录打开 PowerShell：

```powershell
cd E:\Projects\causalens-agent
conda activate langchain_env
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m causalens.cli doctor
```

其中 `$env:PYTHONPATH = (Resolve-Path .\src).Path` 只对当前 PowerShell 窗口生效。关闭终端后重新打开，需要再次执行。

也可以使用项目提供的启动脚本，由脚本自动配置 `PYTHONPATH`：

```powershell
cd E:\Projects\causalens-agent
conda activate langchain_env
& .\scripts\run_thesis_demo.ps1 -Mode doctor
```

确认数据文件存在：

```powershell
Test-Path ".\data\private\gold_tips_thesis\processed\gold_tips_daily_stationary.csv"
```

应返回 `True`。

## 2. 网页主演示

启动页面：

```powershell
python -m streamlit run streamlit_app.py
```

或者使用自动配置环境的启动脚本：

```powershell
& .\scripts\run_thesis_demo.ps1 -Mode web -Port 8505
```

如果脚本提示当前 Python 无法导入 `causalens`，请显式指定项目环境：

```powershell
& .\scripts\run_thesis_demo.ps1 -Mode web -Port 8505 `
    -PythonPath "D:\Anaconda\envs\langchain_env\python.exe"
```

在浏览器中按下面顺序操作：

### 推荐：首次使用的对话探索路径

这条路径最适合展示 Agent 如何把自然语言、数据工具、RAG 和人工确认串起来：

1. 打开 `http://127.0.0.1:8505/`，工作区保持“对话探索工作台”。
2. 在左侧“研究资料”上传 CSV，并在“研究背景与已知条件”填写一段非敏感背景，例如：
   `研究 TIPS 实际收益率变化与黄金收益率的关系；t_diff 是收益率差分，y_ret 是黄金对数收益率，数据按交易日排列。`
3. 点击“建立／更新本地知识库”。系统会把用户背景和当前数据集字段结构切成片段，按当前数据 SHA256 建立本地版本；如果机器没有嵌入模型，会明确显示“离线词项检索”，仍然可以检索。
4. 在左侧勾选模型访问授权。这个勾选必须由演示者本人完成，未授权时发送按钮会保持禁用。
5. 在“现在问助手”输入：
   `检查 t_diff 和 y_ret 的缺失情况，描述统计并绘制散点图；只描述关联，不执行因果估计。`
6. 点击“发送并开始探索”，展示 Agent 先调用 `profile_dataset`、只读 SQL 和 `create_chart`，再引用本地知识库解释结果。
7. 第二轮可输入：
   `结合刚才的结果，说明下一步做因果探索前还需要确认哪些变量和假设。`
   这会复用上一轮结果和 RAG 证据；若选择“只解读上一轮结果”，不会重复运行统计工具。
8. 若要进入因果估计，点击“基于当前问题确认研究设计”，在单独的研究设计页面核对 treatment、outcome、时间字段和方法，再由人工批准执行。

知识库片段可在“查看当前数据版本的知识库片段”中查看；每个片段会标出“用户研究背景”或“当前数据集结构与概况”。修改研究背景后，页面会提示知识库已过期，需要重新建立。删除按钮只清除当前数据版本的本地知识库。

### 论文案例的固定契约路径

1. 上传 `data/private/gold_tips_thesis/processed/gold_tips_daily_stationary.csv`。
2. 选择“快速引导”。
3. Treatment 选择 `t_diff`。
4. Outcome 选择 `y_ret`。
5. 混杂变量先选择 `dollar_ret`、`sp500_ret`、`fed_funds_diff`、`vix_level`。
6. 数据设计选择“连续冲击时间序列（Local Projection）”。
7. 估计目标选择 `IRF`。
8. 时间字段选择 `date`。
9. 目标总体填写“2006 年 1 月至 2025 年 6 月样本覆盖的交易日”。
10. 每一行填写“一个交易日”。
11. 结果窗口填写“冲击当期至未来 5 个交易日”。
12. 高级选项设置最大 horizon 为 5，Treatment 滞后 1，Outcome 滞后 1，控制变量滞后 1，选择累计响应，HAC 留空自动选择。
13. 检查系统生成的契约和准备度结果。
14. 确认变量没有选反后批准执行。
15. 展示 h=0 至 h=5 的响应、置信区间、安慰剂检验、残差诊断和证据等级。

这一路径只有在主动点击“AI 辅助理解”时才需要 LLM。现场演示可以完全不消耗模型 Token。

## 3. CLI 零 Token 验收

推荐直接使用启动脚本进行准备度检查：

```powershell
& .\scripts\run_thesis_demo.ps1 -Mode check
```

正式执行并导出结果：

```powershell
& .\scripts\run_thesis_demo.ps1 -Mode run
```

脚本会自动进入项目根目录、设置 `PYTHONPATH`、验证数据和契约文件，再调用 CausaLens。

如果希望手动执行，使用下面的完整命令。

先只运行准备度检查：

```powershell
python -m causalens.cli contract-run `
  --dataset-path ".\data\private\gold_tips_thesis\processed\gold_tips_daily_stationary.csv" `
  --contract-path ".\examples\case_studies\gold_tips_thesis\lp_core.json"
```

确认后执行并保存结果：

```powershell
python -m causalens.cli contract-run `
  --dataset-path ".\data\private\gold_tips_thesis\processed\gold_tips_daily_stationary.csv" `
  --contract-path ".\examples\case_studies\gold_tips_thesis\lp_core.json" `
  --approve `
  --output-json ".\outputs\case_studies\gold_tips_thesis\lp_core_result.json" `
  --report-path ".\reports\case_studies\gold_tips_thesis\lp_core_report.md" `
  --cache-dir ".\storage\causal_cache"
```

再将契约替换成 `lp_full_controls.json`，输出名改成 `lp_full_controls_result.json` 和 `lp_full_controls_report.md`，用于稳健性对照。

## 4. `No module named 'causalens'` 的处理方法

这个错误表示当前 Python 没有安装项目包，并且当前终端没有配置 `src` 搜索路径。它与 CSV 和因果契约无关。

在项目根目录执行：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -c "import causalens; print(causalens.__file__)"
```

如果输出路径以 `E:\Projects\causalens-agent\src\causalens` 开头，说明已经修复。然后重新运行原命令。

也可以进行一次性可编辑安装，使以后新开的终端不再需要设置 `PYTHONPATH`：

```powershell
python -m pip install -e . --no-deps --no-build-isolation
```

安装后验证：

```powershell
python -m causalens.cli doctor
```

## 5. 建议的 6 分钟讲解顺序

### 第 1 分钟：研究问题

说明论文关注 TIPS 实际利率变化对黄金收益率的影响。明确 Treatment 是 `t_diff`，Outcome 是 `y_ret`。

### 第 2 分钟：真实数据

展示 4538 行、15 列、2006-01-05 至 2025-06-17。指出数据已经平稳化且没有缺失值、重复日期。

### 第 3 分钟：Agent 如何选方法

说明用户或 LLM 负责把自然语言整理成研究契约，真正的方法选择由确定性规则完成。连续 Treatment、单条时间序列、IRF 会路由到 Local Projection。

### 第 4 分钟：执行前安全检查

展示时间排序、重复日期、样本量、变量类型和平稳性提示。强调 Agent 不会在变量角色冲突或准备度不足时直接估计。

### 第 5 分钟：动态响应

展示 h=0 至 h=5 的估计值与置信区间。解释系数单位是 TIPS 收益率增加 1 个百分点时黄金对数收益率的响应。

当前核心契约的 h=0 累计响应约为 `-0.0225`，95% 置信区间约为 `[-0.0385, -0.0064]`；h=5 累计响应约为 `-0.0595`。这些数字用于复现实验展示，不应与论文 DML 的 ATE 做一对一等值比较。

### 第 6 分钟：证据边界

展示未来 Treatment 安慰剂和 Ljung-Box 诊断。说明 Newey-West HAC 处理标准误问题，但不能证明冲击完全外生。最后下载 JSON 和 Markdown 报告。

当前核心契约的未来 Treatment 安慰剂通过（`p=0.1545`），但 Ljung-Box 检验提示残差仍有自相关（`p=0.00236`），因此证据等级为 `exploratory_only`。现场应主动展示这个警告，它能体现 Agent 不会把显著系数包装成无条件的强因果结论。

## 6. 不应这样表述

- 不要说“LLM 自动证明了因果关系”。
- 不要把论文 DML 的 ATE 与 Local Projection 的某一期 IRF 当成同一个估计量。
- 不要因为控制变量更多就宣称结果一定更可信。
- 不要把低频宏观变量的前向填充当作发布日期已经严格处理。
- 不要公开提交带有校内使用限制的原始数据。

## 7. 可回答的常见问题

**为什么不用论文原来的 DML？**

当前 Agent 的横截面 LinearDML 只支持二元 Treatment，而本案例的 `t_diff` 是连续变量。当前版本使用 Local Projection 展示连续冲击的动态响应。连续处理 DML 是后续扩展项。

**LLM 决定方法吗？**

不是。LLM 可以帮助理解自然语言并生成候选契约；用户确认变量角色后，方法由数据结构、Treatment 类型、估计目标和注册表确定。

**这能证明 TIPS 导致黄金变化吗？**

不能单靠一次回归完全证明。结果依赖没有遗漏同步冲击、变量时间对齐正确等识别假设。Agent 会报告诊断和证据边界，而不是把统计显著直接写成确定因果。

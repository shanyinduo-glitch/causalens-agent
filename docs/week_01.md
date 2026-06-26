# Week 1 学习记录

## Day 1：项目初始化与环境搭建

- 创建 GitHub 仓库
- 克隆项目到本地
- 建立基础目录结构

## Day 2：Pydantic 与结构化输出

- 新建 `src/causalens/schemas.py`
- 使用 `BaseModel` 定义 Agent 请求、工具调用、工具结果和最终回答的数据结构
- 使用 `ValidationError` 处理不符合规则的输入
- 学会使用 `model_dump()` 将 Pydantic 对象转换为普通字典

### 今日理解

- Schema 用来规定数据格式，避免 Agent 或用户传入错误参数。
- `ToolCall` 记录 Agent 想调用什么工具和传入哪些参数。
- `ToolResult` 统一记录工具成功或失败的结果。
- 工具失败时应返回 `success=False`、`result=None` 和清楚的 `error` 信息。

## Day 3：实现第一个工具

- 新建 `src/causalens/tools/calculator.py`
- 实现 `calculator(expression)`，用于计算基础数学表达式
- 新建 `src/causalens/tools/data_profile.py`
- 实现 `profile_dataset(dataset_path)`，用于读取 CSV 并生成数据集概况
- 新建 `scripts/tool_demo.py`，集中演示两个工具的成功与失败情况

### 今日理解

- Agent 的工具本质上是可以被程序或模型调用的 Python 函数。
- 工具应接收清楚的输入，并统一返回 `ToolResult`。
- `calculator` 先检查表达式是否只包含允许字符，再尝试计算。
- `profile_dataset` 使用 pandas 读取 CSV，并返回行数、列数、字段名、数据类型、缺失值和数值统计。
- 工具不应该因为一个错误输入让整个程序崩掉；失败时返回 `success=False` 和可读的 `error` 信息。

## Day 4：将 Python 工具包装为千问 Function Tool

* 新建 `src/causalens/agent/tool_registry.py`
* 建立 `TOOL_REGISTRY`，将工具名称 `profile_dataset` 映射到真实 Python 函数
* 编写 `TOOLS`，向千问说明工具名称、用途和参数格式
* 新建 `src/causalens/agent/qwen_client.py`，统一封装千问 OpenAI 兼容接口调用
* 新建 `scripts/function_call_demo.py`，验证千问是否能选择正确工具并生成正确参数

### 今日理解

* `TOOL_REGISTRY` 是工具名称和真实 Python 函数之间的映射表。
* `TOOLS` 是给大模型看的工具说明书，包含工具名称、用途、参数类型和必填参数。
* 模型返回的 `tool_calls` 表示“请求调用工具”，并不代表工具已经被执行。
* 模型返回的工具参数通常是 JSON 字符串，需要用 `json.loads()` 转换成 Python 字典。
* Function Calling 的第一步是：模型负责选择工具和填写参数；程序负责真正执行工具。

```
```

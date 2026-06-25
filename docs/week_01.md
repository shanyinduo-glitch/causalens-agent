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
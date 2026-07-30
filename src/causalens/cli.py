from pathlib import Path

import typer
from rich.console import Console

from causalens.agent.graph import run_causalens_graph
from causalens.schemas import ToolResult
from rich.table import Table
from rich.text import Text
from causalens.response_evidence import build_evidence_rows
from causalens.reporting import save_markdown_report

app = typer.Typer(
    no_args_is_help=True,
    help="CausaLens：一个基于工具调用的 CSV 数据分析助手。",
    invoke_without_command=False,
)

console = Console()

@app.callback()
def main() -> None:
    """CausaLens 命令行工具。"""

@app.command()
def ask(
    dataset_path: str = typer.Option(
        ...,
        "--dataset-path",
        help="要分析的 CSV 文件路径，例如 data/sample.csv。",
    ),
    question: str = typer.Option(
        ...,
        "--question",
        help="你想询问的数据问题。",
    ),
    report_path: Path | None = typer.Option(
        None,
        "--report-path",
         help=(
                 "可选的 Markdown 报告保存路径。"
                 "未提供时只在终端显示结果。"
        ),
    ),
) -> None:
    """通过 LangGraph 工作流分析 CSV 文件并回答问题。"""

    path = Path(dataset_path)

    # 第一层：调用模型前检查数据路径，
    # 避免因为路径无效而发起无意义的模型请求。
    if not path.exists():
        console.print(
            f"[bold red]错误：找不到数据文件：{dataset_path}[/bold red]"
        )
        raise typer.Exit(code=1)

    if not path.is_file():
        console.print(
            f"[bold red]错误：数据路径不是文件：{dataset_path}[/bold red]"
        )
        raise typer.Exit(code=1)

    try:
        final_state = run_causalens_graph(
            dataset_path=dataset_path,
            question=question,
        )

    except Exception as exc:
        console.print(
            f"[bold red]Agent 运行失败：{exc}[/bold red]"
        )
        raise typer.Exit(code=1)

    task_type = final_state.get(
        "task_type",
        "未识别",
    )
    workflow_status = final_state.get(
        "workflow_status",
        "未知",
    )
    plan = final_state.get(
        "plan",
        [],
    )
    tool_calls = final_state.get(
        "tool_calls",
        [],
    )
    raw_tool_results = final_state.get(
        "tool_results",
        [],
    )

    tool_results = [
        (
            tool_result
            if isinstance(tool_result, ToolResult)
            else ToolResult.model_validate(tool_result)
        )
        for tool_result in raw_tool_results
    ]
    final_answer = final_state.get(
        "final_answer",
        "未生成最终回答。",
    )
    error = final_state.get("error")

    console.print("\n[bold cyan]用户问题：[/bold cyan]")
    console.print(question)

    console.print("\n[bold cyan]任务类型：[/bold cyan]")
    console.print(str(task_type))

    console.print("\n[bold cyan]工作流状态：[/bold cyan]")
    console.print(str(workflow_status))

    console.print("\n[bold cyan]分析计划：[/bold cyan]")

    if plan:
        for index, step in enumerate(plan, start=1):
            console.print(f"{index}. {step}")
    else:
        console.print("暂无分析计划。")

    if tool_calls:
        console.print("\n[bold cyan]工具调用：[/bold cyan]")

        for tool_call in tool_calls:
            # LangGraph 状态中的工具调用当前以字典形式保存。
            # 同时保留对旧对象形式的兼容，避免展示层强依赖一种结构。
            if isinstance(tool_call, dict):
                tool_name = tool_call.get(
                    "tool_name",
                    "未知工具",
                )
                arguments = tool_call.get(
                    "arguments",
                    {},
                )
            else:
                tool_name = getattr(
                    tool_call,
                    "tool_name",
                    "未知工具",
                )
                arguments = getattr(
                    tool_call,
                    "arguments",
                    {},
                )

            console.print(
                f"- {tool_name}({arguments})"
            )
    else:
        console.print("\n[bold cyan]工具调用：[/bold cyan]")
        console.print("未调用工具。")

    console.print("\n[bold green]最终回答：[/bold green]")
    console.print(final_answer)

    failed_results = [
        tool_result
        for tool_result in tool_results
        if not tool_result.success
    ]

    if tool_results:
        evidence_rows = build_evidence_rows(
            tool_results,
        )

        evidence_table = Table(
            title="执行依据（来自实际工具结果）",
            show_lines=True,
        )

        evidence_table.add_column(
            "工具",
            style="cyan",
            no_wrap=True,
        )

        evidence_table.add_column(
            "状态",
            no_wrap=True,
        )

        evidence_table.add_column(
            "关键事实",
            overflow="fold",
        )

        for evidence_row in evidence_rows:
            evidence_table.add_row(
                Text(evidence_row["tool_name"]),
                Text(evidence_row["status"]),
                Text(evidence_row["evidence"]),
            )

        console.print()
        console.print(evidence_table)

    if report_path is not None:
        try:
            saved_report_path = save_markdown_report(
                final_state,
                report_path,
                dataset_path=dataset_path,
            )
        except Exception as exc:
            console.print()
            console.print(
                f"[bold red]Markdown 报告保存失败：{exc}[/bold red]"
            )
            raise typer.Exit(code=1)

        console.print()
        console.print(
            "[bold green]Markdown 报告已保存：[/bold green]"
        )
        console.print(str(saved_report_path))

    # Graph 已经进入 Reporter，因此即使失败，
    # 也先展示 Reporter 给出的最终说明，再以非零状态退出。
    if error:
        console.print("\n[bold red]工作流错误：[/bold red]")
        console.print(error)
        raise typer.Exit(code=1)

    if failed_results:
        console.print("\n[bold red]数据分析未完成：[/bold red]")

        for tool_result in failed_results:
            console.print(
                f"- {tool_result.tool_name}：{tool_result.error}"
            )

        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
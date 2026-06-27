from pathlib import Path

import typer
from rich.console import Console

from causalens.agent.tool_loop import run_tool_agent


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
) -> None:
    """分析一个 CSV 文件并回答问题。"""

    path = Path(dataset_path)

    # 第一层：在调用模型前检查路径，避免无意义的 API 请求。
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
        response = run_tool_agent(
            question=question,
            dataset_path=dataset_path,
        )

    except Exception as exc:
        console.print(
            f"[bold red]Agent 运行失败：{exc}[/bold red]"
        )
        raise typer.Exit(code=1)

    failed_results = [
        tool_result
        for tool_result in response.tool_results
        if not tool_result.success
    ]

    # 第二层：工具失败时，CLI 输出明确错误，而不是假装分析成功。
    if failed_results:
        console.print("[bold red]数据分析未完成：[/bold red]")

        for tool_result in failed_results:
            console.print(
                f"- {tool_result.tool_name}：{tool_result.error}"
            )

        raise typer.Exit(code=1)

    console.print("\n[bold green]回答：[/bold green]")
    console.print(response.answer)

    if response.tool_calls:
        console.print("\n[dim]本次调用的工具：[/dim]")

        for tool_call in response.tool_calls:
            console.print(
                f"[dim]- {tool_call.tool_name}"
                f"({tool_call.arguments})[/dim]"
            )


if __name__ == "__main__":
    app()
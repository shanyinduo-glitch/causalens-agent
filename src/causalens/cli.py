import json
from pathlib import Path

import typer
from rich.console import Console

from causalens.schemas import ToolResult
from rich.table import Table
from rich.text import Text
from causalens.response_evidence import build_evidence_rows
from causalens.reporting import save_markdown_report
from causalens.preflight import run_preflight


def run_causalens_graph(*args, **kwargs):
    """延迟加载完整 Agent，使 doctor 在统计依赖缺失时仍可启动。"""

    from causalens.agent.graph import run_causalens_graph as run_graph

    return run_graph(*args, **kwargs)


def run_explicit_causal_contract(*args, **kwargs):
    """延迟加载估计器；同时保留便于测试替换的稳定入口。"""

    from causalens.causal.contract_runner import (
        run_explicit_causal_contract as run_contract,
    )

    return run_contract(*args, **kwargs)

app = typer.Typer(
    no_args_is_help=True,
    help="CausaLens：一个基于工具调用的 CSV 数据分析助手。",
    invoke_without_command=False,
)

console = Console()

@app.callback()
def main() -> None:
    """CausaLens 命令行工具。"""


@app.command("doctor")
def doctor() -> None:
    """离线检查因果分析环境；不会请求模型或显示 API Key。"""

    report = run_preflight(Path.cwd())
    console.print("[bold cyan]CausaLens 运行前检查[/bold cyan]")
    console.print(
        "离线显式契约："
        + ("[green]可用[/green]" if report["offline_contract_runner_ready"] else "[red]不可用[/red]")
    )
    console.print(
        "自然语言 Agent："
        + ("[green]可用[/green]" if report["natural_language_agent_ready"] else "[yellow]尚未就绪[/yellow]")
    )
    for check in report["checks"]:
        marker = "[green]通过[/green]" if check["passed"] else "[red]未通过[/red]"
        console.print(f"- {check['name']}: {marker} — {check['detail']}")
    for warning in report["warnings"]:
        console.print(f"[yellow]提示：{warning}[/yellow]")
    if not report["offline_contract_runner_ready"]:
        raise typer.Exit(code=1)


def _contract_report_state(
    result: dict,
    *,
    dataset_path: str,
    contract_path: Path,
) -> dict:
    estimation = result.get("estimation")
    succeeded = bool(estimation and estimation.get("success") is True)
    errors = result.get("errors") or []
    if result["workflow_status"] == "completed":
        final_answer = "显式因果契约已完成估计；详细估计值、诊断与限制见下方结果。"
    elif result["workflow_status"] == "awaiting_approval":
        final_answer = "研究契约和数据检查已通过；添加 --approve 后才会执行因果估计。"
    else:
        final_answer = "显式因果契约未能执行。\n\n" + "\n".join(
            f"- {item}" for item in errors
        )
    return {
        "dataset_path": dataset_path,
        "question": f"执行显式因果契约：{contract_path}",
        "task_type": "causal_analysis",
        "workflow_status": result["workflow_status"],
        "plan": ["校验研究契约", "检查识别与数据质量", "检查设计准备度", "执行已批准的因果估计"],
        "causal_question": result.get("contract"),
        "causal_identification": result.get("identification"),
        "causal_data_quality": result.get("data_quality"),
        "causal_readiness": result.get("readiness"),
        "causal_diagnostics": result.get("diagnostics"),
        "tool_calls": ([{"tool_name": "run_causal_estimators", "arguments": {}}] if estimation else []),
        "tool_results": (
            [
                {
                    "tool_name": "run_causal_estimators",
                    "success": succeeded,
                    "result": estimation.get("result") if estimation else None,
                    "error": estimation.get("error") if estimation else None,
                }
            ]
            if estimation
            else []
        ),
        "final_answer": final_answer,
        "error": "; ".join(str(item) for item in errors) or None,
    }


@app.command("contract-run")
def contract_run(
    dataset_path: Path = typer.Option(..., "--dataset-path", help="要分析的 CSV 文件路径。"),
    contract_path: Path = typer.Option(..., "--contract-path", help="显式因果研究契约 JSON 文件。"),
    approve: bool = typer.Option(
        False,
        "--approve",
        help="明确批准执行估计；省略时只做校验、诊断和方法规划。",
    ),
    output_json: Path | None = typer.Option(None, "--output-json", help="可选的完整 JSON 结果路径。"),
    report_path: Path | None = typer.Option(None, "--report-path", help="可选的 Markdown 报告路径。"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="可选的本地成功结果缓存目录。"),
    chart_dir: Path | None = typer.Option(None, "--chart-dir", help="可选的诊断图输出目录。"),
) -> None:
    """无需大模型，校验并运行明确写出的因果研究契约。"""

    if not dataset_path.is_file():
        console.print(f"[bold red]错误：找不到数据文件：{dataset_path}[/bold red]")
        raise typer.Exit(code=1)
    if not contract_path.is_file():
        console.print(f"[bold red]错误：找不到契约文件：{contract_path}[/bold red]")
        raise typer.Exit(code=1)
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]契约 JSON 读取失败：{exc}[/bold red]")
        raise typer.Exit(code=1)
    if not isinstance(contract, dict):
        console.print("[bold red]契约 JSON 顶层必须是对象。[/bold red]")
        raise typer.Exit(code=1)

    result = run_explicit_causal_contract(
        dataset_path=dataset_path,
        contract=contract,
        approved=approve,
        result_cache_dir=cache_dir,
        chart_output_dir=chart_dir,
    )
    console.print(f"[bold cyan]工作流状态：[/bold cyan] {result['workflow_status']}")
    console.print(f"[bold cyan]当前阶段：[/bold cyan] {result['stage']}")
    console.print("[bold cyan]模型调用次数：[/bold cyan] 0")
    design = result.get("design_selection") or {}
    if design:
        console.print(
            "[bold cyan]选定估计器：[/bold cyan] "
            + ", ".join(design.get("selected_estimator_names") or [])
        )
    for warning in result.get("warnings") or []:
        console.print(f"[yellow]提示：{warning}[/yellow]")
    for error in result.get("errors") or []:
        console.print(f"[bold red]阻塞/失败：{error}[/bold red]")
    if result["workflow_status"] == "awaiting_approval":
        console.print("[green]检查已通过。确认研究契约后，使用 --approve 执行估计。[/green]")
    elif result["workflow_status"] == "completed":
        console.print("[bold green]因果估计已完成。[/bold green]")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"[green]JSON 结果已保存：[/green]{output_json}")
    if report_path is not None:
        save_markdown_report(
            _contract_report_state(
                result,
                dataset_path=str(dataset_path),
                contract_path=contract_path,
            ),
            report_path,
            dataset_path=dataset_path,
        )
        console.print(f"[green]Markdown 报告已保存：[/green]{report_path}")
    if result["workflow_status"] in {"blocked", "failed"}:
        raise typer.Exit(code=1)

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

    # 因果任务第一次运行只执行 readiness，
    # 不会直接运行 DML。
    if (
        final_state.get("workflow_status")
        == "awaiting_approval"
        and final_state.get(
            "requires_approval"
        )
        is True
    ):
        treatment = final_state.get(
            "treatment"
        )
        outcome = final_state.get(
            "outcome"
        )
        confounders = final_state.get(
            "confounders",
            [],
        )
        causal_question = final_state.get("causal_question") or {}
        variable_roles = (
            causal_question.get("variable_roles") or {}
            if isinstance(causal_question, dict)
            else {}
        )
        negative_control_outcomes = (
            variable_roles.get("negative_control_outcomes") or []
            if isinstance(variable_roles, dict)
            else []
        )
        instruments = (
            variable_roles.get("instruments") or []
            if isinstance(variable_roles, dict)
            else []
        )
        estimand = (
            str(causal_question.get("estimand") or "ATE")
            if isinstance(causal_question, dict)
            else "ATE"
        )
        estimator_description = (
            "TimeSeriesLocalProjection 估计每 1 个原始单位连续冲击的 IRF"
            if isinstance(causal_question, dict)
            and causal_question.get("design_type") == "time_series"
            else "Difference-in-Differences 估计处理组 ATT"
            if isinstance(causal_question, dict)
            and causal_question.get("design_type") == "difference_in_differences"
            else "TwoStageLeastSquares 估计工具变量 LATE"
            if isinstance(causal_question, dict)
            and causal_question.get("design_type") == "instrumental_variable"
            else "SharpRegressionDiscontinuity 估计 cutoff 处的 LATE"
            if isinstance(causal_question, dict)
            and causal_question.get("design_type") == "regression_discontinuity"
            else "SyntheticControl 估计单个处理单位的 ATT"
            if isinstance(causal_question, dict)
            and causal_question.get("design_type") == "synthetic_control"
            else "CausalForestDML 估计条件平均处理效应（CATE）"
            if estimand == "CATE"
            else "AIPW 作为主估计器、LinearDML 作为稳健性估计器，"
            "估计平均处理效应（ATE）"
        )
        readiness = final_state.get(
            "causal_readiness",
            {},
        )

        if not isinstance(readiness, dict):
            readiness = {}

        sample_summary = readiness.get(
            "sample_summary",
            {},
        )
        warnings = readiness.get(
            "warnings",
            [],
        )

        if not isinstance(
            sample_summary,
            dict,
        ):
            sample_summary = {}

        if not isinstance(warnings, list):
            warnings = []

        console.print()
        console.print(
            "[bold yellow]"
            "因果分析准备度检查已通过"
            "[/bold yellow]"
        )

        console.print(
            f"- treatment：{treatment}"
        )
        console.print(
            f"- outcome：{outcome}"
        )
        console.print(
            f"- confounders：{confounders}"
        )
        console.print(
            f"- negative-control outcomes：{negative_control_outcomes}"
        )
        console.print(f"- instruments：{instruments}")
        if isinstance(causal_question, dict) and causal_question.get(
            "design_type"
        ) == "regression_discontinuity":
            console.print(
                f"- running variable：{causal_question.get('running_variable')}"
            )
            console.print(f"- cutoff：{causal_question.get('cutoff')}")
            console.print(f"- bandwidth：{causal_question.get('bandwidth')}")
        if isinstance(causal_question, dict) and causal_question.get(
            "design_type"
        ) == "synthetic_control":
            console.print(f"- unit ID：{causal_question.get('unit_id_column')}")
            console.print(f"- time：{causal_question.get('time_column')}")
            console.print(
                f"- intervention time：{causal_question.get('intervention_time')}"
            )
            console.print(
                f"- treated unit：{causal_question.get('treated_unit') or '自动推断'}"
            )
        if isinstance(causal_question, dict) and causal_question.get(
            "design_type"
        ) == "time_series":
            console.print(f"- time：{causal_question.get('time_column')}")
            console.print(
                f"- maximum horizon：{causal_question.get('max_horizon')}"
            )
            console.print(
                "- response："
                + (
                    "cumulative"
                    if causal_question.get("cumulative_outcome", True)
                    else "point-in-time"
                )
            )
        console.print(
            "- 完整样本数："
            f"{sample_summary.get('complete_cases')}"
        )
        console.print(
            "- treatment 分组："
            f"{sample_summary.get('treatment_counts')}"
        )

        if warnings:
            console.print()
            console.print(
                "[bold yellow]"
                "执行前警告："
                "[/bold yellow]"
            )

            for warning in warnings:
                console.print(
                    f"- {warning}"
                )

        console.print()
        console.print(
            f"下一步将使用 {estimator_description}。"
        )
        console.print(
            "准备度通过只代表满足最低技术条件，"
            "不代表无未观测混杂等因果假设已经成立。"
        )

        approved = typer.confirm(
            f"是否继续执行 {estimator_description}？",
            default=False,
        )

        if approved:
            try:
                final_state = (
                    run_causalens_graph(
                        dataset_path=(
                            dataset_path
                        ),
                        question=question,
                        approved=True,
                    )
                )

            except Exception as exc:
                console.print(
                    "[bold red]"
                    "因果分析执行失败："
                    f"{exc}"
                    "[/bold red]"
                )
                raise typer.Exit(code=1)

        else:
            previous_answer = str(
                final_state.get(
                    "final_answer",
                    "",
                )
            ).strip()

            cancellation_message = (
                "用户未确认继续执行，"
                "因此因果估计器未运行，"
                "也没有产生 ATE 估计。"
            )

            if previous_answer:
                final_answer = (
                    f"{previous_answer}\n\n"
                    f"{cancellation_message}"
                )
            else:
                final_answer = (
                    cancellation_message
                )

            final_state = {
                **final_state,
                "approved": False,
                "requires_approval": False,
                "workflow_status": "completed",
                "final_answer": final_answer,
                "error": None,
            }

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
        raise typer.Exit(code=1)

    if failed_results:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

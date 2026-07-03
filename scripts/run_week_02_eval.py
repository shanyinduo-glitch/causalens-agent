from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from rich.console import Console
from rich.table import Table

from causalens.agent.tool_loop import run_tool_agent
from causalens.evaluation import (
    evaluate_agent_response,
    load_evaluation_cases,
)


CASE_PATH = Path("evals/week_02_cases.jsonl")
RESULT_PATH = Path(
    "outputs/evals/week_02_results.jsonl"
)

DEFAULT_DATASET_PATH = Path(
    "data/student_performance.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CausaLens Week 2 evaluation cases "
            "against the real Agent."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前 N 条评测题。",
    )

    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help=(
            "只运行指定 id，可重复传入。"
            "例如 --case-id w2_01 --case-id w2_14"
        ),
    )

    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="用于评测的 CSV 数据集路径。",
    )

    return parser.parse_args()


def select_cases(
    cases: list[dict[str, Any]],
    case_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected_cases = cases

    if case_ids:
        requested_ids = set(case_ids)

        selected_cases = [
            case
            for case in selected_cases
            if case["id"] in requested_ids
        ]

    if limit is not None:
        selected_cases = selected_cases[:limit]

    return selected_cases


def build_result_record(
    case: dict[str, Any],
    evaluation: dict[str, Any],
    duration_ms: float,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "duration_ms": round(duration_ms, 2),
        "notes": case.get("notes"),
        **evaluation,
    }


def write_results(
    result_path: Path,
    records: list[dict[str, Any]],
) -> None:
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as result_file:
        for record in records:
            result_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    console = Console()

    cases = load_evaluation_cases(CASE_PATH)

    selected_cases = select_cases(
        cases=cases,
        case_ids=args.case_id,
        limit=args.limit,
    )

    if not selected_cases:
        console.print(
            "[red]没有选中任何评测题。"
            "请检查 --case-id 或 --limit 参数。[/red]"
        )
        return

    console.print(
        f"\n[bold]开始运行 {len(selected_cases)} 条评测题。[/bold]"
    )

    records: list[dict[str, Any]] = []

    for index, case in enumerate(
        selected_cases,
        start=1,
    ):
        console.print(
            f"\n[{index}/{len(selected_cases)}] "
            f"{case['id']}: {case['question']}"
        )

        started_at = perf_counter()

        try:
            response = run_tool_agent(
                question=case["question"],
                dataset_path=args.dataset_path,
            )

            duration_ms = (
                perf_counter() - started_at
            ) * 1000

            evaluation = evaluate_agent_response(
                case,
                response,
            )

        except Exception as exc:
            duration_ms = (
                perf_counter() - started_at
            ) * 1000

            evaluation = {
                "id": case["id"],
                "category": case.get(
                    "category",
                    "unknown",
                ),
                "question": case["question"],
                "passed": False,
                "checks": {
                    "agent_execution_completed": False,
                },
                "expected_tools": case.get(
                    "required_tools",
                    [],
                ),
                "forbidden_tools": case.get(
                    "forbidden_tools",
                    [],
                ),
                "expected_chart_type": case.get(
                    "expected_chart_type",
                ),
                "actual_tools": [],
                "tool_success_map": {},
                "answer": "",
                "execution_error": str(exc),
            }

        record = build_result_record(
            case=case,
            evaluation=evaluation,
            duration_ms=duration_ms,
        )

        records.append(record)

        status = (
            "[green]PASS[/green]"
            if record["passed"]
            else "[red]FAIL[/red]"
        )

        console.print(
            f"  {status} | "
            f"工具: {record['actual_tools']} | "
            f"耗时: {record['duration_ms']} ms"
        )

    write_results(
        result_path=RESULT_PATH,
        records=records,
    )

    passed_count = sum(
        record["passed"]
        for record in records
    )

    table = Table(
        title="Week 2 Evaluation Summary",
        show_lines=True,
    )

    table.add_column("类别")
    table.add_column("总数", justify="right")
    table.add_column("通过", justify="right")
    table.add_column("通过率", justify="right")

    categories = sorted(
        {
            record["category"]
            for record in records
        }
    )

    for category in categories:
        category_records = [
            record
            for record in records
            if record["category"] == category
        ]

        category_total = len(category_records)

        category_passed = sum(
            record["passed"]
            for record in category_records
        )

        pass_rate = (
            category_passed / category_total
            if category_total
            else 0
        )

        table.add_row(
            category,
            str(category_total),
            str(category_passed),
            f"{pass_rate:.0%}",
        )

    total_pass_rate = passed_count / len(records)

    table.add_section()

    table.add_row(
        "total",
        str(len(records)),
        str(passed_count),
        f"{total_pass_rate:.0%}",
    )

    console.print()
    console.print(table)

    console.print(
        "\n结果已写入："
        f"[cyan]{RESULT_PATH.as_posix()}[/cyan]"
    )


if __name__ == "__main__":
    main()
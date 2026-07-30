from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
import re


# 确保直接运行 scripts/run_mvp_evals.py 时可以导入 src 下的项目包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causalens.agent.graph import run_causalens_graph  # noqa: E402


DEFAULT_CASES_PATH = Path("evals/mvp_cases.json")
DEFAULT_OUTPUT_DIR = Path("outputs/evals")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description=(
            "运行 CausaLens MVP 端到端评测，"
            "并生成 JSON 明细和 Markdown 汇总报告。"
        )
    )

    parser.add_argument(
        "--cases-path",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=(
            "评测案例文件路径。"
            "默认：evals/mvp_cases.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "评测结果输出目录。"
            "默认：outputs/evals"
        ),
    )

    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help=(
            "只运行指定案例，例如："
            "mvp_01_data_profile。"
            "不填写时运行全部案例。"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="LangGraph 最大工具执行步数。默认：5",
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """
    将相对路径转换为相对于项目根目录的绝对路径。
    """

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_eval_suite(cases_path: Path) -> dict[str, Any]:
    """
    读取并进行基础验证。
    """

    if not cases_path.exists():
        raise FileNotFoundError(
            f"找不到评测案例文件：{cases_path}"
        )

    with cases_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        suite = json.load(file)

    if not isinstance(suite, dict):
        raise ValueError(
            "评测文件最外层必须是 JSON 对象。"
        )

    cases = suite.get("cases")

    if not isinstance(cases, list):
        raise ValueError(
            "评测文件必须包含 cases 列表。"
        )

    if not cases:
        raise ValueError(
            "评测案例列表不能为空。"
        )

    dataset_path = suite.get("dataset_path")

    if not isinstance(dataset_path, str):
        raise ValueError(
            "评测文件必须包含字符串类型的 dataset_path。"
        )

    return suite


def normalize_text(value: Any) -> str:
    """
    统一文本格式，用于稳定地进行关键词检查。
    """

    if value is None:
        return ""

    return str(value).strip().lower()


def extract_tool_name(tool_call: Any) -> str:
    """
    兼容字典和对象形式的工具调用记录。
    """

    if isinstance(tool_call, dict):
        tool_name = (
            tool_call.get("tool_name")
            or tool_call.get("name")
        )
    else:
        tool_name = (
            getattr(tool_call, "tool_name", None)
            or getattr(tool_call, "name", None)
        )

    if not isinstance(tool_name, str):
        return ""

    return tool_name


def extract_tool_order(
    final_state: dict[str, Any],
) -> list[str]:
    """
    从最终 AgentState 中提取实际工具调用顺序。
    """

    raw_tool_calls = final_state.get(
        "tool_calls",
        [],
    )

    if not isinstance(raw_tool_calls, list):
        return []

    tool_order: list[str] = []

    for tool_call in raw_tool_calls:
        tool_name = extract_tool_name(tool_call)

        if tool_name:
            tool_order.append(tool_name)

    return tool_order


def find_missing_terms(
    answer: str,
    required_terms: list[str],
) -> list[str]:
    """
    返回最终回答中没有出现的必需关键词。
    """

    normalized_answer = normalize_text(answer)
    missing_terms: list[str] = []

    for term in required_terms:
        normalized_term = normalize_text(term)

        if (
            normalized_term
            and normalized_term not in normalized_answer
        ):
            missing_terms.append(term)

    return missing_terms


def find_forbidden_terms(
    answer: str,
    forbidden_terms: list[str],
) -> list[str]:
    """
    返回最终回答中错误出现的禁止关键词。
    """

    normalized_answer = normalize_text(answer)
    found_terms: list[str] = []

    for term in forbidden_terms:
        normalized_term = normalize_text(term)

        if (
            normalized_term
            and normalized_term in normalized_answer
        ):
            found_terms.append(term)

    return found_terms


def run_eval_case(
    case: dict[str, Any],
    *,
    dataset_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    """
    运行单个端到端评测案例。
    """

    case_id = str(
        case.get("id", "unknown_case")
    )
    category = str(
        case.get("category", "unknown")
    )
    question = str(
        case.get("question", "")
    )

    expected_task_type = case.get(
        "expected_task_type"
    )
    expected_workflow_status = case.get(
        "expected_workflow_status"
    )
    expected_tool_order = case.get(
        "expected_tool_order",
        [],
    )
    required_terms = case.get(
        "answer_must_include",
        [],
    )
    expected_numbers = case.get(
        "answer_must_include_numbers",
        [],
    )
    forbidden_terms = case.get(
        "answer_must_not_include",
        [],
    )

    if not isinstance(expected_tool_order, list):
        raise ValueError(
            f"{case_id} 的 expected_tool_order 必须是列表。"
        )

    if not isinstance(required_terms, list):
        raise ValueError(
            f"{case_id} 的 answer_must_include 必须是列表。"
        )

    if not isinstance(expected_numbers, list):
        raise ValueError(
            f"{case_id} 的 "
            "answer_must_include_numbers 必须是列表。"
        )

    if not isinstance(forbidden_terms, list):
        raise ValueError(
            f"{case_id} 的 "
            "answer_must_not_include 必须是列表。"
        )

    # 检查数值期望配置是否合法
    for index, expectation in enumerate(
        expected_numbers
    ):
        if not isinstance(expectation, dict):
            raise ValueError(
                f"{case_id} 的 "
                "answer_must_include_numbers"
                f"[{index}] 必须是对象。"
            )

        if "value" not in expectation:
            raise ValueError(
                f"{case_id} 的 "
                "answer_must_include_numbers"
                f"[{index}] 缺少 value。"
            )

        try:
            float(expectation["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{case_id} 的 "
                "answer_must_include_numbers"
                f"[{index}].value 必须是数字。"
            ) from exc

        try:
            tolerance = float(
                expectation.get(
                    "tolerance",
                    0.01,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{case_id} 的 "
                "answer_must_include_numbers"
                f"[{index}].tolerance 必须是数字。"
            ) from exc

        if tolerance < 0:
            raise ValueError(
                f"{case_id} 的 "
                "answer_must_include_numbers"
                f"[{index}].tolerance 不能小于 0。"
            )

    started_at = perf_counter()

    try:
        final_state = run_causalens_graph(
            dataset_path=str(dataset_path),
            question=question,
            max_steps=max_steps,
        )

        duration_seconds = round(
            perf_counter() - started_at,
            4,
        )

        actual_task_type = final_state.get(
            "task_type"
        )
        actual_workflow_status = final_state.get(
            "workflow_status"
        )
        actual_tool_order = extract_tool_order(
            final_state
        )
        final_answer = str(
            final_state.get("final_answer") or ""
        )

        missing_terms = find_missing_terms(
            final_answer,
            required_terms,
        )
        missing_numbers = find_missing_numbers(
            final_answer,
            expected_numbers,
        )
        forbidden_terms_found = find_forbidden_terms(
            final_answer,
            forbidden_terms,
        )

        checks = {
            "task_type_matches": (
                actual_task_type
                == expected_task_type
            ),
            "workflow_status_matches": (
                actual_workflow_status
                == expected_workflow_status
            ),
            "tool_order_matches": (
                actual_tool_order
                == expected_tool_order
            ),
            "final_answer_not_empty": bool(
                final_answer.strip()
            ),
            "answer_contains_required_terms": (
                len(missing_terms) == 0
            ),
            "answer_contains_required_numbers": (
                len(missing_numbers) == 0
            ),
            "answer_avoids_forbidden_terms": (
                len(forbidden_terms_found) == 0
            ),
        }

        passed = all(checks.values())

        return {
            "id": case_id,
            "category": category,
            "question": question,
            "passed": passed,
            "duration_seconds": duration_seconds,
            "expected": {
                "task_type": expected_task_type,
                "workflow_status": (
                    expected_workflow_status
                ),
                "tool_order": expected_tool_order,
                "answer_must_include": (
                    required_terms
                ),
                "answer_must_include_numbers": (
                    expected_numbers
                ),
                "answer_must_not_include": (
                    forbidden_terms
                ),
            },
            "actual": {
                "task_type": actual_task_type,
                "workflow_status": (
                    actual_workflow_status
                ),
                "tool_order": actual_tool_order,
                "final_answer": final_answer,
                "step_count": final_state.get(
                    "step_count"
                ),
                "error": final_state.get(
                    "error"
                ),
            },
            "checks": checks,
            "missing_required_terms": (
                missing_terms
            ),
            "missing_required_numbers": (
                missing_numbers
            ),
            "forbidden_terms_found": (
                forbidden_terms_found
            ),
            "execution_error": None,
            "traceback": None,
        }

    except Exception as exc:
        duration_seconds = round(
            perf_counter() - started_at,
            4,
        )

        return {
            "id": case_id,
            "category": category,
            "question": question,
            "passed": False,
            "duration_seconds": duration_seconds,
            "expected": {
                "task_type": expected_task_type,
                "workflow_status": (
                    expected_workflow_status
                ),
                "tool_order": expected_tool_order,
                "answer_must_include": (
                    required_terms
                ),
                "answer_must_include_numbers": (
                    expected_numbers
                ),
                "answer_must_not_include": (
                    forbidden_terms
                ),
            },
            "actual": {
                "task_type": None,
                "workflow_status": None,
                "tool_order": [],
                "final_answer": "",
                "step_count": None,
                "error": None,
            },
            "checks": {
                "execution_succeeded": False,
            },
            "missing_required_terms": (
                required_terms
            ),
            "missing_required_numbers": (
                expected_numbers
            ),
            "forbidden_terms_found": [],
            "execution_error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "traceback": traceback.format_exc(),
        }

def select_cases(
    cases: list[dict[str, Any]],
    case_id: str | None,
) -> list[dict[str, Any]]:
    """
    根据 --case-id 筛选评测案例。
    """

    if case_id is None:
        return cases

    selected_cases = [
        case
        for case in cases
        if case.get("id") == case_id
    ]

    if selected_cases:
        return selected_cases

    available_ids = [
        str(case.get("id"))
        for case in cases
    ]

    raise ValueError(
        f"不存在案例：{case_id}\n"
        f"可用案例：{', '.join(available_ids)}"
    )


def print_case_result(
    result: dict[str, Any],
) -> None:
    """
    在控制台显示单个案例的简要结果。
    """

    status = (
        "PASS"
        if result["passed"]
        else "FAIL"
    )

    print(
        f"\n[{status}] "
        f"{result['id']} "
        f"({result['duration_seconds']}s)"
    )

    actual = result["actual"]

    print(
        "  task_type: "
        f"{actual.get('task_type')}"
    )
    print(
        "  workflow_status: "
        f"{actual.get('workflow_status')}"
    )
    print(
        "  tool_order: "
        f"{actual.get('tool_order')}"
    )

    if result["passed"]:
        return

    failed_checks = [
        check_name
        for check_name, check_passed
        in result["checks"].items()
        if not check_passed
    ]

    if failed_checks:
        print(
            "  failed_checks: "
            f"{failed_checks}"
        )

    if result["missing_required_terms"]:
        print(
            "  missing_required_terms: "
            f"{result['missing_required_terms']}"
        )

    if result["forbidden_terms_found"]:
        print(
            "  forbidden_terms_found: "
            f"{result['forbidden_terms_found']}"
        )

    if result["execution_error"]:
        print(
            "  execution_error: "
            f"{result['execution_error']}"
        )


def build_markdown_summary(
    report: dict[str, Any],
) -> str:
    """
    生成便于阅读和提交到仓库的 Markdown 汇总报告。
    """

    lines = [
        "# CausaLens MVP Evaluation Summary",
        "",
        f"- Suite: {report['suite_name']}",
        f"- Version: {report['version']}",
        f"- Generated at: {report['generated_at']}",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Total: {report['total_cases']}",
        f"- Passed: {report['passed_cases']}",
        f"- Failed: {report['failed_cases']}",
        f"- Pass rate: {report['pass_rate']}%",
        "",
        "## Case Results",
        "",
        (
            "| Case | Category | Result | "
            "Task type | Tool order | Duration |"
        ),
        (
            "|---|---|---:|---|---|---:|"
        ),
    ]

    for result in report["results"]:
        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        actual = result["actual"]

        tool_order = " → ".join(
            actual.get("tool_order") or []
        )

        if not tool_order:
            tool_order = "无"

        lines.append(
            f"| {result['id']} "
            f"| {result['category']} "
            f"| {status} "
            f"| {actual.get('task_type')} "
            f"| {tool_order} "
            f"| {result['duration_seconds']}s |"
        )

    failed_results = [
        result
        for result in report["results"]
        if not result["passed"]
    ]

    if failed_results:
        lines.extend(
            [
                "",
                "## Failed Case Details",
                "",
            ]
        )

        for result in failed_results:
            failed_checks = [
                check_name
                for check_name, check_passed
                in result["checks"].items()
                if not check_passed
            ]

            lines.extend(
                [
                    f"### {result['id']}",
                    "",
                    f"- Question: {result['question']}",
                    (
                        "- Failed checks: "
                        f"{', '.join(failed_checks) or '无'}"
                    ),
                    (
                        "- Missing required terms: "
                        f"{result['missing_required_terms'] or '无'}"
                    ),
                    (
                        "- Forbidden terms found: "
                        f"{result['forbidden_terms_found'] or '无'}"
                    ),
                    (
                        "- Execution error: "
                        f"{result['execution_error'] or '无'}"
                    ),
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def save_report(
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    保存 JSON 明细和 Markdown 汇总。
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / "latest_results.json"
    )
    summary_path = (
        output_dir
        / "latest_summary.md"
    )

    results_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path.write_text(
        build_markdown_summary(report),
        encoding="utf-8",
    )

    return results_path, summary_path


def main() -> int:
    """
    评测脚本入口。
    """

    args = parse_args()

    # 保证项目内部的相对路径从仓库根目录解析。
    os.chdir(PROJECT_ROOT)

    cases_path = resolve_project_path(
        args.cases_path
    )
    output_dir = resolve_project_path(
        args.output_dir
    )

    try:
        suite = load_eval_suite(
            cases_path
        )

        all_cases = suite["cases"]
        selected_cases = select_cases(
            all_cases,
            args.case_id,
        )

        dataset_path = resolve_project_path(
            Path(suite["dataset_path"])
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"找不到评测数据集：{dataset_path}"
            )

    except Exception as exc:
        print(
            f"[ERROR] 无法启动评测："
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    print(
        f"Suite: {suite.get('suite_name')}"
    )
    print(
        f"Dataset: {suite.get('dataset_path')}"
    )
    print(
        f"Cases: {len(selected_cases)}"
    )

    results: list[dict[str, Any]] = []

    for index, case in enumerate(
        selected_cases,
        start=1,
    ):
        case_id = case.get(
            "id",
            "unknown_case",
        )
        question = case.get(
            "question",
            "",
        )

        print(
            f"\n{'=' * 72}"
        )
        print(
            f"[{index}/{len(selected_cases)}] "
            f"{case_id}"
        )
        print(
            f"Question: {question}"
        )

        result = run_eval_case(
            case,
            dataset_path=dataset_path,
            max_steps=args.max_steps,
        )

        results.append(result)
        print_case_result(result)

    passed_cases = sum(
        result["passed"]
        for result in results
    )
    total_cases = len(results)
    failed_cases = (
        total_cases - passed_cases
    )

    pass_rate = round(
        (
            passed_cases
            / total_cases
            * 100
        )
        if total_cases
        else 0.0,
        2,
    )

    report = {
        "suite_name": suite.get(
            "suite_name"
        ),
        "version": suite.get(
            "version"
        ),
        "description": suite.get(
            "description"
        ),
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "cases_path": str(
            args.cases_path
        ),
        "dataset_path": suite.get(
            "dataset_path"
        ),
        "max_steps": args.max_steps,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": pass_rate,
        "results": results,
    }

    results_path, summary_path = save_report(
        report,
        output_dir,
    )

    print(
        f"\n{'=' * 72}"
    )
    print("MVP Evaluation Finished")
    print(f"Passed: {passed_cases}")
    print(f"Failed: {failed_cases}")
    print(f"Pass rate: {pass_rate}%")
    print(
        f"JSON report: {results_path}"
    )
    print(
        f"Markdown summary: {summary_path}"
    )

    return 0 if failed_cases == 0 else 1


def extract_numbers(text: str) -> list[float]:
    """
    从回答文本中提取整数和小数。
    """

    matches = re.findall(
        r"[-+]?(?:\d+\.\d+|\d+)",
        text,
    )

    numbers: list[float] = []

    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue

    return numbers


def find_missing_numbers(
    answer: str,
    expected_numbers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    使用数值容差判断回答中是否包含预期数字。
    """

    actual_numbers = extract_numbers(answer)
    missing_numbers: list[dict[str, Any]] = []

    for expectation in expected_numbers:
        expected_value = float(
            expectation["value"]
        )
        tolerance = float(
            expectation.get(
                "tolerance",
                0.01,
            )
        )

        matched = any(
            abs(actual - expected_value)
            <= tolerance
            for actual in actual_numbers
        )

        if not matched:
            missing_numbers.append(
                expectation
            )

    return missing_numbers


if __name__ == "__main__":
    raise SystemExit(main())
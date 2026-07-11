from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


CASES_PATH = Path("evals/week_03_rag_cases.jsonl")
RESULTS_PATH = Path("outputs/evals/week_03_rag_results.jsonl")
DATASET_PATH = "data/student_performance.csv"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))

    return cases


def run_cli_question(question: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())

    # Force UTF-8 in the child Python process.
    # This prevents Windows GBK / cp936 output from becoming mojibake in eval logs.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    command = [
        sys.executable,
        "src/causalens/cli.py",
        "ask",
        "--dataset-path",
        DATASET_PATH,
        "--question",
        question,
    ]

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def normalize_text(text: str) -> str:
    return "".join(text.lower().split())


def contains_all(text: str, expected_items: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return all(normalize_text(item) in normalized_text for item in expected_items)


def contains_any(text: str, expected_items: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return any(normalize_text(item) in normalized_text for item in expected_items)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    process = run_cli_question(case["question"])

    output = (process.stdout or "") + "\n" + (process.stderr or "")

    expected_tools = case.get("expected_tools", [])
    expected_any_tools = case.get("expected_any_tools", [])
    forbidden_tools = case.get("forbidden_tools", [])
    expected_sources = case.get("expected_sources", [])
    expected_answer_contains = case.get("expected_answer_contains", [])

    checks = {
        "returncode_ok": process.returncode == 0,
        "expected_tools_ok": contains_all(output, expected_tools),
        "expected_any_tools_ok": (
            True if not expected_any_tools else contains_any(output, expected_any_tools)
        ),
        "forbidden_tools_ok": not contains_any(output, forbidden_tools),
        "expected_sources_ok": contains_all(output, expected_sources),
        "expected_answer_contains_ok": contains_all(output, expected_answer_contains),
    }

    passed = all(checks.values())

    return {
        "case_id": case["case_id"],
        "type": case.get("type", ""),
        "question": case["question"],
        "passed": passed,
        "checks": checks,
        "expected_tools": expected_tools,
        "expected_any_tools": expected_any_tools,
        "forbidden_tools": forbidden_tools,
        "expected_sources": expected_sources,
        "expected_answer_contains": expected_answer_contains,
        "returncode": process.returncode,
        "output_preview": output[:2000],
    }


def main() -> None:
    cases = load_jsonl(CASES_PATH)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    print("=" * 80)
    print("Week 3 RAG Evaluation")
    print("=" * 80)

    for case in cases:
        result = evaluate_case(case)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} | {case['case_id']} | {case['question']}")

        if not result["passed"]:
            print("  Failed checks:")
            for check_name, ok in result["checks"].items():
                if not ok:
                    print(f"  - {check_name}")

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    passed_count = sum(1 for result in results if result["passed"])
    total_count = len(results)

    print("=" * 80)
    print(f"Passed: {passed_count} / {total_count}")
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
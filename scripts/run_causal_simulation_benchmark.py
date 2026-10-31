from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causalens.causal.benchmark import (  # noqa: E402
    SIMULATION_SCENARIOS,
    run_simulation_benchmark,
)


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Causal Estimator Simulation Benchmark",
        "",
        f"- Generated at: `{datetime.now().astimezone().isoformat()}`",
        f"- Replications per scenario: `{result['replications_per_scenario']}`",
        f"- Seed: `{result['seed']}`",
        f"- Passed scenarios: `{result['passed_scenarios']} / {result['scenario_count']}`",
        "",
        "| Scenario | Expectation | Success | Bias | RMSE | 95% coverage | Failure rate | Detection/block rate | Passed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["results"]:
        summary = item["summary"]
        lines.append(
            "| {name} | {expectation} | {success}/{total} | {bias} | {rmse} | {coverage} | {failure} | {detection} | {passed} |".format(
                name=item["scenario"]["name"],
                expectation=summary["expectation"],
                success=summary["successful_replications"],
                total=summary["replications"],
                bias=_number(summary["bias"]),
                rmse=_number(summary["rmse"]),
                coverage=_number(summary["coverage_95"]),
                failure=_number(summary["failure_rate"]),
                detection=_number(
                    summary["detection_rate"]
                    if summary["detection_rate"] is not None
                    else summary["expected_block_rate"]
                ),
                passed="PASS" if summary["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "> Accurate-estimation scenarios test recovery of known effects. Stress scenarios pass only when the expected bias, safety block, or diagnostic detection occurs.",
            "> Small replication counts are smoke tests, not evidence of calibrated coverage.",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic local causal estimator simulations."
    )
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=list(SIMULATION_SCENARIOS),
        help="Repeat to select scenarios; omit to run all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evals/causal_simulation"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_simulation_benchmark(
        scenario_names=args.scenario,
        replications=args.replications,
        seed=args.seed,
        temporary_root=output_dir,
    )
    json_path = output_dir / "benchmark.json"
    markdown_path = output_dir / "benchmark.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    print(_markdown_report(result))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

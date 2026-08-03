from pprint import pprint

from causalens.tools.causal_tool import (
    run_dml,
)


SYNTHETIC_DATASET = (
    "data/synthetic_causal_data.csv"
)

STUDENT_DATASET = (
    "data/student_performance.csv"
)


def print_case(
    title: str,
    response: dict,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    pprint(
        response,
        sort_dicts=False,
    )


if __name__ == "__main__":
    valid_result = run_dml(
        dataset_path=SYNTHETIC_DATASET,
        treatment="ad_exposure",
        outcome="retention_score",
        confounders=[
            "age",
            "log_income",
            "historical_activity",
            "channel_score",
        ],
        random_state=42,
        cv=5,
    )

    print_case(
        title="Valid LinearDML analysis",
        response=valid_result,
    )

    blocked_result = run_dml(
        dataset_path=STUDENT_DATASET,
        treatment="passed",
        outcome="score",
        confounders=[
            "study_hours",
            "attendance_rate",
        ],
        random_state=42,
        cv=5,
    )

    print_case(
        title=(
            "Blocked LinearDML: "
            "student dataset"
        ),
        response=blocked_result,
    )
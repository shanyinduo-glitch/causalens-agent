from pprint import pprint

from causalens.tools.causal_readiness import (
    check_causal_readiness,
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
    valid_result = check_causal_readiness(
        dataset_path=SYNTHETIC_DATASET,
        treatment="ad_exposure",
        outcome="retention_score",
        confounders=[
            "age",
            "log_income",
            "historical_activity",
            "channel_score",
        ],
    )

    print_case(
        "Valid synthetic causal configuration",
        valid_result,
    )

    no_confounders_result = (
        check_causal_readiness(
            dataset_path=SYNTHETIC_DATASET,
            treatment="ad_exposure",
            outcome="retention_score",
            confounders=[],
        )
    )

    print_case(
        "Blocked: no confounders",
        no_confounders_result,
    )

    missing_variable_result = (
        check_causal_readiness(
            dataset_path=SYNTHETIC_DATASET,
            treatment="unknown_treatment",
            outcome="retention_score",
            confounders=["age"],
        )
    )

    print_case(
        "Blocked: treatment does not exist",
        missing_variable_result,
    )

    small_data_result = (
        check_causal_readiness(
            dataset_path=STUDENT_DATASET,
            treatment="passed",
            outcome="score",
            confounders=[
                "study_hours",
                "attendance_rate",
            ],
        )
    )

    print_case(
        "Blocked: student dataset is too small",
        small_data_result,
    )
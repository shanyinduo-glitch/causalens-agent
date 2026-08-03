from pprint import pprint

from causalens.tools.regression_tool import (
    run_ols,
)


STUDENT_DATASET = (
    "data/student_performance.csv"
)

CAUSAL_DATASET = (
    "data/synthetic_causal_data.csv"
)


def print_case(
    title: str,
    result: dict,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    pprint(
        result,
        sort_dicts=False,
    )


if __name__ == "__main__":
    basic_regression = run_ols(
        dataset_path=STUDENT_DATASET,
        outcome="score",
        predictors=[
            "study_hours",
            "attendance_rate",
        ],
    )

    print_case(
        title=(
            "OLS: score ~ study_hours "
            "+ attendance_rate"
        ),
        result=basic_regression,
    )

    categorical_regression = run_ols(
        dataset_path=STUDENT_DATASET,
        outcome="score",
        predictors=[
            "study_hours",
            "attendance_rate",
            "major",
        ],
        categorical_columns=[
            "major",
        ],
    )

    print_case(
        title=(
            "OLS: score ~ study_hours "
            "+ attendance_rate + major"
        ),
        result=categorical_regression,
    )

    synthetic_regression = run_ols(
        dataset_path=CAUSAL_DATASET,
        outcome="retention_score",
        predictors=[
            "ad_exposure",
            "age",
            "log_income",
            "historical_activity",
            "channel_score",
        ],
    )

    print_case(
        title=(
            "OLS: retention_score ~ "
            "ad_exposure + observed covariates"
        ),
        result=synthetic_regression,
    )
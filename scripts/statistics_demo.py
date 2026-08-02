from pprint import pprint

from causalens.tools.statistical_tool import (
    run_correlation,
    run_group_comparison,
)


DATASET_PATH = "data/student_performance.csv"


def print_case(
    title: str,
    result: dict,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    pprint(result, sort_dicts=False)


if __name__ == "__main__":
    pearson_result = run_correlation(
        dataset_path=DATASET_PATH,
        x_column="study_hours",
        y_column="score",
        method="pearson",
    )

    print_case(
        title="Pearson: study_hours vs score",
        result=pearson_result,
    )

    spearman_result = run_correlation(
        dataset_path=DATASET_PATH,
        x_column="study_hours",
        y_column="score",
        method="spearman",
    )

    print_case(
        title="Spearman: study_hours vs score",
        result=spearman_result,
    )

    passed_comparison = run_group_comparison(
        dataset_path=DATASET_PATH,
        group_column="passed",
        value_column="score",
    )

    print_case(
        title="Welch t-test: score by passed",
        result=passed_comparison,
    )

    major_comparison = run_group_comparison(
        dataset_path=DATASET_PATH,
        group_column="major",
        value_column="score",
    )

    print_case(
        title="One-way ANOVA: score by major",
        result=major_comparison,
    )
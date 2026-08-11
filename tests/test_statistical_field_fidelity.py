from causalens.agent.nodes import (
    _check_statistical_field_fidelity,
    _extract_explicit_dataset_columns,
)


DATASET_PATH = "data/student_performance.csv"


def test_extract_explicit_dataset_columns():
    """
    用户明确写出 major 和 city 时，
    应从原始 CSV schema 中识别这两个字段。
    """

    columns = _extract_explicit_dataset_columns(
        question=(
            "请计算 major 和 city 的 "
            "Pearson 相关系数。"
        ),
        dataset_path=DATASET_PATH,
    )

    assert set(columns) == {
        "major",
        "city",
    }


def test_rejects_silent_field_substitution():
    """
    用户要求 major + city，
    但模型偷偷生成 study_hours + score 时，
    必须拒绝。
    """

    error = _check_statistical_field_fidelity(
        question=(
            "请计算 major 和 city 的 "
            "Pearson 相关系数。"
        ),
        dataset_path=DATASET_PATH,
        tool_name="run_correlation",
        arguments={
            "x_column": "study_hours",
            "y_column": "score",
            "method": "pearson",
            "dataset_path": DATASET_PATH,
        },
    )

    assert error is not None

    assert (
        "统计工具参数与用户明确指定字段不一致"
        in error
    )

    assert "major" in error
    assert "city" in error
    assert "study_hours" in error
    assert "score" in error


def test_allows_matching_correlation_fields():
    """
    用户要求 major + city，
    模型也生成 major + city 时，
    忠实性检查本身应通过。

    后续是否能做 Pearson，
    交给原有数据类型 Validator 判断。
    """

    error = _check_statistical_field_fidelity(
        question=(
            "请计算 major 和 city 的 "
            "Pearson 相关系数。"
        ),
        dataset_path=DATASET_PATH,
        tool_name="run_correlation",
        arguments={
            "x_column": "major",
            "y_column": "city",
            "method": "pearson",
            "dataset_path": DATASET_PATH,
        },
    )

    assert error is None
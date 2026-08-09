from pathlib import Path

import causalens.tools.chart_tool as chart_tool


def create_test_csv(tmp_path):
    """
    创建一个专供图表测试使用的临时 CSV。

    pytest 测试结束后，tmp_path 目录会自动清理。
    """
    csv_path = tmp_path / "students.csv"

    csv_path.write_text(
        """student_id,major,score,study_hours
1,math,88,12
2,cs,92,15
3,stats,76,8
4,math,59,4
5,cs,81,10
""",
        encoding="utf-8",
    )

    return csv_path


def create_time_series_csv(tmp_path):
    """
    创建一个包含日期字段的临时 CSV，
    专门用于测试折线图。
    """
    csv_path = tmp_path / "time_series_students.csv"

    csv_path.write_text(
        """student_id,exam_date,score
1,2026-03-01,88
2,2026-03-01,92
3,2026-03-15,76
4,2026-03-15,59
5,2026-04-01,81
""",
        encoding="utf-8",
    )

    return csv_path

def test_create_bar_chart_with_mean_aggregation(tmp_path, monkeypatch):
    """
    验证：
    按 major 分组，对 score 求平均值时，
    能生成真实 PNG，并返回正确的汇总数据。
    """
    csv_path = create_test_csv(tmp_path)

    # 测试时把图片输出到 pytest 临时目录，
    # 避免污染项目真实的 outputs/charts 文件夹。
    output_dir = tmp_path / "charts"

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        output_dir,
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="bar",
        x_column="major",
        y_column="score",
        aggregation="mean",
        title="Average Score by Major",
    )

    assert result.success is True
    assert result.tool_name == "create_chart"

    assert result.result["chart_type"] == "bar"
    assert result.result["x_column"] == "major"
    assert result.result["y_column"] == "score"
    assert result.result["aggregation"] == "mean"
    assert result.result["value_column"] == "mean_score"
    assert result.result["numeric_summary"] is None
    assert result.result["relationship_summary"] is None

    # 确认 PNG 文件真的生成了。
    chart_path = Path(result.result["chart_path"])

    assert chart_path.exists()
    assert chart_path.suffix == ".png"

    # 原始数据中：
    # cs: (92 + 81) / 2 = 86.5
    # stats: 76
    # math: (88 + 59) / 2 = 73.5
    # 工具按均值从高到低排序。
    preview = result.result["data_preview"]

    assert preview[0] == {
        "major": "cs",
        "mean_score": 86.5,
    }

    assert preview[1] == {
        "major": "stats",
        "mean_score": 76.0,
    }

    assert preview[2] == {
        "major": "math",
        "mean_score": 73.5,
    }


def test_create_bar_chart_with_count_aggregation(tmp_path, monkeypatch):
    """
    验证：
    aggregation=count 时，不需要 y_column，
    工具应按类别统计记录数量。
    """
    csv_path = create_test_csv(tmp_path)
    output_dir = tmp_path / "charts"

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        output_dir,
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="bar",
        x_column="major",
        aggregation="count",
        title="Student Count by Major",
    )

    assert result.success is True
    assert result.result["value_column"] == "count"
    assert result.result["numeric_summary"] is None
    assert result.result["relationship_summary"] is None

    preview = result.result["data_preview"]

    assert preview[0]["major"] == "cs"
    assert preview[0]["count"] == 2

    assert preview[1]["major"] == "math"
    assert preview[1]["count"] == 2

    assert preview[2]["major"] == "stats"
    assert preview[2]["count"] == 1


def test_create_chart_returns_error_when_x_column_does_not_exist(
    tmp_path,
    monkeypatch,
):
    """
    验证：
    x_column 不存在时，工具应返回 ToolResult 失败，
    而不是抛出 Traceback。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="bar",
        x_column="not_a_real_column",
        y_column="score",
        aggregation="mean",
    )

    assert result.success is False
    assert "Column does not exist" in result.error
    assert "not_a_real_column" in result.error


def test_create_bar_chart_rejects_non_numeric_y_column(
    tmp_path,
    monkeypatch,
):
    """
    验证：
    平均值或求和柱状图的 y_column 必须是数值列。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="bar",
        x_column="major",
        y_column="major",
        aggregation="mean",
    )

    assert result.success is False
    assert "必须是数值列" in result.error


def test_create_scatter_chart(tmp_path, monkeypatch):
    """
    验证：
    散点图能基于两个数值列生成 PNG，
    并返回实际参与绘图的数据点。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="scatter",
        x_column="study_hours",
        y_column="score",
        title="Study Hours vs Score",
    )

    assert result.success is True
    assert result.tool_name == "create_chart"

    assert result.result["chart_type"] == "scatter"
    assert result.result["x_column"] == "study_hours"
    assert result.result["y_column"] == "score"
    assert result.result["aggregation"] is None
    assert result.result["value_column"] is None
    summary = result.result["relationship_summary"]

    assert summary["point_count"] == 5
    assert summary["x_min"] == 4.0
    assert summary["x_max"] == 15.0
    assert summary["y_min"] == 59.0
    assert summary["y_max"] == 92.0

    # 这份测试数据中的学习时长和成绩呈很强的正线性相关。
    assert summary["pearson_correlation"] == 0.9797

    chart_path = Path(result.result["chart_path"])

    assert chart_path.exists()
    assert chart_path.suffix == ".png"

    # 测试 CSV 有 5 行，且 study_hours 和 score 都没有缺失。
    assert result.result["plotted_row_count"] == 5

    preview = result.result["data_preview"]

    assert len(preview) == 5
    assert {"study_hours": 12, "score": 88} in preview
    assert {"study_hours": 15, "score": 92} in preview


def test_create_scatter_chart_rejects_non_numeric_x_column(
    tmp_path,
    monkeypatch,
):
    """
    散点图横轴必须是数值列。
    major 是类别列，因此应被拒绝。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="scatter",
        x_column="major",
        y_column="score",
    )

    assert result.success is False
    assert "x_column=major" in result.error
    assert "必须是数值列" in result.error


def test_create_scatter_chart_requires_y_column(
    tmp_path,
    monkeypatch,
):
    """
    散点图必须指定两个数值列。
    只给 x_column 不足以画二维点。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="scatter",
        x_column="study_hours",
    )

    assert result.success is False
    assert "必须提供 y_column" in result.error


def test_create_histogram_chart(tmp_path, monkeypatch):
    """
    验证：
    直方图可以对一个数值列生成 PNG，
    并返回真实的基础统计摘要。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="histogram",
        x_column="score",
        title="Score Distribution",
    )

    assert result.success is True
    assert result.tool_name == "create_chart"

    assert result.result["chart_type"] == "histogram"
    assert result.result["x_column"] == "score"
    assert result.result["y_column"] is None
    assert result.result["aggregation"] is None
    assert result.result["value_column"] is None
    assert result.result["relationship_summary"] is None

    chart_path = Path(result.result["chart_path"])

    assert chart_path.exists()
    assert chart_path.suffix == ".png"

    # 测试 CSV 中 score 有 5 条非空数据。
    assert result.result["plotted_row_count"] == 5

    summary = result.result["numeric_summary"]

    assert summary["count"] == 5
    assert summary["mean"] == 79.2
    assert summary["median"] == 81.0
    assert summary["min"] == 59.0
    assert summary["max"] == 92.0

    preview = result.result["data_preview"]

    assert len(preview) == 5
    assert {"score": 88} in preview
    assert {"score": 92} in preview


def test_create_histogram_rejects_non_numeric_x_column(
    tmp_path,
    monkeypatch,
):
    """
    直方图只能展示数值变量的分布。
    major 是类别列，因此应被拒绝。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="histogram",
        x_column="major",
    )

    assert result.success is False
    assert "x_column=major" in result.error
    assert "必须是数值列" in result.error


def test_create_line_chart_with_mean_aggregation(
    tmp_path,
    monkeypatch,
):
    """
    验证：
    折线图能按日期分组，对 score 求平均值，
    并按时间从早到晚返回真实汇总结果。
    """
    csv_path = create_time_series_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="line",
        x_column="exam_date",
        y_column="score",
        aggregation="mean",
        title="Average Score by Exam Date",
    )

    assert result.success is True
    assert result.tool_name == "create_chart"

    assert result.result["chart_type"] == "line"
    assert result.result["x_column"] == "exam_date"
    assert result.result["y_column"] == "score"
    assert result.result["aggregation"] == "mean"
    assert result.result["value_column"] == "mean_score"
    assert result.result["numeric_summary"] is None
    assert result.result["relationship_summary"] is None

    chart_path = Path(result.result["chart_path"])

    assert chart_path.exists()
    assert chart_path.suffix == ".png"

    # 3 个日期，因此折线图有 3 个汇总点。
    assert result.result["plotted_row_count"] == 3

    assert result.result["data_preview"] == [
        {
            "exam_date": "2026-03-01",
            "mean_score": 90.0,
        },
        {
            "exam_date": "2026-03-15",
            "mean_score": 67.5,
        },
        {
            "exam_date": "2026-04-01",
            "mean_score": 81.0,
        },
    ]


def test_create_line_chart_requires_aggregation(
    tmp_path,
    monkeypatch,
):
    """
    折线图必须知道按日期计算什么：
    mean、sum 或 count。
    """
    csv_path = create_time_series_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="line",
        x_column="exam_date",
        y_column="score",
    )

    assert result.success is False
    assert "需要指定 aggregation" in result.error


def test_create_line_chart_rejects_non_date_x_column(
    tmp_path,
    monkeypatch,
):
    """
    折线图横轴必须是可解析的日期列。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="line",
        x_column="major",
        y_column="score",
        aggregation="mean",
    )

    assert result.success is False
    assert "x_column=major" in result.error
    assert "可解析的日期列" in result.error


def test_create_chart_rejects_unknown_chart_type(
    tmp_path,
    monkeypatch,
):
    """
    ChartRequest 只允许项目定义的四种图表类型。
    pie 不在允许范围内，应被 Pydantic 拒绝。
    """
    csv_path = create_test_csv(tmp_path)

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="pie",
        x_column="major",
    )

    assert result.success is False
    assert "chart_type" in result.error

def test_create_bar_count_rejects_all_missing_x_column(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "all_missing_category.csv"

    csv_path.write_text(
        "category,score\n"
        ",88\n"
        ",92\n"
        ",76\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        chart_tool,
        "CHART_OUTPUT_DIR",
        tmp_path / "charts",
    )

    result = chart_tool.create_chart(
        dataset_path=str(csv_path),
        chart_type="bar",
        x_column="category",
        aggregation="count",
    )

    assert result.success is False
    assert "数据为空" in result.error
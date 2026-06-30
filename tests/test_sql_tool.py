from causalens.tools.sql_tool import run_readonly_sql


def create_test_csv(tmp_path):
    """
    创建一个临时 CSV，供每个测试独立使用。

    tmp_path 是 pytest 提供的临时目录：
    测试结束后会自动清理，不会污染 data/ 文件夹。
    """
    csv_path = tmp_path / "students.csv"

    csv_path.write_text(
        """student_id,major,score,study_hours,passed
1,math,88,12,1
2,cs,92,15,1
3,stats,76,8,1
4,math,59,4,0
5,cs,81,10,1
""",
        encoding="utf-8",
    )

    return csv_path


def test_run_readonly_sql_count_rows(tmp_path):
    csv_path = create_test_csv(tmp_path)

    result = run_readonly_sql(
        dataset_path=str(csv_path),
        query="SELECT COUNT(*) AS total_rows FROM dataset",
    )

    assert result.success is True
    assert result.tool_name == "run_readonly_sql"
    assert result.result["row_count"] == 1
    assert result.result["columns"] == ["total_rows"]
    assert result.result["rows"] == [{"total_rows": 5}]


def test_run_readonly_sql_group_by_average(tmp_path):
    csv_path = create_test_csv(tmp_path)

    result = run_readonly_sql(
        dataset_path=str(csv_path),
        query="""
        SELECT major, AVG(score) AS avg_score
        FROM dataset
        GROUP BY major
        ORDER BY avg_score DESC
        """,
    )

    assert result.success is True
    assert result.result["row_count"] == 3
    assert result.result["columns"] == ["major", "avg_score"]

    rows = result.result["rows"]

    # cs 的平均分是 (92 + 81) / 2 = 86.5，应排第一
    assert rows[0]["major"] == "cs"
    assert rows[0]["avg_score"] == 86.5


def test_run_readonly_sql_top_student(tmp_path):
    csv_path = create_test_csv(tmp_path)

    result = run_readonly_sql(
        dataset_path=str(csv_path),
        query="""
        SELECT student_id, score
        FROM dataset
        ORDER BY score DESC
        LIMIT 1
        """,
    )

    assert result.success is True
    assert result.result["row_count"] == 1
    assert result.result["rows"] == [
        {
            "student_id": 2,
            "score": 92,
        }
    ]


def test_run_readonly_sql_file_not_found():
    result = run_readonly_sql(
        dataset_path="data/not_exists.csv",
        query="SELECT * FROM dataset",
    )

    assert result.success is False
    assert "does not exist" in result.error


def test_run_readonly_sql_rejects_non_readonly_query(tmp_path):
    csv_path = create_test_csv(tmp_path)

    result = run_readonly_sql(
        dataset_path=str(csv_path),
        query="DELETE FROM dataset",
    )

    assert result.success is False
    assert "Only read-only SQL is allowed" in result.error


def test_run_readonly_sql_rejects_multiple_statements(tmp_path):
    csv_path = create_test_csv(tmp_path)

    result = run_readonly_sql(
        dataset_path=str(csv_path),
        query="""
        SELECT * FROM dataset;
        DROP TABLE dataset;
        """,
    )

    assert result.success is False
    assert "Only one SQL statement is allowed" in result.error


def test_run_readonly_sql_handles_invalid_column(tmp_path):
    csv_path = create_test_csv(tmp_path)

    result = run_readonly_sql(
        dataset_path=str(csv_path),
        query="SELECT not_a_real_column FROM dataset",
    )

    assert result.success is False
    assert result.error is not None

def test_run_readonly_sql_marks_truncated_results(tmp_path):
    csv_path = tmp_path / "many_students.csv"

    rows = ["student_id,score"]

    for student_id in range(1, 151):
        rows.append(f"{student_id},{student_id}")

    csv_path.write_text(
        "\n".join(rows),
        encoding="utf-8",
    )

    result = run_readonly_sql(
        dataset_path=str(csv_path),
        query="SELECT * FROM dataset ORDER BY student_id",
    )

    assert result.success is True
    assert result.result["row_count"] == 150
    assert result.result["returned_row_count"] == 100
    assert result.result["truncated"] is True
    assert len(result.result["rows"]) == 100


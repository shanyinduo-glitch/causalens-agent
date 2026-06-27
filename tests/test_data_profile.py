from causalens.tools.data_profile import profile_dataset


def test_profile_dataset_returns_profile_for_valid_csv(tmp_path):
    """正常 CSV 应返回正确的数据概况。"""

    csv_path = tmp_path / "students.csv"

    csv_path.write_text(
        "student_id,major,score\n"
        "1,math,88\n"
        "2,cs,92\n",
        encoding="utf-8",
    )

    result = profile_dataset(str(csv_path))

    assert result.success is True
    assert result.error is None
    assert result.result is not None

    assert result.result["rows"] == 2
    assert result.result["columns"] == 3
    assert result.result["column_names"] == [
        "student_id",
        "major",
        "score",
    ]
    assert result.result["missing_values"]["score"] == 0
    assert result.result["numeric_summary"]["score"]["mean"] == 90.0


def test_profile_dataset_returns_friendly_error_for_missing_file(tmp_path):
    """不存在的文件应返回清楚错误，而不是抛出异常。"""

    missing_path = tmp_path / "not_exists.csv"

    result = profile_dataset(str(missing_path))

    assert result.success is False
    assert result.result is None
    assert result.error is not None
    assert "找不到数据文件" in result.error


def test_profile_dataset_returns_friendly_error_for_empty_file(tmp_path):
    """空 CSV 文件应返回清楚错误。"""

    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    result = profile_dataset(str(empty_path))

    assert result.success is False
    assert result.result is None
    assert result.error is not None
    assert "CSV 文件为空" in result.error


def test_profile_dataset_rejects_csv_with_header_only(tmp_path):
    """只有表头、没有数据行的 CSV 不应被当成有效数据集。"""

    header_only_path = tmp_path / "header_only.csv"
    header_only_path.write_text(
        "name,score\n",
        encoding="utf-8",
    )

    result = profile_dataset(str(header_only_path))

    assert result.success is False
    assert result.result is None
    assert result.error is not None
    assert "没有数据行" in result.error


def test_profile_dataset_handles_text_only_csv(tmp_path):
    """没有数值列的 CSV 仍应能生成概况，只是数值统计为空。"""

    text_only_path = tmp_path / "text_only.csv"
    text_only_path.write_text(
        "name,major\n"
        "Alice,math\n"
        "Bob,cs\n",
        encoding="utf-8",
    )

    result = profile_dataset(str(text_only_path))

    assert result.success is True
    assert result.result is not None
    assert result.result["rows"] == 2
    assert result.result["numeric_summary"] == {}
from pathlib import Path
import re

import duckdb
import pandas as pd

from causalens.schemas import ToolResult

def _strip_sql_literals_and_comments(query: str) -> str:
    """
    去掉 SQL 中的字符串、引用标识符和注释，
    避免安全关键字检查产生明显误判。
    """
    pattern = re.compile(
        r"""
        '(?:''|[^'])*'          # 单引号字符串
        |
        "(?:""|[^"])*"          # 双引号标识符
        |
        --[^\r\n]*              # 单行注释
        |
        /\*.*?\*/               # 块注释
        """,
        re.VERBOSE | re.DOTALL,
    )

    return pattern.sub(" ", query)

def _validate_readonly_query(query: str) -> str | None:
    """
    对 SQL 做一个最基础的安全检查。

    当前只允许查询类语句：
    SELECT、WITH、EXPLAIN。

    返回：
    - None：说明通过检查
    - str：说明有错误，内容是错误信息
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        return "SQL query cannot be empty."

    sanitized_query = _strip_sql_literals_and_comments(
        cleaned_query
    ).strip()

    if not sanitized_query:
        return "SQL query cannot be empty."

    # 允许最后一个分号，但不允许多条 SQL。
    if sanitized_query.endswith(";"):
        sanitized_query = (
            sanitized_query[:-1].strip()
        )

    if ";" in sanitized_query:
        return "Only one SQL statement is allowed."

    first_keyword = (
        sanitized_query
        .split(maxsplit=1)[0]
        .upper()
    )

    allowed_keywords = {
        "SELECT",
        "WITH",
        "EXPLAIN",
    }

    if first_keyword not in allowed_keywords:
        return (
            "Only read-only SQL is allowed. "
            "Please use SELECT, WITH, or EXPLAIN."
        )

    forbidden_keywords = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "REPLACE",
        "MERGE",
        "COPY",
        "ATTACH",
        "DETACH",
        "INSTALL",
        "LOAD",
        "EXPORT",
        "IMPORT",
        "CALL",
        "SET",
        "PRAGMA",
        "VACUUM",
    }

    sql_tokens = set(
        re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            sanitized_query.upper(),
        )
    )

    if sql_tokens & forbidden_keywords:
        return (
            "Only read-only SQL is allowed. "
            "Please use SELECT, WITH, or EXPLAIN."
        )

    return None


def run_readonly_sql(dataset_path: str, query: str) -> ToolResult:
    """
    读取 CSV，将其临时注册为 dataset 表，并执行只读 SQL 查询。

    参数：
    - dataset_path: CSV 文件路径
    - query: 查询 SQL。CSV 在 SQL 中统一叫 dataset。

    返回：
    - 成功：查询结果的行数、列名和前 100 行数据
    - 失败：统一的错误信息
    """
    connection = None

    try:
        file_path = Path(dataset_path)

        if not file_path.exists():
            return ToolResult(
                tool_name="run_readonly_sql",
                success=False,
                error=f"Dataset does not exist: {dataset_path}",
            )

        if not file_path.is_file():
            return ToolResult(
                tool_name="run_readonly_sql",
                success=False,
                error=f"Dataset path is not a file: {dataset_path}",
            )

        validation_error = _validate_readonly_query(query)

        if validation_error:
            return ToolResult(
                tool_name="run_readonly_sql",
                success=False,
                error=validation_error,
            )

        df = pd.read_csv(file_path)

        connection = duckdb.connect(database=":memory:")

        # 把 pandas DataFrame 临时注册为 SQL 表。
        # 后续 SQL 永远使用 dataset，不依赖真实 CSV 文件名。
        connection.register("dataset", df)

        result_df = connection.execute(query).fetchdf()

        max_rows_to_return = 100
        returned_df = result_df.head(max_rows_to_return)

        return ToolResult(
            tool_name="run_readonly_sql",
            success=True,
            result={
                # SQL 实际查询一共得到多少行
                "row_count": len(result_df),

                # 实际返回给 Agent / 模型多少行
                "returned_row_count": len(returned_df),

                # True 表示查询结果太多，只返回了前 100 行
                "truncated": len(result_df) > max_rows_to_return,

                "columns": result_df.columns.tolist(),
                "rows": returned_df.to_dict(orient="records"),
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="run_readonly_sql",
            success=False,
            error=str(exc),
        )

    finally:
        if connection is not None:
            connection.close()
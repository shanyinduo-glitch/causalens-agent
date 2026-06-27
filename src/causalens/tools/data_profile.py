from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from causalens.schemas import DatasetProfile, ToolResult


def profile_dataset(dataset_path: str) -> ToolResult:
    """读取 CSV 文件，并返回数据集的基础概况。"""

    path = Path(dataset_path)

    if not path.exists():
        return ToolResult(
            tool_name="profile_dataset",
            success=False,
            error=f"找不到数据文件：{dataset_path}",
        )

    if not path.is_file():
        return ToolResult(
            tool_name="profile_dataset",
            success=False,
            error=f"数据路径不是文件：{dataset_path}",
        )

    try:
        df = pd.read_csv(path)

    except EmptyDataError:
        return ToolResult(
            tool_name="profile_dataset",
            success=False,
            error=f"CSV 文件为空，无法分析：{dataset_path}",
        )

    except UnicodeDecodeError:
        return ToolResult(
            tool_name="profile_dataset",
            success=False,
            error=(
                f"无法使用 UTF-8 读取 CSV 文件：{dataset_path}。"
                "该文件可能使用了其他编码。"
            ),
        )

    except (ParserError, ValueError) as exc:
        return ToolResult(
            tool_name="profile_dataset",
            success=False,
            error=f"CSV 文件格式无法解析：{exc}",
        )

    except OSError as exc:
        return ToolResult(
            tool_name="profile_dataset",
            success=False,
            error=f"读取文件失败：{exc}",
        )

    if df.empty:
        return ToolResult(
            tool_name="profile_dataset",
            success=False,
            error=f"CSV 文件没有数据行，无法生成统计概况：{dataset_path}",
        )

    try:
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            summary = {}
        else:
            summary = numeric_df.describe().to_dict()

        profile = DatasetProfile(
            rows=df.shape[0],
            columns=df.shape[1],
            column_names=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            missing_values=df.isna().sum().to_dict(),
            numeric_summary=summary,
        )

        return ToolResult(
            tool_name="profile_dataset",
            success=True,
            result=profile.model_dump(),
        )

    except Exception as exc:
        return ToolResult(
            tool_name="profile_dataset",
            success=False,
            error=f"生成数据集概况时发生错误：{exc}",
        )
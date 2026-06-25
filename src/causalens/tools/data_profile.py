import pandas as pd

from causalens.schemas import DatasetProfile, ToolResult


def profile_dataset(dataset_path: str) -> ToolResult:
    """读取 CSV 文件，并返回数据集的基础概况。"""

    try:
        df = pd.read_csv(dataset_path)

        numeric_df = df.select_dtypes(include="number")
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
            error=str(exc),
        )
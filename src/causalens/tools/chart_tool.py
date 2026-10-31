from datetime import datetime
from pathlib import Path
import re

import matplotlib

# 必须放在导入 pyplot 之前。
# Agg 是无界面后端，适合服务器、CLI、pytest 中生成 PNG 文件。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from pandas.api.types import is_numeric_dtype

from causalens.schemas import ChartRequest, ToolResult

# Prefer a CJK font when available so the customer-facing chart does not show
# missing-glyph boxes. DejaVu remains the portable fallback for CI/Linux.
_available_fonts = {font.name for font in font_manager.fontManager.ttflist}
for _font in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
    if _font in _available_fonts:
        plt.rcParams['font.sans-serif'] = [_font, 'DejaVu Sans']
        break
plt.rcParams['axes.unicode_minus'] = False


CHART_OUTPUT_DIR = Path("outputs/charts")
MAX_DATA_PREVIEW_ROWS = 20

DISPLAY_LABELS = {
    't_diff': 'TIPS 实际收益率变化',
    'y_ret': '黄金对数收益率',
    'dollar_ret': '美元指数变化',
    'sp500_ret': '标普500收益率',
    'fed_funds_diff': '联邦基金利率变化',
    'vix_level': 'VIX 波动率指数',
}


def display_label(column: str) -> str:
    """Use a stable human label while retaining the raw field in tool output."""
    return DISPLAY_LABELS.get(column, column)


def _build_safe_filename(text: str) -> str:
    """
    把字段名、标题等文本转换成适合做文件名的字符串。

    例如：
    "Average Score by Major"
    → "average_score_by_major"
    """
    normalized = text.strip().lower()
    safe_name = re.sub(r"[^a-z0-9_-]+", "_", normalized)
    safe_name = safe_name.strip("_")

    return safe_name or "chart"


def _build_bar_summary(
    df: pd.DataFrame,
    request: ChartRequest,
) -> tuple[pd.DataFrame, str]:
    """
    根据 bar 图参数，生成实际用于绘图的汇总数据。

    返回：
    - summary_df：例如 major 和 avg_score 两列
    - value_column：汇总数值所在列名
    """
    x_column = request.x_column

    if request.aggregation is None:
        raise ValueError(
            "柱状图需要指定 aggregation。"
            "例如 mean、sum 或 count。"
        )

    if request.aggregation == "count":
        summary_df = (
            df.dropna(subset=[x_column])
            .groupby(x_column)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        if summary_df.empty:
            raise ValueError(
                "用于生成柱状图的数据为空。"
                "请检查分类列是否全部为缺失值。"
            )

        return summary_df, "count"

    if request.y_column is None:
        raise ValueError(
            f"aggregation={request.aggregation} 时，"
            "必须提供 y_column。"
        )

    y_column = request.y_column

    if not is_numeric_dtype(df[y_column]):
        raise ValueError(
            f"无法生成柱状图：y_column={y_column} "
            "必须是数值列。"
        )

    clean_df = df[[x_column, y_column]].dropna()

    if clean_df.empty:
        raise ValueError(
            "用于生成柱状图的数据为空。"
            "请检查所选列是否全部为缺失值。"
        )

    aggregation_map = {
        "mean": "mean",
        "sum": "sum",
    }

    value_column = f"{request.aggregation}_{y_column}"

    summary_df = (
        clean_df.groupby(x_column)[y_column]
        .agg(aggregation_map[request.aggregation])
        .reset_index(name=value_column)
        .sort_values(value_column, ascending=False)
    )

    return summary_df, value_column


def _create_bar_chart(
    df: pd.DataFrame,
    request: ChartRequest,
) -> tuple[Path, pd.DataFrame, str]:
    """
    生成柱状图，并返回：
    - 图片路径
    - 绘图使用的汇总数据
    - 汇总数值列名
    """
    summary_df, value_column = _build_bar_summary(df, request)

    CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chart_title = request.title or (
        f"{request.aggregation} {request.y_column or 'records'} "
        f"by {request.x_column}"
    )

    filename_prefix = _build_safe_filename(chart_title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    chart_path = CHART_OUTPUT_DIR / f"{filename_prefix}_{timestamp}.png"

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary_df[request.x_column].astype(str),
        summary_df[value_column],
    )

    plt.xlabel(request.x_column)
    plt.ylabel(value_column)
    plt.title(chart_title)

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    return chart_path, summary_df, value_column


def _create_scatter_chart(
    df: pd.DataFrame,
    request: ChartRequest,
) -> tuple[Path, pd.DataFrame, dict[str, int | float | None]]:
    """
    生成散点图。

    散点图要求：
    - x_column 必须是数值列；
    - y_column 必须存在且是数值列；
    - 缺失值行不会参与绘图。

    返回：
    - 图片路径；
    - 实际用于绘图的二维数据。
    """
    x_column = request.x_column
    y_column = request.y_column

    if y_column is None:
        raise ValueError(
            "生成散点图时，必须提供 y_column。"
        )

    if not is_numeric_dtype(df[x_column]):
        raise ValueError(
            f"无法生成散点图：x_column={x_column} "
            "必须是数值列。"
        )

    if not is_numeric_dtype(df[y_column]):
        raise ValueError(
            f"无法生成散点图：y_column={y_column} "
            "必须是数值列。"
        )

    plot_df = df[[x_column, y_column]].dropna().copy()

    if plot_df.empty:
        raise ValueError(
            "用于生成散点图的数据为空。"
            "请检查所选列是否全部为缺失值。"
        )

    CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chart_title = request.title or f"{display_label(y_column)}与{display_label(x_column)}的关系"

    filename_prefix = _build_safe_filename(chart_title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    chart_path = CHART_OUTPUT_DIR / f"{filename_prefix}_{timestamp}.png"

    plt.figure(figsize=(8, 5))

    plt.scatter(
        plot_df[x_column],
        plot_df[y_column],
        s=14,
        alpha=0.35,
        edgecolors='none',
        color='#2f75b5',
    )

    if len(plot_df) >= 2 and plot_df[x_column].nunique() > 1:
        slope, intercept = __import__('numpy').polyfit(plot_df[x_column], plot_df[y_column], 1)
        xs = plot_df[x_column].sort_values()
        plt.plot(xs, slope * xs + intercept, color='#d9534f', linewidth=2,
                 label='线性趋势（仅描述）')
        plt.legend(frameon=False)

    plt.axhline(0, color='#b8c2cc', linewidth=0.8)
    plt.axvline(0, color='#b8c2cc', linewidth=0.8)
    plt.grid(True, alpha=0.18)

    plt.xlabel(display_label(x_column))
    plt.ylabel(display_label(y_column))
    plt.title(chart_title)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    correlation = plot_df[x_column].corr(
        plot_df[y_column],
        method="pearson",
    )

    if pd.isna(correlation):
        pearson_correlation = None
    else:
        pearson_correlation = round(
            float(correlation),
            4,
        )

    relationship_summary = {
        "point_count": int(len(plot_df)),
        "x_min": float(plot_df[x_column].min()),
        "x_max": float(plot_df[x_column].max()),
        "y_min": float(plot_df[y_column].min()),
        "y_max": float(plot_df[y_column].max()),
        "pearson_correlation": pearson_correlation,
    }

    return chart_path, plot_df, relationship_summary


def _create_histogram_chart(
    df: pd.DataFrame,
    request: ChartRequest,
) -> tuple[Path, pd.DataFrame, dict[str, int | float]]:
    """
    生成直方图。

    直方图要求：
    - x_column 必须是数值列；
    - 不需要 y_column；
    - 缺失值不会参与绘图。

    返回：
    - 图片路径；
    - 实际参与绘图的一列数据；
    - 这列数据的基础统计摘要。
    """
    x_column = request.x_column

    if not is_numeric_dtype(df[x_column]):
        raise ValueError(
            f"无法生成直方图：x_column={x_column} "
            "必须是数值列。"
        )

    plot_df = df[[x_column]].dropna().copy()

    if plot_df.empty:
        raise ValueError(
            "用于生成直方图的数据为空。"
            "请检查所选列是否全部为缺失值。"
        )

    CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chart_title = request.title or f"Distribution of {x_column}"

    filename_prefix = _build_safe_filename(chart_title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    chart_path = CHART_OUTPUT_DIR / f"{filename_prefix}_{timestamp}.png"

    plt.figure(figsize=(8, 5))

    plt.hist(
        plot_df[x_column],
        bins="auto",
    )

    plt.xlabel(x_column)
    plt.ylabel("Frequency")
    plt.title(chart_title)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    values = plot_df[x_column]

    numeric_summary = {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }

    return chart_path, plot_df, numeric_summary


def _create_line_chart(
    df: pd.DataFrame,
    request: ChartRequest,
) -> tuple[Path, pd.DataFrame, str]:
    """
    生成折线图。

    折线图要求：
    - x_column 必须能解析为日期；
    - aggregation 必须是 mean、sum 或 count；
    - 当 aggregation 是 mean 或 sum 时，
      y_column 必须存在且是数值列。

    返回：
    - 图片路径；
    - 实际用于绘图的按日期汇总数据；
    - 汇总数值列名。
    """
    x_column = request.x_column

    if request.aggregation is None:
        raise ValueError(
            "折线图需要指定 aggregation。"
            "例如 mean、sum 或 count。"
        )

    parsed_dates = pd.to_datetime(
        df[x_column],
        errors="coerce",
        format="mixed",
    )

    if parsed_dates.notna().sum() == 0:
        raise ValueError(
            f"无法生成折线图：x_column={x_column} "
            "必须是可解析的日期列。"
        )

    time_df = df.copy()
    time_df["_chart_time"] = parsed_dates

    if request.aggregation == "count":
        summary_df = (
            time_df.dropna(subset=["_chart_time"])
            .groupby("_chart_time")
            .size()
            .reset_index(name="count")
            .sort_values("_chart_time")
        )

        value_column = "count"

    else:
        if request.y_column is None:
            raise ValueError(
                f"aggregation={request.aggregation} 时，"
                "必须提供 y_column。"
            )

        y_column = request.y_column

        if not is_numeric_dtype(df[y_column]):
            raise ValueError(
                f"无法生成折线图：y_column={y_column} "
                "必须是数值列。"
            )

        clean_df = time_df[
            ["_chart_time", y_column]
        ].dropna()

        if clean_df.empty:
            raise ValueError(
                "用于生成折线图的数据为空。"
                "请检查日期列和数值列是否全部为缺失值。"
            )

        aggregation_map = {
            "mean": "mean",
            "sum": "sum",
        }

        value_column = f"{request.aggregation}_{y_column}"

        summary_df = (
            clean_df.groupby("_chart_time")[y_column]
            .agg(aggregation_map[request.aggregation])
            .reset_index(name=value_column)
            .sort_values("_chart_time")
        )

    if summary_df.empty:
        raise ValueError(
            "用于生成折线图的数据为空。"
        )

    CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chart_title = request.title or (
        f"{request.aggregation} {request.y_column or 'records'} "
        f"by {x_column}"
    )

    filename_prefix = _build_safe_filename(chart_title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    chart_path = CHART_OUTPUT_DIR / f"{filename_prefix}_{timestamp}.png"

    plt.figure(figsize=(8, 5))

    plt.plot(
        summary_df["_chart_time"],
        summary_df[value_column],
        marker="o",
    )

    plt.xlabel(x_column)
    plt.ylabel(value_column)
    plt.title(chart_title)

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

    # 内部用 datetime 排序和绘图，
    # 但返回给模型时转为普通日期字符串，便于 JSON 传递。
    result_df = summary_df.rename(
        columns={
            "_chart_time": x_column,
        }
    ).copy()

    result_df[x_column] = result_df[x_column].dt.strftime(
        "%Y-%m-%d"
    )

    return chart_path, result_df, value_column


def create_chart(
    dataset_path: str,
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
    group_by: str | None = None,
    aggregation: str | None = None,
    title: str | None = None,
) -> ToolResult:
    """
    根据 CSV 数据生成图表。

    当前第一版已支持：
    - bar：类别对比柱状图

    后续会在同一个工具中加入：
    - line
    - scatter
    - histogram
    """
    try:
        request = ChartRequest(
            dataset_path=dataset_path,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            group_by=group_by,
            aggregation=aggregation,
            title=title,
        )

        file_path = Path(request.dataset_path)

        if not file_path.exists():
            return ToolResult(
                tool_name="create_chart",
                success=False,
                error=f"Dataset does not exist: {dataset_path}",
            )

        if not file_path.is_file():
            return ToolResult(
                tool_name="create_chart",
                success=False,
                error=f"Dataset path is not a file: {dataset_path}",
            )

        df = pd.read_csv(file_path)

        if request.x_column not in df.columns:
            return ToolResult(
                tool_name="create_chart",
                success=False,
                error=(
                    f"Column does not exist: {request.x_column}. "
                    f"Available columns: {df.columns.tolist()}"
                ),
            )

        if (
            request.y_column is not None
            and request.y_column not in df.columns
        ):
            return ToolResult(
                tool_name="create_chart",
                success=False,
                error=(
                    f"Column does not exist: {request.y_column}. "
                    f"Available columns: {df.columns.tolist()}"
                ),
            )

        if request.chart_type == "bar":
            chart_path, plot_data_df, value_column = _create_bar_chart(
                df,
                request,
            )

            plotted_row_count = len(plot_data_df)
            numeric_summary = None
            relationship_summary = None


        elif request.chart_type == "scatter":

            chart_path, plot_data_df, relationship_summary = (
                _create_scatter_chart(
                    df,
                    request,
                )
            )

            value_column = None
            plotted_row_count = len(plot_data_df)
            numeric_summary = None

        elif request.chart_type == "histogram":
            chart_path, plot_data_df, numeric_summary = (
                _create_histogram_chart(
                    df,
                    request,
                )
            )

            value_column = None
            plotted_row_count = len(plot_data_df)
            relationship_summary = None

        elif request.chart_type == "line":
            chart_path, plot_data_df, value_column = _create_line_chart(
                df,
                request,
            )

            plotted_row_count = len(plot_data_df)
            numeric_summary = None
            relationship_summary = None

        else:
            return ToolResult(
                tool_name="create_chart",
                success=False,
                error=(
                    f"chart_type={request.chart_type} "
                    "is not implemented yet."
                ),
            )

        return ToolResult(
            tool_name="create_chart",
            success=True,
            result={
                "chart_path": chart_path.as_posix(),
                "chart_type": request.chart_type,
                "x_column": request.x_column,
                "y_column": request.y_column,
                "aggregation": request.aggregation,
                "value_column": value_column,

                # 实际参与绘图的数据行数。
                # 对 scatter 而言，缺失 x 或 y 的行不会被画入图中。
                "plotted_row_count": plotted_row_count,
                "numeric_summary": numeric_summary,
                "relationship_summary": relationship_summary,

                # bar 返回分组后的汇总数据；
                # scatter 返回实际参与绘图的 x / y 点。
                "data_preview": (
                    plot_data_df.head(MAX_DATA_PREVIEW_ROWS)
                    .to_dict(orient="records")
                ),
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="create_chart",
            success=False,
            error=f"Unable to create chart: {exc}",
        )

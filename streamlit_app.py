from __future__ import annotations

from io import BytesIO
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from causalens.agent.graph import run_causalens_graph


# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="CausaLens",
    page_icon="📊",
    layout="wide",
)


# =========================
# 基础路径
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

UPLOAD_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================
# 上传文件保存函数
# =========================

def save_uploaded_csv(uploaded_file) -> Path:
    """
    将 Streamlit 上传的 CSV
    保存到项目 outputs/uploads 目录。

    当前 LangGraph 使用 dataset_path，
    因此 Web 层需要先把上传文件保存为真实路径。
    """

    safe_name = Path(
        uploaded_file.name
    ).name

    unique_name = (
        f"{uuid.uuid4().hex}_{safe_name}"
    )

    destination = (
        UPLOAD_DIR / unique_name
    )

    destination.write_bytes(
        uploaded_file.getbuffer()
    )

    return destination


def load_uploaded_csv(
    uploaded_file,
) -> pd.DataFrame:
    """
    读取用户上传的 CSV，
    用于 Web 页面中的数据预览。

    这里只负责 UI 预览，
    不代替 Agent 内部的数据分析工具。
    """

    file_bytes = uploaded_file.getvalue()

    dataframe = pd.read_csv(
        BytesIO(file_bytes)
    )

    return dataframe


def extract_chart_paths(
    final_state: dict,
) -> list[Path]:
    """
    从 Agent 最终状态中提取成功生成的图表路径。
    """

    tool_results = final_state.get(
        "tool_results",
        [],
    )

    chart_paths: list[Path] = []

    if not isinstance(tool_results, list):
        return chart_paths

    for tool_result in tool_results:

        if not isinstance(tool_result, dict):
            continue

        if (
            tool_result.get("tool_name")
            != "create_chart"
        ):
            continue

        if not tool_result.get("success"):
            continue

        result = tool_result.get("result")

        if not isinstance(result, dict):
            continue

        raw_chart_path = result.get(
            "chart_path"
        )

        if not raw_chart_path:
            continue

        chart_path = Path(
            str(raw_chart_path)
        )

        # Agent 可能返回相对于项目根目录的路径，
        # Streamlit 展示前统一解析成真实路径。
        if not chart_path.is_absolute():
            chart_path = (
                PROJECT_ROOT
                / chart_path
            )

        if (
            chart_path.exists()
            and chart_path not in chart_paths
        ):
            chart_paths.append(
                chart_path
            )

    return chart_paths


def extract_rag_sources(
    final_state: dict,
) -> list[dict]:
    """
    从 retrieve_docs 的工具结果中
    提取 RAG 证据来源。
    """

    tool_results = final_state.get(
        "tool_results",
        [],
    )

    sources: list[dict] = []

    if not isinstance(
        tool_results,
        list,
    ):
        return sources

    for tool_result in tool_results:

        if not isinstance(
            tool_result,
            dict,
        ):
            continue

        if (
            tool_result.get("tool_name")
            != "retrieve_docs"
        ):
            continue

        if not tool_result.get(
            "success"
        ):
            continue

        result = tool_result.get(
            "result",
            {},
        )

        if not isinstance(
            result,
            dict,
        ):
            continue

        rag_results = result.get(
            "results",
            [],
        )

        if not isinstance(
            rag_results,
            list,
        ):
            continue

        for rag_result in rag_results:

            if not isinstance(
                rag_result,
                dict,
            ):
                continue

            source = rag_result.get(
                "source"
            )

            if not source:
                continue

            source_info = {
                "source": str(source),
                "section": rag_result.get(
                    "section"
                ),
                "chunk_id": rag_result.get(
                    "chunk_id"
                ),
                "distance": rag_result.get(
                    "distance"
                ),
            }

            # 避免同一个检索片段重复显示。
            if source_info not in sources:
                sources.append(
                    source_info
                )

    return sources


def render_rag_sources(
    final_state: dict,
) -> None:
    """
    在 Web 页面中展示 RAG 检索证据来源。
    """

    sources = extract_rag_sources(
        final_state
    )

    if not sources:
        return

    st.subheader(
        "RAG 证据来源"
    )

    for source_info in sources:

        source = source_info.get(
            "source",
            "unknown",
        )

        section = source_info.get(
            "section"
        )

        chunk_id = source_info.get(
            "chunk_id"
        )

        st.markdown(
            f"✓ `{source}`"
        )

        if section:
            st.caption(
                f"Section: {section}"
            )

        elif chunk_id:
            st.caption(
                f"Chunk: {chunk_id}"
            )


def build_markdown_report(
    final_state: dict,
) -> str:
    """
    根据 Agent 最终状态生成 Markdown 分析报告。

    报告只整理已经存在的分析结果，
    不重新执行任何统计或因果计算。
    """

    question = str(
        final_state.get(
            "question",
            "",
        )
    )

    task_type = str(
        final_state.get(
            "task_type",
            "unknown",
        )
    )

    workflow_status = str(
        final_state.get(
            "workflow_status",
            "unknown",
        )
    )

    final_answer = str(
        final_state.get(
            "final_answer",
            "",
        )
        or ""
    )

    dataset_path = str(
        final_state.get(
            "dataset_path",
            "",
        )
    )

    plan = final_state.get(
        "plan",
        [],
    )

    tool_results = final_state.get(
        "tool_results",
        [],
    )

    report_lines: list[str] = []

    # =========================
    # 标题
    # =========================

    report_lines.extend(
        [
            "# CausaLens Analysis Report",
            "",
            "## Research Question",
            "",
            question or "未提供。",
            "",
        ]
    )

    # =========================
    # 数据集概览
    # =========================

    report_lines.extend(
        [
            "## Dataset Overview",
            "",
        ]
    )

    if dataset_path:

        try:
            dataset_df = pd.read_csv(
                dataset_path
            )

        except Exception:
            report_lines.append(
                "数据集概览读取失败。"
            )

        else:
            report_lines.extend(
                [
                    (
                        f"- Rows: "
                        f"{len(dataset_df)}"
                    ),
                    (
                        f"- Columns: "
                        f"{len(dataset_df.columns)}"
                    ),
                    (
                        f"- Missing values: "
                        f"{int(dataset_df.isna().sum().sum())}"
                    ),
                    (
                        "- Column names: "
                        + ", ".join(
                            str(column)
                            for column
                            in dataset_df.columns
                        )
                    ),
                ]
            )

    else:
        report_lines.append(
            "未记录数据集路径。"
        )

    report_lines.append("")

    # =========================
    # 工作流信息
    # =========================

    report_lines.extend(
        [
            "## Workflow",
            "",
            f"- Task type: `{task_type}`",
            (
                "- Workflow status: "
                f"`{workflow_status}`"
            ),
            "",
        ]
    )

    # =========================
    # 分析计划
    # =========================

    report_lines.extend(
        [
            "## Analysis Plan",
            "",
        ]
    )

    if isinstance(plan, list) and plan:

        for index, step in enumerate(
            plan,
            start=1,
        ):
            report_lines.append(
                f"{index}. {step}"
            )

    else:
        report_lines.append(
            "未记录分析计划。"
        )

    report_lines.append("")

    # =========================
    # 方法与工具
    # =========================

    report_lines.extend(
        [
            "## Methods and Tool Execution",
            "",
        ]
    )

    executed_tool_names: list[str] = []

    if isinstance(
        tool_results,
        list,
    ):

        for tool_result in tool_results:

            if not isinstance(
                tool_result,
                dict,
            ):
                continue

            tool_name = str(
                tool_result.get(
                    "tool_name",
                    "unknown_tool",
                )
            )

            success = bool(
                tool_result.get(
                    "success",
                    False,
                )
            )

            executed_tool_names.append(
                tool_name
            )

            status = (
                "Success"
                if success
                else "Failed"
            )

            report_lines.append(
                f"- `{tool_name}`: {status}"
            )

    if not executed_tool_names:
        report_lines.append(
            "未执行分析工具。"
        )

    report_lines.append("")

    # =========================
    # 因果配置
    # =========================

    if task_type == "causal_analysis":

        treatment = final_state.get(
            "treatment"
        )

        outcome = final_state.get(
            "outcome"
        )

        confounders = final_state.get(
            "confounders",
            [],
        )

        report_lines.extend(
            [
                "## Causal Configuration",
                "",
                f"- Treatment: `{treatment}`",
                f"- Outcome: `{outcome}`",
                (
                    "- Confounders: "
                    + ", ".join(
                        str(item)
                        for item
                        in confounders
                    )
                ),
                "",
            ]
        )

    # =========================
    # 最终结果
    # =========================

    report_lines.extend(
        [
            "## Results",
            "",
            (
                final_answer
                if final_answer
                else "没有生成最终回答。"
            ),
            "",
        ]
    )

    # =========================
    # 图表
    # =========================

    chart_paths = extract_chart_paths(
        final_state
    )

    if chart_paths:

        report_lines.extend(
            [
                "## Visualization",
                "",
            ]
        )

        for chart_path in chart_paths:
            report_lines.append(
                f"- {chart_path.name}"
            )

        report_lines.append("")

    # =========================
    # RAG 来源
    # =========================

    rag_sources = extract_rag_sources(
        final_state
    )

    if rag_sources:

        report_lines.extend(
            [
                "## Sources",
                "",
            ]
        )

        for source_info in rag_sources:

            source = source_info.get(
                "source",
                "unknown",
            )

            section = source_info.get(
                "section"
            )

            if section:
                report_lines.append(
                    f"- {source} — {section}"
                )

            else:
                report_lines.append(
                    f"- {source}"
                )

        report_lines.append("")

    # =========================
    # 因果假设与限制
    # =========================

    if task_type == "causal_analysis":

        readiness = final_state.get(
            "causal_readiness",
            {},
        )

        warnings = []

        if isinstance(
            readiness,
            dict,
        ):
            warnings = readiness.get(
                "warnings",
                [],
            )

        report_lines.extend(
            [
                "## Assumptions and Limitations",
                "",
            ]
        )

        if warnings:

            for warning in warnings:
                report_lines.append(
                    f"- {warning}"
                )

        else:
            report_lines.append(
                "- 当前状态中未记录额外的因果分析警告。"
            )

        report_lines.append("")

    # =========================
    # 页尾
    # =========================

    report_lines.extend(
        [
            "---",
            "",
            (
                "Generated by CausaLens "
                "CSV Data Research Agent."
            ),
        ]
    )

    return "\n".join(
        report_lines
    )


def render_report_download(
    final_state: dict,
) -> None:
    """
    生成并展示 Markdown 报告下载按钮。
    """

    workflow_status = final_state.get(
        "workflow_status"
    )

    if workflow_status not in {
        "completed",
        "failed",
        "blocked",
    }:
        return

    markdown_report = build_markdown_report(
        final_state
    )

    st.subheader(
        "分析报告"
    )

    with st.expander(
        "预览 Markdown 报告"
    ):
        st.markdown(
            markdown_report
        )

    st.download_button(
        label="下载 Markdown 报告",
        data=markdown_report,
        file_name=(
            "causalens_analysis_report.md"
        ),
        mime=(
            "text/markdown; charset=utf-8"
        ),
        type="primary",
        on_click="ignore",
    )


def render_analysis_plan(
    final_state: dict,
) -> None:
    """
    以产品化形式展示 Agent 的分析计划。

    这里只展示 Planner 生成的显式步骤，
    不展示模型隐藏推理过程。
    """

    plan = final_state.get(
        "plan",
        [],
    )

    if not plan:
        return

    st.subheader(
        "分析计划"
    )

    for step in plan:
        st.markdown(
            f"✓ {step}"
        )


def render_execution_trace(
    final_state: dict,
) -> None:
    """
    展示已经实际发生的工具执行轨迹。
    """

    tool_results = final_state.get(
        "tool_results",
        [],
    )

    if not tool_results:
        return

    st.subheader(
        "Agent 执行轨迹"
    )

    for tool_result in tool_results:

        if not isinstance(
            tool_result,
            dict,
        ):
            continue

        tool_name = tool_result.get(
            "tool_name",
            "unknown_tool",
        )

        success = tool_result.get(
            "success",
            False,
        )

        if success:
            st.markdown(
                f"✓ `{tool_name}`"
            )

        else:
            st.markdown(
                f"✗ `{tool_name}`"
            )


# =========================
# 页面标题
# =========================

st.title("CausaLens")

st.caption(
    "面向 CSV 数据集的统计与因果分析 Agent"
)

st.write(
    "上传 CSV 数据集，"
    "然后使用自然语言描述你的分析问题。"
)


# =========================
# CSV 上传
# =========================

uploaded_file = st.file_uploader(
    "上传 CSV 数据集",
    type=["csv"],
)


# =========================
# 数据集预览
# =========================

if uploaded_file is not None:

    try:
        preview_df = load_uploaded_csv(
            uploaded_file
        )

    except Exception as exc:
        st.error(
            "CSV 文件读取失败，"
            "请检查文件格式或编码。"
        )

        with st.expander(
            "查看读取错误"
        ):
            st.exception(exc)

    else:
        st.subheader(
            "数据集预览"
        )

        row_count = len(
            preview_df
        )

        column_count = len(
            preview_df.columns
        )

        file_size_kb = (
            len(uploaded_file.getvalue())
            / 1024
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "行数",
                row_count,
            )

        with col2:
            st.metric(
                "列数",
                column_count,
            )

        with col3:
            st.metric(
                "文件大小",
                f"{file_size_kb:.1f} KB",
            )

        st.markdown(
            "**字段：** "
            + " · ".join(f"`{col}`" for col in preview_df.columns)
        )

        st.dataframe(
            preview_df.head(10),
            use_container_width=True,
        )

        st.caption(
            "当前展示前 10 行数据。"
        )


# =========================
# 用户问题
# =========================

question = st.text_area(
    "输入你的分析问题",
    placeholder=(
        "例如：study_hours 和 score "
        "是否存在显著相关关系？"
    ),
    height=120,
)


# =========================
# 分析按钮
# =========================

analyze_clicked = st.button(
    "开始分析",
    type="primary",
)


# =========================
# Session State 初始化
# =========================

if "final_state" not in st.session_state:
    st.session_state.final_state = None

if "current_dataset_path" not in st.session_state:
    st.session_state.current_dataset_path = None

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "causal_cancelled" not in st.session_state:
    st.session_state.causal_cancelled = False


# =========================
# 执行新的 Agent 请求
# =========================

if analyze_clicked:

    if uploaded_file is None:
        st.warning(
            "请先上传一个 CSV 文件。"
        )

    elif not question.strip():
        st.warning(
            "请输入需要分析的问题。"
        )

    else:
        dataset_path = save_uploaded_csv(
            uploaded_file
        )

        # 保存当前任务，
        # 后续人工审批时继续使用。
        st.session_state.current_dataset_path = str(
            dataset_path
        )

        st.session_state.current_question = (
            question.strip()
        )

        st.session_state.causal_cancelled = False

        with st.spinner(
            "CausaLens 正在分析..."
        ):
            try:
                final_state = (
                    run_causalens_graph(
                        dataset_path=str(
                            dataset_path
                        ),
                        question=question.strip(),
                    )
                )

            except Exception as exc:
                st.session_state.final_state = None

                st.error(
                    "分析过程中发生未处理异常。"
                )

                with st.expander(
                    "开发调试信息"
                ):
                    st.exception(exc)

            else:
                st.session_state.final_state = (
                    final_state
                )


# =========================
# 读取当前分析状态
# =========================

final_state = st.session_state.final_state


# =========================
# 展示 Agent 结果
# =========================

if final_state is not None:

    workflow_status = final_state.get(
        "workflow_status",
        "unknown",
    )

    task_type = final_state.get(
        "task_type",
        "unknown",
    )

    final_answer = (
        final_state.get(
            "final_answer"
        )
        or ""
    )

    # =========================
    # 因果分析人工审批
    # =========================

    if workflow_status == "awaiting_approval":

        render_analysis_plan(
            final_state
        )

        render_execution_trace(
            final_state
        )


        st.subheader(
            "因果分析审批"
        )

        st.warning(
            "因果分析准备度检查已经通过，"
            "但 LinearDML 尚未执行。"
            "需要你的明确确认后才能继续。"
        )

        treatment = final_state.get(
            "treatment"
        )

        outcome = final_state.get(
            "outcome"
        )

        confounders = final_state.get(
            "confounders",
            [],
        )

        readiness = final_state.get(
            "causal_readiness",
            {},
        )

        if not isinstance(
            readiness,
            dict,
        ):
            readiness = {}

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Treatment",
                str(treatment),
            )

        with col2:
            st.metric(
                "Outcome",
                str(outcome),
            )

        st.write(
            "控制变量：",
            confounders,
        )

        sample_summary = readiness.get(
            "sample_summary",
            {},
        )

        if isinstance(
            sample_summary,
            dict,
        ):
            complete_cases = (
                sample_summary.get(
                    "complete_cases"
                )
            )

            total_rows = (
                sample_summary.get(
                    "total_rows"
                )
            )

            if (
                complete_cases is not None
                or total_rows is not None
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "完整样本数",
                        str(
                            complete_cases
                        ),
                    )

                with col2:
                    st.metric(
                        "总样本数",
                        str(
                            total_rows
                        ),
                    )

        warnings = readiness.get(
            "warnings",
            [],
        )

        if warnings:
            st.markdown(
                "**重要限制：**"
            )

            for warning in warnings:
                st.warning(
                    str(warning)
                )

        st.info(
            "准备度检查通过只表示数据满足当前 "
            "DML 工具的最低技术条件，"
            "不代表因果识别假设已经被证明。"
        )

        approve_col, cancel_col = (
            st.columns(2)
        )

        with approve_col:
            approve_clicked = st.button(
                "继续执行 LinearDML",
                type="primary",
            )

        with cancel_col:
            cancel_clicked = st.button(
                "取消因果分析",
            )

        # =========================
        # 用户批准
        # =========================

        if approve_clicked:

            dataset_path = (
                st.session_state
                .current_dataset_path
            )

            saved_question = (
                st.session_state
                .current_question
            )

            if (
                dataset_path
                and saved_question
            ):

                with st.spinner(
                    "正在执行 LinearDML 因果效应估计..."
                ):
                    try:
                        approved_state = (
                            run_causalens_graph(
                                dataset_path=(
                                    dataset_path
                                ),
                                question=(
                                    saved_question
                                ),
                                approved=True,
                            )
                        )

                    except Exception as exc:
                        st.error(
                            "LinearDML 执行过程中"
                            "发生未处理异常。"
                        )

                        with st.expander(
                            "开发调试信息"
                        ):
                            st.exception(exc)

                    else:
                        st.session_state.final_state = (
                            approved_state
                        )

                        st.session_state.causal_cancelled = (
                            False
                        )

                        st.rerun()

        # =========================
        # 用户取消
        # =========================

        if cancel_clicked:
            st.session_state.causal_cancelled = (
                True
            )

            st.session_state.final_state = None

            st.info(
                "已取消因果分析，"
                "LinearDML 没有执行。"
            )

            st.rerun()

    # =========================
    # 普通完成 / 失败结果
    # =========================

    else:
        render_analysis_plan(
            final_state
        )

        render_execution_trace(
            final_state
        )

        st.subheader(
            "分析结果"
        )

        if workflow_status == "completed":
            st.success(
                "分析完成"
            )

        elif workflow_status == "failed":
            st.error(
                "分析未完成"
            )

        elif workflow_status == "blocked":
            st.warning(
                "分析已被安全检查阻止"
            )

        else:
            st.info(
                f"当前状态："
                f"{workflow_status}"
            )

        if final_answer:
            st.markdown(
                final_answer
            )

        else:
            st.info(
                "当前工作流没有生成最终文本回答。"
            )

        render_rag_sources(
            final_state
        )

        # =========================
        # Agent 生成图表
        # =========================

        chart_paths = extract_chart_paths(
            final_state
        )

        if chart_paths:
            st.subheader(
                "生成图表"
            )

            for chart_path in chart_paths:
                st.image(
                    str(chart_path),
                    caption=chart_path.name,
                    use_container_width=True,
                )

        # =========================
        # Markdown 分析报告
        # =========================

        render_report_download(
            final_state
        )

        # =========================
        # 工作流基本信息
        # =========================

        st.subheader(
            "工作流信息"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "任务类型",
                str(task_type),
            )

        with col2:
            st.metric(
                "工作流状态",
                str(workflow_status),
            )

        # =========================
        # Agent 调试信息
        # =========================

        with st.expander(
            "查看 Agent 执行详情"
        ):

            st.write(
                "分析计划："
            )

            st.write(
                final_state.get(
                    "plan",
                    [],
                )
            )

            st.write(
                "工具调用："
            )

            st.write(
                final_state.get(
                    "tool_calls",
                    [],
                )
            )

            st.write(
                "工具结果："
            )

            st.write(
                final_state.get(
                    "tool_results",
                    [],
                )
            )


# =========================
# 已取消状态
# =========================

elif st.session_state.causal_cancelled:

    st.info(
        "因果分析已取消。"
        "LinearDML 未执行，"
        "也没有生成因果效应估计。"
    )
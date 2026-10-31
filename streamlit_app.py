from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from causalens.agent.graph import run_causalens_graph
from causalens.causal.contract_runner import run_explicit_causal_contract
from causalens.ui.research_advisor import render_research_advisor
from causalens.ui.control_selection import render_control_selection
from causalens.ui.analysis_identity import analysis_identity
from causalens.ui.planning_handoff import render_context_inputs, missing_context, attach_planning
from causalens.ui.data_workbench import select_analysis_version
from causalens.ui.result_comparison import render_comparison
from causalens.reporting.plain_summary import summarize_time_series
from causalens.ui.causal_guidance import (
    assess_guided_compatibility,
    build_guided_contract,
    contract_result_to_agent_state,
    normalize_guided_roles,
    parse_professional_contract,
    profile_columns,
    suggest_causal_roles,
    suggest_causal_roles_with_model,
)


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

        if not tool_result.get("success"):
            continue

        result = tool_result.get("result")

        if not isinstance(result, dict):
            continue

        raw_chart_paths: list[object] = []
        if tool_result.get("tool_name") == "create_chart":
            raw_chart_paths.append(result.get("chart_path"))
        elif tool_result.get("tool_name") == "run_causal_estimators":
            for estimate in result.get("estimates", []) or []:
                if isinstance(estimate, dict):
                    raw_chart_paths.append(estimate.get("chart_path"))
        for raw_chart_path in raw_chart_paths:
            if not raw_chart_path:
                continue

            chart_path = Path(str(raw_chart_path))

        # Agent 可能返回相对于项目根目录的路径，
        # Streamlit 展示前统一解析成真实路径。
            if not chart_path.is_absolute():
                chart_path = PROJECT_ROOT / chart_path

            if chart_path.exists() and chart_path not in chart_paths:
                chart_paths.append(chart_path)

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


def _format_optional_number(value: object, digits: int = 4) -> str:
    """Format optional numeric diagnostics without hiding missing values."""

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "未知"


def render_causal_estimator_results(final_state: dict) -> None:
    """直接展示确定性估计器输出，不依赖 LLM 是否完整转述。"""

    for tool_result in final_state.get("tool_results", []) or []:
        if not isinstance(tool_result, dict):
            continue
        if tool_result.get("tool_name") != "run_causal_estimators":
            continue
        if tool_result.get("success") is not True:
            continue
        result = tool_result.get("result") or {}
        if not isinstance(result, dict):
            continue

        st.subheader("因果估计器比较")
        estimand = str(result.get("estimand") or "ATE")
        st.caption(
            f"Estimand：{estimand}；"
            "主估计器："
            f"{result.get('primary_estimator', '未知')}；"
            "稳健性估计器："
            + "、".join(result.get("robustness_estimators") or [])
        )
        evidence_grade = result.get("evidence_grade") or {}
        if isinstance(evidence_grade, dict) and evidence_grade.get("grade"):
            grade_text = (
                "因果证据等级："
                f"{evidence_grade.get('label_zh', '未知')} "
                f"(`{evidence_grade.get('grade')}`)"
            )
            if evidence_grade.get("grade") == "stronger_support":
                st.success(grade_text)
            elif evidence_grade.get("grade") in {
                "limited_support",
                "exploratory_only",
            }:
                st.warning(grade_text)
            else:
                st.info(grade_text)
            with st.expander("查看证据等级依据"):
                st.write(
                    "**支持信号：**",
                    evidence_grade.get("supporting_signals") or [],
                )
                st.write(
                    "**红旗：**",
                    evidence_grade.get("red_flags") or [],
                )
                st.write(
                    "**尚缺检查：**",
                    evidence_grade.get("missing_checks") or [],
                )
                st.caption(
                    "这是对现有诊断的确定性汇总，不是因果命题为真的概率，"
                    "也不是识别成立的证明。"
                )
        data_quality = result.get("data_quality") or {}
        design_selection = result.get("design_selection") or {}
        if isinstance(data_quality, dict) and data_quality:
            with st.expander("查看数据质量与方法路由"):
                st.write(
                    "**数据质量门禁：**",
                    "通过" if data_quality.get("ready") else "未通过",
                )
                st.write(
                    "**完整样本比例：**",
                    (data_quality.get("sample_summary") or {}).get(
                        "complete_case_fraction"
                    ),
                )
                st.write("**质量警告：**", data_quality.get("warnings") or [])
                st.write(
                    "**确定性路由：**",
                    design_selection.get("study_design"),
                    "→",
                    design_selection.get("selected_estimator_names") or [],
                )
                st.caption("系统不会自动插补、编码类别变量或删除异常值。")
        estimates = result.get("estimates") or []
        if estimates:
            rows = []
            for estimate in estimates:
                if not isinstance(estimate, dict):
                    continue
                interval = estimate.get("confidence_interval") or []
                effect_label = (
                    "当期响应（每 1 单位冲击）"
                    if estimand == "IRF"
                    else f"平均 {estimand}"
                )
                rows.append(
                    {
                        "估计器": estimate.get("estimator"),
                        effect_label: estimate.get("ate"),
                        "95% CI 下限": interval[0] if len(interval) > 0 else None,
                        "95% CI 上限": interval[1] if len(interval) > 1 else None,
                        "样本量": estimate.get("sample_size"),
                    }
                )
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    width='stretch',
                    hide_index=True,
                )

        primary_estimate = estimates[0] if estimates else {}
        heterogeneity = (
            primary_estimate.get("heterogeneity") or {}
            if isinstance(primary_estimate, dict)
            else {}
        )
        if heterogeneity:
            st.markdown("#### 异质性摘要")
            heterogeneity_columns = st.columns(4)
            for column, label, key in zip(
                heterogeneity_columns,
                ["CATE P10", "CATE 中位数", "CATE P90", "正效应比例"],
                ["cate_p10", "cate_median", "cate_p90", "positive_effect_fraction"],
            ):
                value = heterogeneity.get(key)
                with column:
                    st.metric(
                        label,
                        (
                            f"{float(value):.1%}"
                            if key == "positive_effect_fraction" and value is not None
                            else f"{float(value):.4f}"
                            if value is not None
                            else "未知"
                        ),
                    )
            feature_importances = heterogeneity.get("feature_importances") or {}
            if feature_importances:
                st.write("**Effect modifier 重要性：**", feature_importances)
            subgroup_rows = heterogeneity.get("median_split_subgroups") or []
            if subgroup_rows:
                st.dataframe(
                    pd.DataFrame(subgroup_rows),
                    width='stretch',
                    hide_index=True,
                )

        if (
            isinstance(primary_estimate, dict)
            and primary_estimate.get("estimator") == "TwoStageLeastSquares"
        ):
            st.markdown("#### 工具变量诊断")
            first_stage = primary_estimate.get("first_stage") or {}
            endogeneity = primary_estimate.get("endogeneity_test") or {}
            overidentification = (
                primary_estimate.get("overidentification_test") or {}
            )
            iv_columns = st.columns(3)
            with iv_columns[0]:
                value = first_stage.get("f_statistic")
                st.metric(
                    "第一阶段 F",
                    f"{float(value):.3f}" if value is not None else "未知",
                )
            with iv_columns[1]:
                value = first_stage.get("partial_r2")
                st.metric(
                    "第一阶段 partial R²",
                    f"{float(value):.4f}" if value is not None else "未知",
                )
            with iv_columns[2]:
                st.metric(
                    "弱工具警告",
                    "是"
                    if first_stage.get("weak_instrument_warning")
                    else "否",
                )
            st.write("**工具变量：**", result.get("instruments") or [])
            st.write("**内生性检验：**", endogeneity)
            st.write("**过度识别检验：**", overidentification)
            st.warning(
                "第一阶段强度可由数据诊断，但工具独立性和排除限制不能由"
                "观测数据自动证明。"
            )

        if (
            isinstance(primary_estimate, dict)
            and primary_estimate.get("estimator")
            == "SharpRegressionDiscontinuity"
        ):
            st.markdown("#### 回归不连续诊断")
            density = primary_estimate.get("density_diagnostic") or {}
            continuity = primary_estimate.get("covariate_continuity") or {}
            sensitivity = primary_estimate.get("bandwidth_sensitivity") or {}
            donut = primary_estimate.get("donut_sensitivity") or {}
            support = primary_estimate.get("local_support") or {}
            rdd_columns = st.columns(4)
            with rdd_columns[0]:
                st.metric("Cutoff", primary_estimate.get("cutoff"))
            with rdd_columns[1]:
                st.metric("Bandwidth", primary_estimate.get("bandwidth"))
            with rdd_columns[2]:
                st.metric(
                    "左/右局部样本",
                    f"{support.get('left_count')}/{support.get('right_count')}",
                )
            with rdd_columns[3]:
                st.metric(
                    "带宽方向稳定",
                    "是" if sensitivity.get("direction_stable") else "否",
                )
            st.write("**Running variable：**", primary_estimate.get("running_variable"))
            st.write("**密度计数诊断：**", density)
            st.write("**协变量连续性：**", continuity)
            st.write("**Donut 敏感性：**", donut)
            st.warning(
                "密度结果是局部计数启发式诊断，不是正式 McCrary 检验；"
                "RDD 解释仍依赖 cutoff 附近连续性与无精确操纵。"
            )

        if (
            isinstance(primary_estimate, dict)
            and primary_estimate.get("estimator") == "SyntheticControl"
        ):
            st.markdown("#### 合成控制诊断")
            fit = primary_estimate.get("fit_diagnostics") or {}
            placebo = primary_estimate.get("placebo_inference") or {}
            scm_columns = st.columns(4)
            with scm_columns[0]:
                st.metric("处理单位", primary_estimate.get("treated_unit"))
            with scm_columns[1]:
                st.metric("干预时点", primary_estimate.get("intervention_time"))
            with scm_columns[2]:
                st.metric("处理前 RMSPE", fit.get("pre_rmspe"))
            with scm_columns[3]:
                st.metric("空间安慰剂 p", placebo.get("p_value"))
            st.write("**供体权重：**", primary_estimate.get("donor_weights") or [])
            trajectories = primary_estimate.get("trajectories") or []
            if trajectories:
                chart_frame = pd.DataFrame(trajectories).set_index("time")
                st.line_chart(chart_frame[["treated_outcome", "synthetic_outcome"]])
            st.warning(
                "合成控制结果只针对该处理单位，并依赖无溢出、供体未受干预、"
                "稳定处理前拟合和无同期单位特异冲击。"
            )

        if (
            isinstance(primary_estimate, dict)
            and primary_estimate.get("estimator") == "TimeSeriesLocalProjection"
        ):
            st.markdown("#### 连续冲击的时间序列动态响应")
            configuration = primary_estimate.get("model_configuration") or {}
            dynamic_effects = primary_estimate.get("dynamic_effects") or []
            st.caption(
                "每个系数表示 treatment 原始列增加 1 个单位时 outcome 的动态响应；"
                "这是 IRF / 局部投影结果，不是二元处理 ATE。"
            )
            if dynamic_effects:
                dynamic_frame = pd.DataFrame(dynamic_effects).rename(
                    columns={
                        "horizon": "期数 h",
                        "estimate": "动态响应",
                        "standard_error": "HAC 标准误",
                        "confidence_interval": "95% CI",
                        "p_value": "p 值",
                        "sample_size": "样本量",
                        "hac_lags": "HAC 滞后",
                    }
                )
                st.dataframe(
                    dynamic_frame,
                    width='stretch',
                    hide_index=True,
                )
                chart_frame = pd.DataFrame(dynamic_effects).set_index("horizon")
                st.line_chart(chart_frame[["estimate"]])
            stationarity = primary_estimate.get("stationarity") or {}
            placebo = primary_estimate.get("placebo_test") or {}
            residual = primary_estimate.get("residual_diagnostics") or {}
            diagnostic_columns = st.columns(4)
            with diagnostic_columns[0]:
                st.metric(
                    "Treatment ADF p",
                    _format_optional_number(
                        (stationarity.get("treatment") or {}).get("p_value")
                    ),
                )
            with diagnostic_columns[1]:
                st.metric(
                    "Outcome ADF p",
                    _format_optional_number(
                        (stationarity.get("outcome") or {}).get("p_value")
                    ),
                )
            with diagnostic_columns[2]:
                st.metric(
                    "未来冲击 Placebo p",
                    _format_optional_number(placebo.get("p_value")),
                )
            with diagnostic_columns[3]:
                st.metric(
                    "残差自相关警告",
                    "是"
                    if residual.get("residual_autocorrelation_warning")
                    else "否",
                )
            st.write(
                "**模型配置：**",
                {
                    "响应口径": configuration.get("response"),
                    "最大期数": configuration.get("max_horizon"),
                    "treatment 滞后": configuration.get("treatment_lags"),
                    "outcome 滞后": configuration.get("outcome_lags"),
                    "控制变量滞后": configuration.get("control_lag"),
                    "协方差": configuration.get("covariance"),
                },
            )
            st.warning(
                "HAC 只修正序列相关下的标准误，不会自动建立因果识别。"
                "仍需论证冲击在给定滞后信息后近似外生，且不存在同时影响"
                " treatment 与 outcome 的遗漏冲击。"
            )

        if (
            isinstance(primary_estimate, dict)
            and primary_estimate.get("estimator") == "DifferenceInDifferences"
        ):
            st.markdown("#### Difference-in-Differences 设计摘要")
            st.metric(
                "未调整四单元格 DiD",
                f"{float(primary_estimate.get('unadjusted_did')):.4f}",
            )
            st.write("**四单元格均值：**", primary_estimate.get("cell_means"))
            configuration = primary_estimate.get("model_configuration") or {}
            st.write("**估计模式：**", configuration.get("design_mode"))
            st.warning(
                "DiD 的因果解释依赖平行趋势、无提前反应和无组别特异同期冲击。"
            )

        if (
            isinstance(primary_estimate, dict)
            and primary_estimate.get("estimator") == "EventStudy"
        ):
            st.markdown("#### Event Study 动态效应")
            dynamic_effects = primary_estimate.get("dynamic_effects") or []
            if dynamic_effects:
                st.dataframe(
                    pd.DataFrame(dynamic_effects),
                    width='stretch',
                    hide_index=True,
                )
            pretrend = primary_estimate.get("pretrend_test") or {}
            p_value = pretrend.get("p_value")
            st.metric(
                "联合预趋势检验 p 值",
                f"{float(p_value):.4f}" if p_value is not None else "未知",
            )
            placebo = primary_estimate.get("placebo_test") or {}
            sensitivity = primary_estimate.get("cohort_sensitivity") or {}
            robustness_columns = st.columns(2)
            with robustness_columns[0]:
                placebo_pvalue = placebo.get("p_value")
                st.metric(
                    "提前一期 Placebo p 值",
                    (
                        f"{float(placebo_pvalue):.4f}"
                        if placebo_pvalue is not None
                        else "不可估计"
                    ),
                )
            with robustness_columns[1]:
                max_deviation = sensitivity.get("max_absolute_deviation")
                st.metric(
                    "逐 cohort 剔除最大偏移",
                    (
                        f"{float(max_deviation):.4f}"
                        if max_deviation is not None
                        else "不可估计"
                    ),
                )
            leave_one_out = sensitivity.get("leave_one_cohort_out") or []
            if leave_one_out:
                st.write("**逐 cohort 剔除结果：**")
                st.dataframe(
                    pd.DataFrame(leave_one_out),
                    width='stretch',
                    hide_index=True,
                )
            st.warning(
                "预趋势检验不显著不等于平行趋势已被证明；仍需结合制度背景论证。"
            )

        if (
            isinstance(primary_estimate, dict)
            and primary_estimate.get("estimator") == "StaggeredAdoptionDID"
        ):
            st.markdown("#### 错位处理 Cohort-Time ATT")
            staggered_configuration = (
                primary_estimate.get("model_configuration") or {}
            )
            st.write(
                "**调整模式：**",
                staggered_configuration.get("adjustment"),
            )
            st.write(
                "**基线混杂变量：**",
                staggered_configuration.get("confounders") or [],
            )
            nuisance_diagnostics = (
                staggered_configuration.get("nuisance_diagnostics") or {}
            )
            if nuisance_diagnostics:
                diagnostic_columns = st.columns(2)
                with diagnostic_columns[0]:
                    st.metric(
                        "最小加权对照 ESS",
                        f"{float(nuisance_diagnostics.get('min_weighted_control_ess')):.2f}",
                    )
                with diagnostic_columns[1]:
                    st.metric(
                        "最大倾向截断比例",
                        f"{float(nuisance_diagnostics.get('max_propensity_clip_fraction')):.1%}",
                    )
            group_time = primary_estimate.get("group_time_effects") or []
            if group_time:
                st.dataframe(
                    pd.DataFrame(group_time),
                    width='stretch',
                    hide_index=True,
                )
            dynamic = primary_estimate.get("dynamic_effects") or []
            if dynamic:
                st.markdown("#### 按相对时期聚合的动态效应")
                st.dataframe(
                    pd.DataFrame(dynamic),
                    width='stretch',
                    hide_index=True,
                )
            pretrend = primary_estimate.get("pretrend_test") or {}
            p_value = pretrend.get("p_value")
            st.metric(
                "联合预趋势检验 p 值",
                f"{float(p_value):.4f}" if p_value is not None else "未知",
            )
            st.warning(
                "该方法避免把已经处理的单位作为后续 cohort 的对照；"
                "预趋势不显著仍不能证明 cohort-specific 平行趋势。"
            )

        comparison = result.get("comparison") or {}
        if "absolute_ate_difference" in comparison:
            comparison_columns = st.columns(2)
            with comparison_columns[0]:
                st.metric(
                    "ATE 绝对差",
                    f"{float(comparison['absolute_ate_difference']):.4f}",
                )
            with comparison_columns[1]:
                st.metric(
                    "置信区间重叠",
                    "是"
                    if comparison.get("confidence_intervals_overlap")
                    else "否",
                )
        return


def build_markdown_report(
    final_state: dict,
) -> str:
    """
    根据 Agent 最终状态生成 Markdown 分析报告。

    报告只整理已经存在的分析结果，
    不重新执行任何统计或因果计算。
    """

    from causalens.reporting.chinese import time_series_report

    chinese_report = time_series_report(final_state)
    if chinese_report is not None:
        from causalens.reporting.planning_trace import planning_section
        return chinese_report + '\n\n' + planning_section(final_state)

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

        causal_question = final_state.get(
            "causal_question",
            {},
        )
        if not isinstance(causal_question, dict):
            causal_question = {}
        variable_roles = causal_question.get("variable_roles") or {}
        negative_control_outcomes = (
            variable_roles.get("negative_control_outcomes") or []
            if isinstance(variable_roles, dict)
            else []
        )
        instruments = (
            variable_roles.get("instruments") or []
            if isinstance(variable_roles, dict)
            else []
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
                (
                    "- Negative-control outcomes: "
                    + ", ".join(str(item) for item in negative_control_outcomes)
                ),
                "- Instruments: " + ", ".join(str(item) for item in instruments),
                f"- Estimand: `{causal_question.get('estimand', 'ATE')}`",
                (
                    "- Design type: `"
                    f"{causal_question.get('design_type', 'cross_sectional')}`"
                ),
                f"- Post column: `{causal_question.get('post_column')}`",
                f"- Event time column: `{causal_question.get('event_time_column')}`",
                f"- Reference period: `{causal_question.get('reference_period', -1)}`",
                f"- Time column: `{causal_question.get('time_column')}`",
                (
                    "- Treatment start column: `"
                    f"{causal_question.get('treatment_start_column')}`"
                ),
                f"- Unit ID column: `{causal_question.get('unit_id_column')}`",
                f"- Running variable: `{causal_question.get('running_variable')}`",
                f"- Cutoff: `{causal_question.get('cutoff')}`",
                (
                    "- Treatment above cutoff: `"
                    f"{causal_question.get('treatment_above_cutoff', True)}`"
                ),
                f"- Bandwidth: `{causal_question.get('bandwidth')}`",
                (
                    "- Intervention time: `"
                    f"{causal_question.get('intervention_time')}`"
                ),
                f"- Treated unit: `{causal_question.get('treated_unit')}`",
                (
                    "- Target population: "
                    + str(
                        causal_question.get("target_population")
                        or "未明确"
                    )
                ),
                (
                    "- Unit of analysis: "
                    + str(
                        causal_question.get("unit_of_analysis")
                        or "未明确"
                    )
                ),
                (
                    "- Outcome horizon: "
                    + str(
                        causal_question.get("outcome_horizon")
                        or "未明确"
                    )
                ),
                "",
            ]
        )

        identification = final_state.get(
            "causal_identification",
            {},
        )
        if isinstance(identification, dict) and identification:
            report_lines.extend(
                [
                    "### Identification Assessment",
                    "",
                    f"- Status: `{identification.get('status')}`",
                    f"- DAG provided: `{identification.get('dag_provided')}`",
                    f"- Identified: `{identification.get('identified')}`",
                    "",
                ]
            )

        diagnostics = final_state.get(
            "causal_diagnostics",
            {},
        )
        if (
            isinstance(diagnostics, dict)
            and diagnostics.get("analysis_type") == "iv_design_diagnostics"
        ):
            first_stage = diagnostics.get("first_stage") or {}
            report_lines.extend(
                [
                    "### IV Design Diagnostics",
                    "",
                    f"- Passed: `{diagnostics.get('passed')}`",
                    f"- First-stage F: `{first_stage.get('f_statistic')}`",
                    f"- Partial R²: `{first_stage.get('partial_r2')}`",
                    (
                        "- Weak-instrument warning: `"
                        f"{first_stage.get('weak_instrument_warning')}`"
                    ),
                    "",
                ]
            )
        elif (
            isinstance(diagnostics, dict)
            and diagnostics.get("analysis_type") == "did_design_diagnostics"
        ):
            design_summary = diagnostics.get("design_summary") or {}
            report_lines.extend(
                [
                    "### DiD Design Diagnostics",
                    "",
                    f"- Passed: `{diagnostics.get('passed')}`",
                    f"- Cell counts: `{design_summary.get('cell_counts')}`",
                    f"- Cell means: `{design_summary.get('cell_means')}`",
                    (
                        "- Complete two-period units: `"
                        f"{design_summary.get('complete_two_period_units')}`"
                    ),
                    "",
                ]
            )
        elif isinstance(diagnostics, dict) and diagnostics:
            propensity = diagnostics.get("propensity_summary") or {}
            weights = diagnostics.get("weight_summary") or {}
            report_lines.extend(
                [
                    "### Overlap and Balance Diagnostics",
                    "",
                    f"- Passed: `{diagnostics.get('passed')}`",
                    (
                        "- Common support: "
                        f"`{propensity.get('common_support')}`"
                    ),
                    (
                        "- Extreme propensity fraction: "
                        f"`{propensity.get('extreme_fraction')}`"
                    ),
                    (
                        "- Effective sample size: "
                        f"`{weights.get('effective_sample_size')}`"
                    ),
                    "",
                    "| Covariate | SMD before IPW | SMD after IPW |",
                    "|---|---:|---:|",
                ]
            )
            for row in diagnostics.get("balance", []) or []:
                if not isinstance(row, dict):
                    continue
                report_lines.append(
                    "| {variable} | {before} | {after} |".format(
                        variable=row.get("variable"),
                        before=row.get("smd_before"),
                        after=row.get("smd_after"),
                    )
                )
            report_lines.append("")

        for tool_result in tool_results if isinstance(tool_results, list) else []:
            if not isinstance(tool_result, dict):
                continue
            if tool_result.get("tool_name") != "run_causal_estimators":
                continue
            if tool_result.get("success") is not True:
                continue
            suite = tool_result.get("result") or {}
            if not isinstance(suite, dict):
                continue
            is_irf = str(suite.get("estimand") or "") == "IRF"
            effect_heading = (
                "Impact response per one-unit shock" if is_irf else "Average effect"
            )
            report_lines.extend(
                [
                    "### Estimator Comparison",
                    "",
                    f"- Primary estimator: `{suite.get('primary_estimator')}`",
                    (
                        "- Robustness estimators: "
                        + ", ".join(suite.get("robustness_estimators") or [])
                    ),
                    "",
                    f"| Estimator | {effect_heading} | 95% CI | Sample size |",
                    "|---|---:|---|---:|",
                ]
            )
            for estimate in suite.get("estimates", []) or []:
                if not isinstance(estimate, dict):
                    continue
                report_lines.append(
                    "| {name} | {ate} | {interval} | {sample_size} |".format(
                        name=estimate.get("estimator"),
                        ate=estimate.get("ate"),
                        interval=estimate.get("confidence_interval"),
                        sample_size=estimate.get("sample_size"),
                    )
                )
            comparison = suite.get("comparison") or {}
            report_lines.extend(
                [
                    "",
                    (
                        "- Absolute ATE difference: "
                        f"`{comparison.get('absolute_ate_difference')}`"
                    ),
                    (
                        "- Confidence intervals overlap: "
                        f"`{comparison.get('confidence_intervals_overlap')}`"
                    ),
                    "",
                ]
            )
            evidence_grade = suite.get("evidence_grade") or {}
            if isinstance(evidence_grade, dict) and evidence_grade.get("grade"):
                report_lines.extend(
                    [
                        "### Causal Evidence Grade",
                        "",
                        f"- Grade: `{evidence_grade.get('grade')}`",
                        f"- Label: {evidence_grade.get('label_zh')}",
                        (
                            "- Supporting signals: `"
                            f"{evidence_grade.get('supporting_signals') or []}`"
                        ),
                        (
                            "- Red flags: `"
                            f"{evidence_grade.get('red_flags') or []}`"
                        ),
                        (
                            "- Missing checks: `"
                            f"{evidence_grade.get('missing_checks') or []}`"
                        ),
                        "",
                    ]
                )
            iv_estimate = next(
                (
                    estimate
                    for estimate in suite.get("estimates", []) or []
                    if isinstance(estimate, dict)
                    and estimate.get("estimator") == "TwoStageLeastSquares"
                ),
                None,
            )
            if isinstance(iv_estimate, dict):
                first_stage = iv_estimate.get("first_stage") or {}
                endogeneity = iv_estimate.get("endogeneity_test") or {}
                overidentification = (
                    iv_estimate.get("overidentification_test") or {}
                )
                report_lines.extend(
                    [
                        "### Instrumental-Variable Diagnostics",
                        "",
                        f"- Instruments: `{suite.get('instruments') or []}`",
                        f"- First-stage F: `{first_stage.get('f_statistic')}`",
                        f"- Partial R²: `{first_stage.get('partial_r2')}`",
                        (
                            "- Endogeneity-test p-value: `"
                            f"{endogeneity.get('p_value')}`"
                        ),
                        (
                            "- Hansen J p-value: `"
                            f"{overidentification.get('p_value')}`"
                        ),
                        "",
                    ]
                )
            rdd_estimate = next(
                (
                    estimate
                    for estimate in suite.get("estimates", []) or []
                    if isinstance(estimate, dict)
                    and estimate.get("estimator")
                    == "SharpRegressionDiscontinuity"
                ),
                None,
            )
            if isinstance(rdd_estimate, dict):
                density = rdd_estimate.get("density_diagnostic") or {}
                sensitivity = rdd_estimate.get("bandwidth_sensitivity") or {}
                support = rdd_estimate.get("local_support") or {}
                report_lines.extend(
                    [
                        "### Regression-Discontinuity Diagnostics",
                        "",
                        (
                            "- Running variable: `"
                            f"{rdd_estimate.get('running_variable')}`"
                        ),
                        f"- Cutoff: `{rdd_estimate.get('cutoff')}`",
                        f"- Bandwidth: `{rdd_estimate.get('bandwidth')}`",
                        (
                            "- Local left / right observations: `"
                            f"{support.get('left_count')} / "
                            f"{support.get('right_count')}`"
                        ),
                        f"- Density p-value: `{density.get('p_value')}`",
                        (
                            "- Bandwidth direction stable: `"
                            f"{sensitivity.get('direction_stable')}`"
                        ),
                        "",
                    ]
                )
            synthetic_estimate = next(
                (
                    estimate
                    for estimate in suite.get("estimates", []) or []
                    if isinstance(estimate, dict)
                    and estimate.get("estimator") == "SyntheticControl"
                ),
                None,
            )
            if isinstance(synthetic_estimate, dict):
                fit = synthetic_estimate.get("fit_diagnostics") or {}
                placebo = synthetic_estimate.get("placebo_inference") or {}
                report_lines.extend(
                    [
                        "### Synthetic-Control Diagnostics",
                        "",
                        f"- Treated unit: `{synthetic_estimate.get('treated_unit')}`",
                        (
                            "- Intervention time: `"
                            f"{synthetic_estimate.get('intervention_time')}`"
                        ),
                        f"- Pre-treatment RMSPE: `{fit.get('pre_rmspe')}`",
                        (
                            "- Post/pre RMSPE ratio: `"
                            f"{fit.get('post_pre_rmspe_ratio')}`"
                        ),
                        f"- Placebo rank p-value: `{placebo.get('p_value')}`",
                        (
                            "- Non-zero donor weights: `"
                            f"{synthetic_estimate.get('donor_weights') or []}`"
                        ),
                        "",
                    ]
                )
            time_series_estimate = next(
                (
                    estimate
                    for estimate in suite.get("estimates", []) or []
                    if isinstance(estimate, dict)
                    and estimate.get("estimator") == "TimeSeriesLocalProjection"
                ),
                None,
            )
            if isinstance(time_series_estimate, dict):
                configuration = time_series_estimate.get("model_configuration") or {}
                stationarity = time_series_estimate.get("stationarity") or {}
                placebo = time_series_estimate.get("placebo_test") or {}
                residual = time_series_estimate.get("residual_diagnostics") or {}
                report_lines.extend(
                    [
                        "### Time-Series Local-Projection Diagnostics",
                        "",
                        (
                            "- Estimand: `IRF per one source-unit shock` "
                            "(not a binary-treatment ATE)"
                        ),
                        f"- Response: `{configuration.get('response')}`",
                        f"- Maximum horizon: `{configuration.get('max_horizon')}`",
                        f"- Covariance: `{configuration.get('covariance')}`",
                        (
                            "- Treatment / outcome ADF p-values: `"
                            f"{(stationarity.get('treatment') or {}).get('p_value')} / "
                            f"{(stationarity.get('outcome') or {}).get('p_value')}`"
                        ),
                        f"- Future-treatment placebo p-value: `{placebo.get('p_value')}`",
                        (
                            "- Residual autocorrelation warning: `"
                            f"{residual.get('residual_autocorrelation_warning')}`"
                        ),
                        f"- Dynamic responses: `{time_series_estimate.get('dynamic_effects') or []}`",
                        "",
                        (
                            "> HAC inference adjusts standard errors for serial "
                            "correlation; it does not establish causal identification."
                        ),
                        "",
                    ]
                )
            break

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
    from causalens.reporting.customer_report import customer_report
    readable_report = customer_report(final_state)

    st.subheader(
        "分析报告"
    )

    with st.expander(
        "预览客户简报：结论、影响大小与注意事项"
    ):
        st.markdown(
            readable_report
        )

    st.download_button(
        label="下载客户简报",
        data=readable_report,
        file_name=(
            "causalens_customer_report.md"
        ),
        mime=(
            "text/markdown; charset=utf-8"
        ),
        type="primary",
        on_click="ignore",
    )
    with st.expander('技术与审计记录（研究人员查看）'):
        st.caption('包含模型参数、诊断、AI 规划、引用与修订记录；不会重新调用模型或计算。')
        st.download_button('下载完整技术记录', markdown_report,
                           file_name='causalens_technical_record.md',
                           mime='text/markdown', on_click='ignore')
        st.markdown(markdown_report)


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
    "融合 RAG 与统计工具调用的因果分析探索 Agent"
)

st.write(
    "从理解数据到探索因果：上传 CSV，描述你关注的因素与结果。"
    "可以先比较、绘图和检索背景，再决定是否形成因果研究方案。"
)
st.caption("AI 辅助理解与规划，统计工具计算，用户确认研究方案。差异、相关性和统计显著性不自动代表因果关系。")


# =========================
# CSV 上传
# =========================

entry_mode = st.radio('工作区', ['对话探索工作台', '研究设计与人工配置', '快速论文演示（不调用模型）'], horizontal=True, key='workspace_navigation')
if entry_mode == '对话探索工作台':
    from causalens.ui.research_workspace import render_workspace
    render_workspace(PROJECT_ROOT, save_uploaded_csv, load_uploaded_csv,
                     extract_chart_paths, render_report_download)
    st.stop()
if entry_mode == '快速论文演示（不调用模型）':
    from causalens.ui.demo_mode import render_demo
    render_demo()
    st.stop()

uploaded_file = st.file_uploader(
    "上传 CSV 数据集",
    type=["csv"],
)


# =========================
# 数据集预览
# =========================

if uploaded_file is not None:

    uploaded_file = select_analysis_version(uploaded_file)

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
            width='stretch',
        )

        st.caption(
            "当前展示前 10 行数据。"
        )

        with st.expander("查看自动识别的字段类型"):
            profile_table = pd.DataFrame(profile_columns(preview_df)).rename(
                columns={
                    "column": "字段",
                    "semantic_type": "识别类型",
                    "dtype": "原始类型",
                    "unique_count": "唯一值数",
                    "missing_fraction": "缺失比例",
                    "binary_01": "是否为 0/1",
                    "identifier_like": "疑似 ID",
                }
            )
            st.dataframe(
                profile_table[
                    [
                        "字段",
                        "识别类型",
                        "原始类型",
                        "唯一值数",
                        "缺失比例",
                        "是否为 0/1",
                        "疑似 ID",
                    ]
                ],
                width='stretch',
                hide_index=True,
            )


# =========================
# 使用模式与用户问题
# =========================

analysis_mode = st.radio(
    "选择使用方式",
    options=["快速引导", "自由提问", "专业契约"],
    horizontal=True,
    help=(
        "快速引导适合第一次使用；自由提问保留自然语言 Agent；"
        "专业契约适合已经准备好 JSON 研究设计的用户。"
    ),
    format_func=lambda mode: {'快速引导': '设计因果研究 · 引导确认',
                              '自由提问': '探索研究线索 · 自然语言',
                              '专业契约': '专业配置 · 研究契约'}[mode],
)
if analysis_mode == '自由提问':
    st.info('从一个具体问题开始，例如：按已有分组比较结果指标并绘图，或解释某个字段的背景。探索结果不自动构成因果结论。')
elif analysis_mode == '快速引导':
    st.caption('已有明确的影响问题？结合研究背景提出方法候选，再由你确认变量、假设和执行配置。只想比较或绘图，可切换到“探索研究线索”。')

question = ""
guided_contract = None
guided_compatibility = None
professional_contract = None
professional_contract_error = None

if analysis_mode == "快速引导":
    st.subheader("研究目标与方案")
    st.caption("操作顺序：① 上传并确认数据 → ② 描述目标、补充背景 → ③ AI 提议与澄清 → ④ 确认变量和方案 → ⑤ 查看结果与下载简报")
    st.caption(
        "先用普通语言描述目的，再从真实 CSV 字段中确认变量。"
        "系统只提供候选，不会替你确认因果角色。"
    )
    question = st.text_area(
        "你想研究什么影响？",
        placeholder="例如：分析美国十年期 TIPS 收益率变化对黄金收益率的影响",
        height=100,
        key="guided_question",
    )

    ai_value = None
    if uploaded_file is not None and "preview_df" in locals():
        from causalens.ui.ai_planning import render_ai_planning
        with st.expander('AI 辅助规划：补充背景、获取建议、回答澄清问题', expanded=True):
            ai_value = render_ai_planning(uploaded_file.getvalue(), question, preview_df)

    if uploaded_file is None:
        st.info("上传 CSV 后，这里会显示字段候选和方法适配检查。")
    elif "preview_df" in locals():
        suggestions = suggest_causal_roles(question, preview_df)
        if ai_value:
            for role in ("treatment", "outcome", "adjustment", "time"):
                suggestions[role] = list(
                    dict.fromkeys(
                        [
                            *(ai_value.get(role) or []),
                            *(suggestions.get(role) or []),
                        ]
                    )
                )
        columns = [str(column) for column in preview_df.columns]
        empty_option = "— 请选择 —"

        st.markdown("#### 确认方案 · 1. 变量含义")
        mapping_col1, mapping_col2 = st.columns(2)
        treatment_options = [empty_option, *columns]
        outcome_options = [empty_option, *columns]
        suggested_treatment = (suggestions.get("treatment") or [None])[0]
        suggested_outcome = (suggestions.get("outcome") or [None])[0]
        with mapping_col1:
            treatment = st.selectbox(
                "你想改变或比较哪个因素？（Treatment）",
                treatment_options,
                index=(
                    treatment_options.index(suggested_treatment)
                    if suggested_treatment in treatment_options
                    else 0
                ),
                help="候选来自问题文字和字段特征，必须由你确认。",
            )
        with mapping_col2:
            outcome = st.selectbox(
                "你希望观察哪个结果？（Outcome）",
                outcome_options,
                index=(
                    outcome_options.index(suggested_outcome)
                    if suggested_outcome in outcome_options
                    else 0
                ),
                help="如果存在多个收益率或结果字段，请依据数据字典选择。",
            )
        treatment = None if treatment == empty_option else treatment
        outcome = None if outcome == empty_option else outcome

        st.markdown("#### 确认方案 · 2. 研究设计与背景")
        design_labels = {
            "横截面调整（0/1 处理）": "cross_sectional",
            "双重差分 / Event Study": "difference_in_differences",
            "工具变量": "instrumental_variable",
            "断点回归": "regression_discontinuity",
            "合成控制": "synthetic_control",
            "连续冲击时间序列（Local Projection）": "time_series",
        }
        design_type, estimand, selected_time_column = render_research_advisor(
            preview_df, treatment, outcome, design_labels
        )
        if design_type is None:
            st.info("下一步：在上方补充场景并确认方案。确认前无需填写控制变量或运行分析。若预检提示风险，请先确定数据处理方案，系统不会自动转换。")
            st.stop()
        st.markdown("#### 补充研究背景，再开始分析")
        st.caption("控制变量不是越多越好。不确定时先查数据说明；不要把所有候选全部选入。高级参数可保留默认值，但应在报告中说明。")

        reserved_primary_columns = {
            value
            for value in (treatment, outcome, selected_time_column)
            if value
        }
        candidate_adjustments = [
            column
            for column in suggestions.get("adjustment", [])
            if column not in reserved_primary_columns
        ]
        confounders, controls_confirmed = render_control_selection(
            uploaded_file.getvalue(), columns, reserved_primary_columns,
            treatment, outcome, design_type,
        )
        if not controls_confirmed:
            st.info("请先确认控制方案，再填写研究背景并开始分析。")
            st.stop()

        context_identity = analysis_identity(uploaded_file.getvalue(), 'context', question,
                                             {'t': treatment, 'y': outcome, 'design': design_type})['signature']
        target_population, unit_of_analysis, outcome_horizon = render_context_inputs(ai_value, context_identity)

        effect_modifiers: list[str] = []
        instruments: list[str] = []
        negative_controls: list[str] = []
        post_column = None
        event_time_column = None
        treatment_start_column = None
        unit_id_column = None
        running_variable = None
        cutoff = None
        intervention_time = None
        treated_unit = None
        max_horizon = 5
        treatment_lags = 1
        outcome_lags = 1
        control_lag = 1
        cumulative_outcome = True
        hac_lags = None

        with st.expander("设计专属字段与高级选项"):
            if estimand == "CATE":
                effect_modifiers = st.multiselect(
                    "处理前效应修饰变量",
                    [
                        column
                        for column in columns
                        if column not in reserved_primary_columns
                    ],
                )
            if design_type == "instrumental_variable":
                instruments = st.multiselect(
                    "工具变量",
                    [
                        column
                        for column in columns
                        if column not in reserved_primary_columns
                    ],
                )
            elif design_type == "difference_in_differences":
                did_mode = st.radio(
                    "DiD 类型",
                    ["两期 DiD", "Event Study", "错位处理 DiD"],
                    horizontal=True,
                )
                if did_mode == "两期 DiD":
                    post_column = st.selectbox("Post 字段", [empty_option, *columns])
                    post_column = None if post_column == empty_option else post_column
                elif did_mode == "Event Study":
                    event_time_column = st.selectbox(
                        "相对事件时间字段", [empty_option, *columns]
                    )
                    event_time_column = (
                        None if event_time_column == empty_option else event_time_column
                    )
                else:
                    treatment_start_column = st.selectbox(
                        "首次处理时期字段", [empty_option, *columns]
                    )
                    treatment_start_column = (
                        None
                        if treatment_start_column == empty_option
                        else treatment_start_column
                    )
                    unit_id_column = st.selectbox(
                        "面板单位 ID", [empty_option, *columns]
                    )
                    unit_id_column = (
                        None if unit_id_column == empty_option else unit_id_column
                    )
            elif design_type == "regression_discontinuity":
                running_variable = st.selectbox(
                    "阈值运行变量", [empty_option, *columns]
                )
                running_variable = (
                    None if running_variable == empty_option else running_variable
                )
                cutoff = st.number_input("处理阈值", value=0.0)
            elif design_type == "synthetic_control":
                unit_id_column = st.selectbox("单位 ID", [empty_option, *columns])
                unit_id_column = (
                    None if unit_id_column == empty_option else unit_id_column
                )
                intervention_time = st.number_input("首次干预时期", value=0.0)
                treated_unit_text = st.text_input("处理单位（可留空自动识别）")
                treated_unit = treated_unit_text.strip() or None
            elif design_type == "time_series":
                max_horizon = int(
                    st.number_input(
                        "最大响应期数",
                        min_value=0,
                        max_value=60,
                        value=5,
                        step=1,
                    )
                )
                treatment_lags = int(
                    st.number_input("Treatment 滞后阶数", 1, 20, 1, 1)
                )
                outcome_lags = int(
                    st.number_input("Outcome 滞后阶数", 1, 20, 1, 1)
                )
                control_lag = int(
                    st.number_input(
                        "控制变量滞后阶数",
                        1,
                        20,
                        1,
                        1,
                        help="默认只使用前一期控制变量，避免把同期中介直接控制掉。",
                    )
                )
                cumulative_outcome = st.checkbox(
                    "报告截至各期的累计结果响应",
                    value=True,
                )
                automatic_hac = st.checkbox("自动选择 HAC 最大滞后", value=True)
                if not automatic_hac:
                    hac_lags = int(st.number_input("HAC 最大滞后", 1, 60, 5, 1))
            advanced_reserved_columns = {
                *reserved_primary_columns,
                *(
                    value
                    for value in (
                        post_column,
                        event_time_column,
                        treatment_start_column,
                        unit_id_column,
                        running_variable,
                    )
                    if value
                ),
            }
            negative_controls = st.multiselect(
                "负对照结果变量（可选）",
                [
                    column
                    for column in columns
                    if column not in advanced_reserved_columns
                ],
            )

        normalized_roles, role_conflicts = normalize_guided_roles(
            treatment=treatment or "",
            outcome=outcome or "",
            confounders=confounders,
            effect_modifiers=effect_modifiers,
            instruments=instruments,
            negative_control_outcomes=negative_controls,
            post_column=post_column,
            event_time_column=event_time_column,
            time_column=selected_time_column,
            treatment_start_column=treatment_start_column,
            unit_id_column=unit_id_column,
            running_variable=running_variable,
        )
        confounders = normalized_roles["confounders"]
        effect_modifiers = normalized_roles["effect_modifiers"]
        instruments = normalized_roles["instruments"]
        negative_controls = normalized_roles["negative_control_outcomes"]
        if role_conflicts:
            excluded_text = "；".join(
                f"{role}: {', '.join(values)}"
                for role, values in role_conflicts.items()
            )
            st.warning(
                "已自动从因果变量角色中排除与 Treatment、Outcome、时间或"
                f"设计专属字段重复的选择：{excluded_text}。"
            )

        guided_compatibility = assess_guided_compatibility(
            preview_df,
            treatment=treatment,
            outcome=outcome,
            design_type=design_type,
            time_column=selected_time_column,
        )
        st.markdown("#### 确认方案 · 3. 执行前检查")
        detected = guided_compatibility.get("detected") or {}
        if detected.get("treatment_kind"):
            st.write("**处理变量类型：**", detected["treatment_kind"])
        for warning in guided_compatibility.get("warnings") or []:
            st.warning(warning)
        for issue in guided_compatibility.get("blocking_issues") or []:
            st.error(issue)
        if guided_compatibility.get("ready"):
            st.success("当前变量类型与所选方法在技术上兼容，可以继续校验研究契约。")
        else:
            with st.expander("可以怎样调整？", expanded=True):
                for alternative in guided_compatibility.get("alternatives") or []:
                    st.markdown(f"- {alternative}")

        if treatment and outcome and design_type and estimand:
            guided_contract = build_guided_contract(
                treatment=treatment,
                outcome=outcome,
                confounders=confounders,
                estimand=estimand,
                design_type=design_type,
                target_population=target_population,
                unit_of_analysis=unit_of_analysis,
                outcome_horizon=outcome_horizon,
                effect_modifiers=effect_modifiers,
                instruments=instruments,
                negative_control_outcomes=negative_controls,
                post_column=post_column,
                event_time_column=event_time_column,
                time_column=(
                    selected_time_column
                    if design_type
                    in {"difference_in_differences", "synthetic_control", "time_series"}
                    else None
                ),
                treatment_start_column=treatment_start_column,
                unit_id_column=unit_id_column,
                running_variable=running_variable,
                cutoff=cutoff,
                intervention_time=intervention_time,
                treated_unit=treated_unit,
                max_horizon=max_horizon,
                treatment_lags=treatment_lags,
                outcome_lags=outcome_lags,
                control_lag=control_lag,
                cumulative_outcome=cumulative_outcome,
                hac_lags=hac_lags,
            )
            with st.expander("预览即将校验的研究契约"):
                st.json(guided_contract)

    with st.expander("查看提问示例"):
        st.code(
            "广告曝光是否影响用户留存？\n"
            "政策实施是否提高了处理地区的就业率？\n"
            "录取分数线附近，被录取是否影响后续成绩？\n"
            "TIPS 收益率变化是否影响黄金收益率？",
            language="text",
        )

elif analysis_mode == "专业契约":
    st.subheader("专业因果契约")
    st.caption("直接填写 JSON。仍会经过字段、数据质量、识别和准备度检查。")
    professional_text = st.text_area(
        "因果研究契约 JSON",
        value=json.dumps(
            {
                "treatment": "",
                "outcome": "",
                "estimand": "ATE",
                "design_type": "cross_sectional",
                "target_population": "",
                "unit_of_analysis": "",
                "outcome_horizon": "",
                "confounders": [],
                "effect_modifiers": [],
                "mediators": [],
                "colliders": [],
                "instruments": [],
                "negative_control_outcomes": [],
                "dag_edges": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        height=360,
    )
    professional_contract, professional_contract_error = parse_professional_contract(
        professional_text
    )
    if professional_contract_error:
        st.error(professional_contract_error)
    question = "执行网页中填写的显式因果研究契约"

else:
    question = st.text_area(
        "输入你的分析问题",
        placeholder="例如：study_hours 和 score 是否存在显著相关关系？",
        height=120,
    )
    st.caption(
        "因果问题若使用自由提问，请明确写出 treatment、outcome、confounders、"
        "目标人群、分析单位和结果时间窗；不确定时建议使用“快速引导”。"
    )


# =========================
# 分析按钮
# =========================

context_missing = missing_context(guided_contract) if analysis_mode == '快速引导' else []
if context_missing:
    st.warning('请先补充研究背景：' + '、'.join(context_missing) + '。灰色示例不算填写内容。')
analyze_clicked = st.button("开始分析", type="primary", disabled=bool(context_missing))


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

if "current_contract" not in st.session_state:
    st.session_state.current_contract = None

if "current_execution_mode" not in st.session_state:
    st.session_state.current_execution_mode = "free_text"


# =========================
# 执行新的 Agent 请求
# =========================

if analyze_clicked:

    if uploaded_file is None:
        st.warning(
            "请先上传一个 CSV 文件。"
        )

    elif analysis_mode == "自由提问" and not question.strip():
        st.warning(
            "请输入需要分析的问题。"
        )

    elif analysis_mode == "快速引导" and guided_contract is None:
        st.warning("请先确认处理变量和结果变量。")

    elif (
        analysis_mode == "快速引导"
        and guided_compatibility
        and guided_compatibility.get("ready") is not True
    ):
        st.warning(
            "当前配置未通过方法适配检查。请根据页面提示修改变量或研究设计；"
            "系统没有调用模型，也没有执行估计。"
        )

    elif analysis_mode == "专业契约" and professional_contract_error:
        st.warning("请先修正契约 JSON。")

    elif analysis_mode == "专业契约" and professional_contract is None:
        st.warning("请输入有效的契约 JSON。")

    else:
        dataset_path = save_uploaded_csv(
            uploaded_file
        )
        resolved_question = (
            question.strip()
            or "通过网页快速引导填写的显式因果研究契约"
        )

        # 保存当前任务，
        # 后续人工审批时继续使用。
        st.session_state.current_dataset_path = str(
            dataset_path
        )

        st.session_state.current_question = (
            resolved_question
        )

        st.session_state.causal_cancelled = False

        selected_contract = (
            guided_contract
            if analysis_mode == "快速引导"
            else professional_contract
            if analysis_mode == "专业契约"
            else None
        )
        st.session_state.current_contract = selected_contract
        import copy
        snapshot = st.session_state.get('active_planning_snapshot') if analysis_mode == '快速引导' else None
        st.session_state.submitted_planning = copy.deepcopy(snapshot) if snapshot else None
        if st.session_state.submitted_planning:
            st.session_state.submitted_planning.update(question=question, final_contract=copy.deepcopy(selected_contract))
        st.session_state.submitted_identity = analysis_identity(
            uploaded_file.getvalue(), analysis_mode, question, selected_contract
        )
        st.session_state.current_execution_mode = (
            "explicit_contract" if selected_contract is not None else "free_text"
        )

        with st.spinner(
            "CausaLens 正在分析..."
        ):
            try:
                if selected_contract is not None:
                    contract_result = run_explicit_causal_contract(
                        dataset_path=str(dataset_path),
                        contract=selected_contract,
                        approved=False,
                        result_cache_dir=str(PROJECT_ROOT / "storage" / "causal_cache"),
                    )
                    final_state = contract_result_to_agent_state(
                        contract_result,
                        dataset_path=str(dataset_path),
                        question=resolved_question,
                    )
                else:
                    final_state = run_causalens_graph(
                        dataset_path=str(dataset_path),
                        question=resolved_question,
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
                attach_planning(final_state, st.session_state.get('submitted_planning'))
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
    submitted = st.session_state.get('submitted_identity')
    active_contract = guided_contract if analysis_mode == '快速引导' else professional_contract if analysis_mode == '专业契约' else None
    current_identity = analysis_identity(uploaded_file.getvalue(), analysis_mode, question, active_contract) if uploaded_file else None
    if not submitted or not current_identity or submitted['signature'] != current_identity['signature']:
        st.warning('当前输入与上次分析不一致（或旧结果没有版本记录）。旧结果与执行确认已暂停展示，请点击“开始分析”重新校验。已下载报告不受影响。')
        st.stop()
    st.caption(f'本次结果对应的数据版本：{submitted["data_sha256"][:12]}；配置标识：{submitted["signature"][:12]}。与当前输入一致。')

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

        approval_question = final_state.get("causal_question") or {}
        approval_estimand = (
            str(approval_question.get("estimand") or "ATE")
            if isinstance(approval_question, dict)
            else "ATE"
        )
        approval_design_type = (
            str(approval_question.get("design_type") or "cross_sectional")
            if isinstance(approval_question, dict)
            else "cross_sectional"
        )
        approval_estimator_label = (
            "TimeSeriesLocalProjection（连续冲击 IRF）"
            if approval_design_type == "time_series"
            else "StaggeredAdoptionDID"
            if approval_design_type == "difference_in_differences"
            and isinstance(approval_question, dict)
            and approval_question.get("time_column")
            and approval_question.get("treatment_start_column")
            else "Event Study"
            if approval_design_type == "difference_in_differences"
            and isinstance(approval_question, dict)
            and approval_question.get("event_time_column")
            else "Difference-in-Differences"
            if approval_design_type == "difference_in_differences"
            else "Sharp Regression Discontinuity"
            if approval_design_type == "regression_discontinuity"
            else "Synthetic Control"
            if approval_design_type == "synthetic_control"
            else "Instrumental Variables / 2SLS"
            if approval_design_type == "instrumental_variable"
            else "CausalForestDML"
            if approval_estimand == "CATE"
            else "AIPW + LinearDML"
        )

        st.warning(
            "因果分析准备度检查已经通过，"
            f"但 {approval_estimator_label} 尚未执行。"
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

        causal_question = final_state.get(
            "causal_question",
            {},
        )
        if not isinstance(causal_question, dict):
            causal_question = {}

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

        variable_roles = causal_question.get("variable_roles") or {}
        if isinstance(variable_roles, dict):
            st.write(
                "效应修饰变量：",
                variable_roles.get("effect_modifiers") or [],
            )
            st.write(
                "负对照结果变量：",
                variable_roles.get("negative_control_outcomes") or [],
            )
            st.write(
                "工具变量：",
                variable_roles.get("instruments") or [],
            )

        st.markdown("#### 因果问题契约")
        contract_col1, contract_col2 = st.columns(2)
        with contract_col1:
            st.write(
                "**Estimand：**",
                causal_question.get("estimand", "ATE"),
            )
            st.write(
                "**目标人群：**",
                causal_question.get("target_population") or "未明确",
            )
            st.write(
                "**研究设计：**",
                causal_question.get("design_type", "cross_sectional"),
            )
        with contract_col2:
            st.write(
                "**分析单位：**",
                causal_question.get("unit_of_analysis") or "未明确",
            )
            st.write(
                "**结果时间窗：**",
                causal_question.get("outcome_horizon") or "未明确",
            )
            if causal_question.get("design_type") == "difference_in_differences":
                st.write("**Post 字段：**", causal_question.get("post_column"))
                st.write(
                    "**Event time 字段：**",
                    causal_question.get("event_time_column"),
                )
                st.write(
                    "**参考期：**",
                    causal_question.get("reference_period", -1),
                )
                st.write("**时期字段：**", causal_question.get("time_column"))
                st.write(
                    "**首次处理时期字段：**",
                    causal_question.get("treatment_start_column"),
                )
                st.write(
                    "**Unit ID：**",
                    causal_question.get("unit_id_column") or "重复横截面",
                )
            elif causal_question.get("design_type") == "regression_discontinuity":
                st.write(
                    "**Running variable：**",
                    causal_question.get("running_variable"),
                )
                st.write(
                    "**Cutoff：**",
                    causal_question.get("cutoff"),
                )
                st.write(
                    "**Cutoff 上方接受处理：**",
                    causal_question.get("treatment_above_cutoff", True),
                )
                st.write(
                    "**局部带宽：**",
                    causal_question.get("bandwidth") or "自动选择",
                )
            elif causal_question.get("design_type") == "instrumental_variable":
                st.write(
                    "**工具变量：**",
                    (causal_question.get("variable_roles") or {}).get(
                        "instruments"
                    )
                    or [],
                )
            elif causal_question.get("design_type") == "synthetic_control":
                st.write(
                    "**干预时点：**",
                    causal_question.get("intervention_time"),
                )
                st.write(
                    "**处理单位：**",
                    causal_question.get("treated_unit") or "从 treatment 推断",
                )

        missing_design_fields = final_state.get(
            "missing_design_fields",
            [],
        )
        if missing_design_fields:
            st.warning(
                "以下研究设计信息尚未明确："
                + "、".join(
                    str(item) for item in missing_design_fields
                )
                + "。继续执行表示你接受报告将明确记录这些限制。"
            )

        identification = final_state.get(
            "causal_identification",
            {},
        )
        if isinstance(identification, dict) and identification:
            identification_status = identification.get(
                "status",
                "not_assessed",
            )
            if identification_status == "identified":
                st.success("DAG 后门识别检查已通过。")
            elif (
                identification_status == "not_assessed"
                and causal_question.get("design_type")
                == "difference_in_differences"
            ):
                st.warning(
                    "DiD 识别依赖平行趋势、无提前反应及无组别特异"
                    "同期冲击；两期数据不能自动证明这些假设。"
                )
            elif identification_status == "not_assessed":
                st.warning(
                    "未提供 DAG：当前只能确认技术可执行性，"
                    "尚不能验证调整集是否阻断全部后门路径。"
                )

        diagnostics = final_state.get(
            "causal_diagnostics",
            {},
        )
        if isinstance(diagnostics, dict) and diagnostics:
            diagnostic_type = diagnostics.get("analysis_type")
            is_did_diagnostics = diagnostic_type == "did_design_diagnostics"
            is_event_diagnostics = (
                diagnostic_type == "event_study_design_diagnostics"
            )
            is_staggered_diagnostics = (
                diagnostic_type == "staggered_did_design_diagnostics"
            )
            st.markdown(
                "#### DiD 四单元格与面板诊断"
                if is_did_diagnostics
                else "#### Event Study 时期与面板诊断"
                if is_event_diagnostics
                else "#### 错位处理 Cohort-Time 诊断"
                if is_staggered_diagnostics
                else "#### 重叠性与平衡性诊断"
            )
            if is_did_diagnostics:
                design_summary = diagnostics.get("design_summary") or {}
                st.write("**四单元格样本量：**", design_summary.get("cell_counts"))
                st.write("**四单元格均值：**", design_summary.get("cell_means"))
                if design_summary.get("total_units") is not None:
                    st.write(
                        "**完整两期单位：**",
                        design_summary.get("complete_two_period_units"),
                        "/",
                        design_summary.get("total_units"),
                    )
                for warning in diagnostics.get("warnings", []) or []:
                    st.warning(str(warning))
            elif is_event_diagnostics:
                design_summary = diagnostics.get("design_summary") or {}
                st.write("**相对时期：**", design_summary.get("periods"))
                st.write("**处理前时期：**", design_summary.get("pre_periods"))
                st.write("**处理后时期：**", design_summary.get("post_periods"))
                st.write("**组别×时期样本量：**", design_summary.get("cell_counts"))
                for warning in diagnostics.get("warnings", []) or []:
                    st.warning(str(warning))
            elif is_staggered_diagnostics:
                design_summary = diagnostics.get("design_summary") or {}
                st.write("**处理 cohorts：**", design_summary.get("cohorts"))
                st.write("**各 cohort 单位数：**", design_summary.get("cohort_counts"))
                st.write(
                    "**永不处理单位数：**",
                    design_summary.get("never_treated_units"),
                )
                st.write(
                    "**可估计 cohort-time 单元数：**",
                    design_summary.get("supported_group_time_cells"),
                )
                for warning in diagnostics.get("warnings", []) or []:
                    st.warning(str(warning))
            else:
                propensity = diagnostics.get("propensity_summary") or {}
                weights = diagnostics.get("weight_summary") or {}
                diagnostic_columns = st.columns(3)
                with diagnostic_columns[0]:
                    st.metric(
                        "共同支持区间",
                        str(propensity.get("common_support")),
                    )
                with diagnostic_columns[1]:
                    extreme_fraction = propensity.get("extreme_fraction")
                    st.metric(
                        "极端倾向比例",
                        (
                            f"{float(extreme_fraction):.1%}"
                            if extreme_fraction is not None
                            else "未知"
                        ),
                    )
                with diagnostic_columns[2]:
                    effective_sample_size = weights.get(
                        "effective_sample_size"
                    )
                    st.metric(
                        "加权有效样本量",
                        (
                            f"{float(effective_sample_size):.1f}"
                            if effective_sample_size is not None
                            else "未知"
                        ),
                    )

                balance_rows = diagnostics.get("balance") or []
                if balance_rows:
                    st.dataframe(
                        pd.DataFrame(balance_rows).rename(
                            columns={
                                "variable": "协变量",
                                "smd_before": "IPW 前 SMD",
                                "smd_after": "IPW 后 SMD",
                            }
                        ),
                        width='stretch',
                        hide_index=True,
                    )

                for warning in diagnostics.get("warnings", []) or []:
                    st.warning(str(warning))

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
                f"继续执行 {approval_estimator_label}",
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
                    f"正在执行 {approval_estimator_label} 因果效应估计..."
                ):
                    try:
                        if (
                            st.session_state.current_execution_mode
                            == "explicit_contract"
                            and st.session_state.current_contract
                        ):
                            approved_result = run_explicit_causal_contract(
                                dataset_path=dataset_path,
                                contract=st.session_state.current_contract,
                                approved=True,
                                result_cache_dir=str(
                                    PROJECT_ROOT / "storage" / "causal_cache"
                                ),
                            )
                            approved_state = contract_result_to_agent_state(
                                approved_result,
                                dataset_path=dataset_path,
                                question=saved_question,
                            )
                        else:
                            approved_state = run_causalens_graph(
                                dataset_path=dataset_path,
                                question=saved_question,
                                approved=True,
                            )

                    except Exception as exc:
                        st.error(
                            "因果估计器执行过程中"
                            "发生未处理异常。"
                        )

                        with st.expander(
                            "开发调试信息"
                        ):
                            st.exception(exc)

                    else:
                        attach_planning(approved_state, st.session_state.get('submitted_planning'))
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
                f"{approval_estimator_label} 没有执行。"
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

        plain_summary = summarize_time_series(final_state)
        if plain_summary:
            st.markdown(plain_summary)
            render_comparison(final_state)
            with st.expander("查看统计表、诊断与模型参数", expanded=False):
                render_causal_estimator_results(final_state)
        else:
            render_causal_estimator_results(final_state)

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
                    width='stretch',
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
        "AIPW 与 LinearDML 未执行，"
        "也没有生成因果效应估计。"
    )

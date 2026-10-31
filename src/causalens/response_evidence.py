from __future__ import annotations

from typing import Any

from causalens.schemas import ToolResult


def _format_number(value: Any) -> str:
    """
    将数值转为适合 CLI 展示的简短文本。
    """
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def _format_columns(
    column_names: list[Any],
    max_columns: int = 6,
) -> str:
    """
    只展示前几个字段名，避免 CLI 表格过长。
    """
    names = [str(column_name) for column_name in column_names]

    if not names:
        return "未返回字段信息"

    visible_names = names[:max_columns]
    text = "、".join(visible_names)

    if len(names) > max_columns:
        text += "……"

    return text


def _build_profile_evidence(
    result: dict[str, Any],
) -> str:
    rows = result.get("rows", "未知")
    columns = result.get("columns", "未知")

    column_names = result.get("column_names", [])
    columns_text = _format_columns(column_names)

    return (
        f"数据集规模：{rows} 行 × {columns} 列；"
        f"字段：{columns_text}"
    )


def _build_sql_evidence(
    result: dict[str, Any],
) -> str:
    returned_row_count = result.get(
        "returned_row_count",
        result.get("row_count", "未知"),
    )

    columns = result.get("columns", [])
    columns_text = _format_columns(columns)

    truncated = result.get("truncated", False)
    truncation_text = "；结果已截断" if truncated else ""

    return (
        f"SQL 返回 {returned_row_count} 行；"
        f"结果字段：{columns_text}"
        f"{truncation_text}"
    )


def _build_chart_evidence(
    result: dict[str, Any],
) -> str:
    chart_type = result.get("chart_type", "未知")
    plotted_row_count = result.get("plotted_row_count")

    parts = [f"生成 {chart_type} 图"]

    if plotted_row_count is not None:
        parts.append(f"参与绘图数据：{plotted_row_count} 条")

    numeric_summary = result.get("numeric_summary")

    if isinstance(numeric_summary, dict):
        numeric_parts = []

        for key in ["count", "mean", "median", "min", "max"]:
            if key in numeric_summary:
                numeric_parts.append(
                    f"{key}={_format_number(numeric_summary[key])}"
                )

        if numeric_parts:
            parts.append(
                "统计摘要：" + "，".join(numeric_parts)
            )

    relationship_summary = result.get("relationship_summary")

    if isinstance(relationship_summary, dict):
        relationship_parts = []

        for key in [
            "point_count",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "pearson_correlation",
        ]:
            if key in relationship_summary:
                relationship_parts.append(
                    f"{key}="
                    f"{_format_number(relationship_summary[key])}"
                )

        if relationship_parts:
            parts.append(
                "关系摘要：" + "，".join(relationship_parts)
            )

    chart_path = result.get("chart_path")

    if chart_path:
        parts.append(f"图表：{chart_path}")

    return "\n".join(parts)


def _build_retrieve_docs_evidence(result: dict) -> str:
    query = result.get("query", "")
    retrieved_items = result.get("results", [])

    if not retrieved_items:
        return "未检索到相关项目文档片段。"

    parts = [
        f"检索问题：{query}",
        f"返回片段数：{len(retrieved_items)}",
    ]

    sources = []

    for item in retrieved_items:
        source = item.get("source", "")
        section = item.get("section", "")
        chunk_id = item.get("chunk_id", "")
        distance = item.get("distance", None)

        if not source:
            continue

        source_label = source
        if section:
            source_label = f"{source}，{section}"

        if distance is None:
            sources.append(f"{source_label} ({chunk_id})")
        else:
            sources.append(f"{source_label} ({chunk_id}, distance={distance:.4f})")

    if sources:
        parts.append("来源片段：")
        parts.extend(f"- {source}" for source in sources)

    return "\n".join(parts)


def _build_causal_estimation_evidence(result: dict[str, Any]) -> str:
    estimand = str(result.get("estimand") or "ATE")
    parts = [
        f"主估计器：{result.get('primary_estimator', '未知')}",
        f"样本量：{result.get('sample_size', '未知')}",
    ]
    for estimate in result.get("estimates", []) or []:
        if not isinstance(estimate, dict):
            continue
        parts.append(
            "{name}: {estimand}={ate}, 95% CI={interval}".format(
                name=estimate.get("estimator", "未知"),
                estimand=estimand,
                ate=_format_number(estimate.get("ate")),
                interval=estimate.get("confidence_interval"),
            )
        )
        sensitivity = estimate.get("sensitivity_analysis") or {}
        if sensitivity.get("available", True):
            parts.append(
                "未观测混杂敏感性：RV(point)="
                + _format_number(
                    sensitivity.get("robustness_value_point_estimate")
                )
                + "，RV(significance)="
                + _format_number(
                    sensitivity.get("robustness_value_significance")
                )
                + "，风险分级="
                + str(sensitivity.get("unmeasured_confounding_risk"))
            )
            benchmark = sensitivity.get("observed_confounder_benchmark") or {}
            if benchmark.get("available"):
                parts.append(
                    "最强已观测混杂基准="
                    + str(benchmark.get("strongest_confounder"))
                    + "（equivalent RV="
                    + _format_number(
                        benchmark.get("strongest_equivalent_robustness_value")
                    )
                    + "，assessment="
                    + str(benchmark.get("assessment"))
                    + "）"
                )
        elif sensitivity:
            parts.append(
                "未观测混杂敏感性：不可用（"
                + str(sensitivity.get("reason") or "未提供原因")
                + "）"
            )
        if estimate.get("estimator") == "TwoStageLeastSquares":
            first_stage = estimate.get("first_stage") or {}
            endogeneity = estimate.get("endogeneity_test") or {}
            overidentification = estimate.get("overidentification_test") or {}
            parts.append(
                "IV 第一阶段：F="
                + _format_number(first_stage.get("f_statistic"))
                + "，partial R²="
                + _format_number(first_stage.get("partial_r2"))
                + "，weak="
                + str(first_stage.get("weak_instrument_warning"))
            )
            parts.append(
                "内生性检验 p="
                + _format_number(endogeneity.get("p_value"))
            )
            if overidentification.get("available"):
                parts.append(
                    "Hansen J p="
                    + _format_number(overidentification.get("p_value"))
                    + "，passed="
                    + str(overidentification.get("passed"))
                )
        if estimate.get("estimator") == "SharpRegressionDiscontinuity":
            density = estimate.get("density_diagnostic") or {}
            continuity = estimate.get("covariate_continuity") or {}
            bandwidth = estimate.get("bandwidth_sensitivity") or {}
            donut = estimate.get("donut_sensitivity") or {}
            support = estimate.get("local_support") or {}
            parts.append(
                "RDD：running variable="
                + str(estimate.get("running_variable"))
                + "，cutoff="
                + _format_number(estimate.get("cutoff"))
                + "，bandwidth="
                + _format_number(estimate.get("bandwidth"))
            )
            parts.append(
                "断点两侧样本="
                f"{support.get('left_count')}/{support.get('right_count')}，"
                "density p="
                + _format_number(density.get("p_value"))
                + "，covariates passed="
                + str(continuity.get("all_passed"))
            )
            parts.append(
                "带宽方向稳定="
                + str(bandwidth.get("direction_stable"))
                + "，donut estimate="
                + _format_number(donut.get("estimate"))
            )
        if estimate.get("estimator") == "SyntheticControl":
            fit = estimate.get("fit_diagnostics") or {}
            placebo = estimate.get("placebo_inference") or {}
            weights = estimate.get("donor_weights") or []
            parts.append(
                "合成控制：treated unit="
                + str(estimate.get("treated_unit"))
                + "，intervention time="
                + _format_number(estimate.get("intervention_time"))
            )
            parts.append(
                "pre RMSPE="
                + _format_number(fit.get("pre_rmspe"))
                + "，post/pre RMSPE ratio="
                + _format_number(fit.get("post_pre_rmspe_ratio"))
                + "，placebo p="
                + _format_number(placebo.get("p_value"))
            )
            parts.append(f"非零供体权重：{weights}")
        if estimate.get("estimator") == "TimeSeriesLocalProjection":
            configuration = estimate.get("model_configuration") or {}
            stationarity = estimate.get("stationarity") or {}
            placebo = estimate.get("placebo_test") or {}
            residual = estimate.get("residual_diagnostics") or {}
            parts.append(
                "连续冲击 IRF（每 1 个原始单位，不是二元处理 ATE）："
                + str(configuration.get("response"))
            )
            parts.append(
                "ADF treatment/outcome p="
                + _format_number(
                    (stationarity.get("treatment") or {}).get("p_value")
                )
                + "/"
                + _format_number(
                    (stationarity.get("outcome") or {}).get("p_value")
                )
                + "，future-treatment placebo p="
                + _format_number(placebo.get("p_value"))
                + "，residual autocorrelation warning="
                + str(residual.get("residual_autocorrelation_warning"))
            )
            parts.append(f"动态响应：{estimate.get('dynamic_effects')}")
        if estimate.get("estimator") == "DifferenceInDifferences":
            parts.append(
                "未调整四单元格 DiD："
                + _format_number(estimate.get("unadjusted_did"))
            )
            parts.append(f"四单元格均值：{estimate.get('cell_means')}")
        if estimate.get("estimator") == "EventStudy":
            pretrend = estimate.get("pretrend_test") or {}
            parts.append(
                "联合预趋势检验 p 值："
                + _format_number(pretrend.get("p_value"))
            )
            parts.append(f"动态效应：{estimate.get('dynamic_effects')}")
            if estimate.get("chart_path"):
                parts.append(f"动态图：{estimate.get('chart_path')}")
        if estimate.get("estimator") == "StaggeredAdoptionDID":
            pretrend = estimate.get("pretrend_test") or {}
            configuration = estimate.get("model_configuration") or {}
            parts.append(f"调整模式：{configuration.get('adjustment')}")
            if configuration.get("nuisance_diagnostics"):
                parts.append(
                    "Nuisance 诊断："
                    f"{configuration.get('nuisance_diagnostics')}"
                )
            placebo = estimate.get("placebo_test") or {}
            sensitivity = estimate.get("cohort_sensitivity") or {}
            if placebo:
                parts.append(
                    "提前一期 Placebo p 值："
                    + _format_number(placebo.get("p_value"))
                )
            if sensitivity:
                parts.append(
                    "逐 cohort 剔除最大偏移："
                    + _format_number(sensitivity.get("max_absolute_deviation"))
                )
            parts.append(
                "cohort-time ATT 单元数："
                f"{len(estimate.get('group_time_effects') or [])}"
            )
            parts.append(
                "联合预趋势检验 p 值："
                + _format_number(pretrend.get("p_value"))
            )
            parts.append(f"动态效应：{estimate.get('dynamic_effects')}")
    comparison = result.get("comparison") or {}
    if "absolute_ate_difference" in comparison:
        parts.append(
            "估计值绝对差："
            + _format_number(comparison["absolute_ate_difference"])
        )
    if "confidence_intervals_overlap" in comparison:
        parts.append(
            "置信区间重叠："
            f"{comparison['confidence_intervals_overlap']}"
        )
    negative_controls = result.get("negative_control_summary") or {}
    if negative_controls.get("requested_outcomes"):
        parts.append(
            "负对照 outcome："
            f"{negative_controls.get('passed_count')}/"
            f"{negative_controls.get('result_count')} 通过"
        )
        for row in negative_controls.get("results", []) or []:
            if not isinstance(row, dict):
                continue
            parts.append(
                "负对照 {outcome}: effect={effect}, 95% CI={interval}, "
                "passed={passed}".format(
                    outcome=row.get("outcome"),
                    effect=_format_number(row.get("estimate")),
                    interval=row.get("confidence_interval"),
                    passed=row.get("passed"),
                )
            )
    permutation = result.get("permutation_placebo") or {}
    if permutation.get("available"):
        parts.append(
            "残差化伪 treatment：observed coefficient="
            + _format_number(permutation.get("observed_adjusted_coefficient"))
            + "，placebo mean="
            + _format_number(permutation.get("placebo_mean"))
            + "，empirical p="
            + _format_number(permutation.get("empirical_two_sided_p_value"))
            + "，status="
            + str(permutation.get("signal_status"))
        )
    evidence_grade = result.get("evidence_grade") or {}
    if evidence_grade.get("grade"):
        parts.append(
            "因果证据等级："
            + str(evidence_grade.get("label_zh") or "未知")
            + " ("
            + str(evidence_grade.get("grade"))
            + ")"
        )
        if evidence_grade.get("red_flags"):
            parts.append(f"证据红旗：{evidence_grade.get('red_flags')}")
        if evidence_grade.get("missing_checks"):
            parts.append(f"尚缺检查：{evidence_grade.get('missing_checks')}")
    data_quality = result.get("data_quality") or {}
    if data_quality:
        summary = data_quality.get("sample_summary") or {}
        parts.append(
            "数据质量门禁：ready="
            + str(data_quality.get("ready"))
            + "，complete-case fraction="
            + _format_number(summary.get("complete_case_fraction"))
        )
    design_selection = result.get("design_selection") or {}
    if design_selection:
        parts.append(
            "确定性方法路由："
            + str(design_selection.get("study_design"))
            + " → "
            + str(design_selection.get("selected_estimator_names"))
        )
    return "\n".join(parts)


def build_evidence_rows(
    tool_results: list[ToolResult],
) -> list[dict[str, str]]:
    """
    将 Agent 的 ToolResult 转换为适合 CLI 展示的证据行。

    返回的每一项都只使用工具实际返回的结构化数据，
    不做额外统计推断或自然语言扩写。
    """
    evidence_rows: list[dict[str, str]] = []

    for tool_result in tool_results:
        if not tool_result.success:
            evidence_rows.append(
                {
                    "tool_name": tool_result.tool_name,
                    "status": "失败",
                    "evidence": (
                        f"错误：{tool_result.error or '未知错误'}"
                    ),
                }
            )
            continue

        result = tool_result.result or {}

        if tool_result.tool_name == "profile_dataset":
            evidence = _build_profile_evidence(result)

        elif tool_result.tool_name == "run_readonly_sql":
            evidence = _build_sql_evidence(result)

        elif tool_result.tool_name == "create_chart":
            evidence = _build_chart_evidence(result)

        elif tool_result.tool_name == "retrieve_docs":
            evidence = _build_retrieve_docs_evidence(result)

        elif tool_result.tool_name == "run_causal_estimators":
            evidence = _build_causal_estimation_evidence(result)

        else:
            result_keys = _format_columns(
                list(result.keys()),
            )
            evidence = f"返回字段：{result_keys}"

        evidence_rows.append(
            {
                "tool_name": tool_result.tool_name,
                "status": "成功",
                "evidence": evidence,
            }
        )

    return evidence_rows

'''
def _format_retrieve_docs_evidence(tool_result) -> str:
    if not tool_result.success:
        return "文档检索失败。"

    result = tool_result.result or {}
    query = result.get("query", "")
    retrieved_items = result.get("results", [])

    if not retrieved_items:
        return "未检索到相关项目文档片段。"

    parts = [
        f"检索问题：{query}",
        f"返回片段数：{len(retrieved_items)}",
    ]

    sources = []
    for item in retrieved_items:
        source = item.get("source", "")
        chunk_id = item.get("chunk_id", "")
        distance = item.get("distance", None)

        if source:
            if distance is None:
                sources.append(f"{source} ({chunk_id})")
            else:
                sources.append(f"{source} ({chunk_id}, distance={distance:.4f})")

    if sources:
        parts.append("来源片段：")
        parts.extend(f"- {source}" for source in sources)

    return "\n".join(parts)
'''

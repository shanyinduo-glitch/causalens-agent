from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def _to_plain_value(value: Any) -> Any:
    """
    将 Pydantic 模型、Path 和嵌套对象转换为可 JSON 序列化的普通值。
    """

    if hasattr(value, "model_dump"):
        return _to_plain_value(
            value.model_dump()
        )

    if isinstance(value, Mapping):
        return {
            str(key): _to_plain_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_plain_value(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


def _format_json(value: Any) -> str:
    """
    将工具参数或结果格式化为 Markdown JSON 代码块。
    """

    plain_value = _to_plain_value(value)

    return json.dumps(
        plain_value,
        ensure_ascii=False,
        indent=2,
    )


def _normalize_tool_call(
    tool_call: Any,
) -> dict[str, Any]:
    """
    兼容字典和对象形式的工具调用记录。
    """

    plain_call = _to_plain_value(tool_call)

    if not isinstance(plain_call, dict):
        return {
            "id": None,
            "name": "unknown_tool",
            "arguments": {},
        }

    tool_name = (
        plain_call.get("tool_name")
        or plain_call.get("name")
        or "unknown_tool"
    )

    arguments = (
        plain_call.get("arguments")
        or plain_call.get("args")
        or {}
    )

    return {
        "id": plain_call.get("id"),
        "name": str(tool_name),
        "arguments": arguments,
    }


def _normalize_tool_result(
    tool_result: Any,
) -> dict[str, Any]:
    """
    兼容字典和 Pydantic ToolResult。
    """

    plain_result = _to_plain_value(
        tool_result
    )

    if not isinstance(plain_result, dict):
        return {
            "tool_name": "unknown_tool",
            "success": False,
            "result": None,
            "error": (
                "无法解析工具结果。"
            ),
        }

    tool_name = (
        plain_result.get("tool_name")
        or plain_result.get("name")
        or "unknown_tool"
    )

    return {
        "tool_name": str(tool_name),
        "success": (
            plain_result.get("success")
            is True
        ),
        "result": plain_result.get(
            "result"
        ),
        "error": plain_result.get(
            "error"
        ),
    }


def _match_tool_results(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[
    tuple[
        dict[str, Any],
        dict[str, Any] | None,
    ]
]:
    """
    按工具名称为调用记录匹配对应结果。

    在名称无法匹配时，使用尚未匹配的下一条结果，
    避免报告因记录格式变化而崩溃。
    """

    matched_items: list[
        tuple[
            dict[str, Any],
            dict[str, Any] | None,
        ]
    ] = []

    used_result_indexes: set[int] = set()

    for tool_call in tool_calls:
        matched_index: int | None = None

        for index, tool_result in enumerate(
            tool_results
        ):
            if index in used_result_indexes:
                continue

            if (
                tool_result["tool_name"]
                == tool_call["name"]
            ):
                matched_index = index
                break

        if matched_index is None:
            for index in range(
                len(tool_results)
            ):
                if index not in used_result_indexes:
                    matched_index = index
                    break

        if matched_index is None:
            matched_result = None
        else:
            used_result_indexes.add(
                matched_index
            )
            matched_result = tool_results[
                matched_index
            ]

        matched_items.append(
            (
                tool_call,
                matched_result,
            )
        )

    return matched_items


def _format_compact_value(
    value: Any,
) -> str:
    """
    将数值格式化为适合摘要展示的紧凑文本。
    """

    if isinstance(value, float):
        return (
            f"{value:.4f}"
            .rstrip("0")
            .rstrip(".")
        )

    return str(value)


def _extract_document_sources(
    result: Mapping[str, Any],
) -> list[str]:
    """
    从 RAG 结果中提取来源文件和章节。

    最多保留前三个不同来源，避免摘要过长。
    """

    raw_results = result.get("results")

    if not isinstance(raw_results, list):
        return []

    source_labels: list[str] = []

    for item in raw_results:
        if not isinstance(item, Mapping):
            continue

        metadata = item.get("metadata")

        if not isinstance(metadata, Mapping):
            metadata = {}

        source = (
            item.get("source")
            or item.get("source_file")
            or item.get("document_name")
            or metadata.get("source")
            or metadata.get("source_file")
        )

        section = (
            item.get("section")
            or metadata.get("section")
        )

        if not source:
            continue

        label = str(source)

        if section:
            label = (
                f"{label} / {section}"
            )

        if label not in source_labels:
            source_labels.append(label)

        if len(source_labels) >= 3:
            break

    return source_labels


def _summarize_tool_result(
    tool_result: Mapping[str, Any],
) -> str:
    """
    将单条工具结果转换为确定性的 Markdown 摘要。

    只整理工具实际返回的字段，
    不补充业务解释或统计推断。
    """

    tool_name = str(
        tool_result.get(
            "tool_name",
            "unknown_tool",
        )
    )

    success = (
        tool_result.get("success")
        is True
    )

    if not success:
        error = (
            tool_result.get("error")
            or "未提供具体错误信息"
        )

        return (
            f"- `{tool_name}`：失败；"
            f"{error}"
        )

    result = tool_result.get("result")

    if not isinstance(result, Mapping):
        return (
            f"- `{tool_name}`：成功；"
            "工具未返回结构化结果。"
        )

    parts: list[str] = ["成功"]

    if tool_name == "profile_dataset":
        row_count = (
            result.get("row_count")
            or result.get("rows")
        )
        column_names = (
            result.get("column_names")
            or result.get("columns")
        )

        if isinstance(row_count, int):
            parts.append(
                f"数据共 {row_count} 行"
            )

        if isinstance(column_names, list):
            parts.append(
                f"共 {len(column_names)} 个字段"
            )
            parts.append(
                "字段为 "
                + "、".join(
                    str(name)
                    for name in column_names
                )
            )

    elif tool_name == "retrieve_docs":
        query = result.get("query")
        raw_results = result.get("results")
        result_count = result.get(
            "result_count"
        )

        if (
            result_count is None
            and isinstance(raw_results, list)
        ):
            result_count = len(raw_results)

        if query:
            parts.append(
                f"检索问题为“{query}”"
            )

        if isinstance(result_count, int):
            parts.append(
                f"返回 {result_count} 个片段"
            )

        source_labels = (
            _extract_document_sources(
                result
            )
        )

        if source_labels:
            parts.append(
                "来源包括 "
                + "、".join(source_labels)
            )

    elif tool_name == "run_readonly_sql":
        rows = result.get("rows")
        columns = result.get("columns")
        row_count = result.get(
            "row_count"
        )

        if (
            row_count is None
            and isinstance(rows, list)
        ):
            row_count = len(rows)

        if isinstance(row_count, int):
            parts.append(
                f"返回 {row_count} 行"
            )

        if isinstance(columns, list):
            parts.append(
                "结果字段为 "
                + "、".join(
                    str(column)
                    for column in columns
                )
            )

    elif tool_name == "create_chart":
        chart_type = result.get(
            "chart_type"
        )
        plotted_row_count = result.get(
            "plotted_row_count"
        )
        chart_path = result.get(
            "chart_path"
        )

        if chart_type:
            parts.append(
                f"生成 {chart_type} 图"
            )

        if isinstance(
            plotted_row_count,
            int,
        ):
            parts.append(
                f"参与绘图数据为 "
                f"{plotted_row_count} 条"
            )

        numeric_summary = result.get(
            "numeric_summary"
        )

        if isinstance(
            numeric_summary,
            Mapping,
        ):
            summary_parts = []

            for key in (
                "count",
                "mean",
                "median",
                "min",
                "max",
            ):
                if key in numeric_summary:
                    summary_parts.append(
                        f"{key}="
                        f"{_format_compact_value(numeric_summary[key])}"
                    )

            if summary_parts:
                parts.append(
                    "数值摘要为 "
                    + "，".join(summary_parts)
                )

        relationship_summary = (
            result.get(
                "relationship_summary"
            )
        )

        if isinstance(
            relationship_summary,
            Mapping,
        ):
            relationship_parts = []

            for key in (
                "point_count",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "pearson_correlation",
            ):
                if key in relationship_summary:
                    relationship_parts.append(
                        f"{key}="
                        f"{_format_compact_value(relationship_summary[key])}"
                    )

            if relationship_parts:
                parts.append(
                    "关系摘要为 "
                    + "，".join(
                        relationship_parts
                    )
                )

        if chart_path:
            parts.append(
                f"图表路径为 `{chart_path}`"
            )

    elif tool_name == "run_causal_estimators":
        primary_estimator = result.get("primary_estimator")
        estimand = str(result.get("estimand") or "ATE")
        sample_size = result.get("sample_size")
        estimates = result.get("estimates") or []
        comparison = result.get("comparison") or {}

        if primary_estimator:
            parts.append(f"主估计器为 {primary_estimator}")
        if isinstance(sample_size, int):
            parts.append(f"完整样本量为 {sample_size}")
        for estimate in estimates:
            if not isinstance(estimate, Mapping):
                continue
            parts.append(
                "{name} {estimand}={ate}，95% CI={interval}".format(
                    name=estimate.get("estimator"),
                    estimand=estimand,
                    ate=_format_compact_value(estimate.get("ate")),
                    interval=estimate.get("confidence_interval"),
                )
            )
            sensitivity = estimate.get("sensitivity_analysis") or {}
            if (
                isinstance(sensitivity, Mapping)
                and sensitivity.get("available", True)
            ):
                point_rv = sensitivity.get("robustness_value_point_estimate")
                significance_rv = sensitivity.get(
                    "robustness_value_significance"
                )
                parts.append(
                    "未观测混杂敏感性 RV(point/significance)="
                    f"{_format_compact_value(point_rv)}/"
                    f"{_format_compact_value(significance_rv)}，"
                    "风险分级="
                    f"{sensitivity.get('unmeasured_confounding_risk')}"
                )
            elif isinstance(sensitivity, Mapping) and sensitivity:
                parts.append(
                    "未观测混杂敏感性不可用："
                    f"{sensitivity.get('reason')}"
                )
            if estimate.get("estimator") == "TwoStageLeastSquares":
                first_stage = estimate.get("first_stage") or {}
                endogeneity = estimate.get("endogeneity_test") or {}
                overidentification = (
                    estimate.get("overidentification_test") or {}
                )
                parts.append(
                    "IV 第一阶段 F/partial R²="
                    f"{_format_compact_value(first_stage.get('f_statistic'))}/"
                    f"{_format_compact_value(first_stage.get('partial_r2'))}，"
                    f"weak={first_stage.get('weak_instrument_warning')}"
                )
                parts.append(
                    "内生性检验 p="
                    f"{_format_compact_value(endogeneity.get('p_value'))}"
                )
                if overidentification.get("available"):
                    parts.append(
                        "Hansen J p="
                        f"{_format_compact_value(overidentification.get('p_value'))}，"
                        f"passed={overidentification.get('passed')}"
                    )
            if estimate.get("estimator") == "SharpRegressionDiscontinuity":
                density = estimate.get("density_diagnostic") or {}
                continuity = estimate.get("covariate_continuity") or {}
                bandwidth = estimate.get("bandwidth_sensitivity") or {}
                donut = estimate.get("donut_sensitivity") or {}
                support = estimate.get("local_support") or {}
                parts.append(
                    "RDD running/cutoff/bandwidth="
                    f"{estimate.get('running_variable')}/"
                    f"{_format_compact_value(estimate.get('cutoff'))}/"
                    f"{_format_compact_value(estimate.get('bandwidth'))}"
                )
                parts.append(
                    "断点两侧样本="
                    f"{support.get('left_count')}/{support.get('right_count')}，"
                    "density p="
                    f"{_format_compact_value(density.get('p_value'))}，"
                    f"covariates passed={continuity.get('all_passed')}"
                )
                parts.append(
                    f"带宽方向稳定={bandwidth.get('direction_stable')}，"
                    "donut estimate="
                    f"{_format_compact_value(donut.get('estimate'))}"
                )
            if estimate.get("estimator") == "SyntheticControl":
                fit = estimate.get("fit_diagnostics") or {}
                placebo = estimate.get("placebo_inference") or {}
                parts.append(
                    "Synthetic control unit/time="
                    f"{estimate.get('treated_unit')}/"
                    f"{_format_compact_value(estimate.get('intervention_time'))}"
                )
                parts.append(
                    "pre RMSPE / post-pre ratio / placebo p="
                    f"{_format_compact_value(fit.get('pre_rmspe'))}/"
                    f"{_format_compact_value(fit.get('post_pre_rmspe_ratio'))}/"
                    f"{_format_compact_value(placebo.get('p_value'))}"
                )
            if estimate.get("estimator") == "TimeSeriesLocalProjection":
                configuration = estimate.get("model_configuration") or {}
                stationarity = estimate.get("stationarity") or {}
                placebo = estimate.get("placebo_test") or {}
                residual = estimate.get("residual_diagnostics") or {}
                parts.append(
                    "连续冲击 IRF（每 1 个原始单位，不是二元处理 ATE）："
                    f"response={configuration.get('response')}，"
                    f"horizon=0..{configuration.get('max_horizon')}，"
                    f"covariance={configuration.get('covariance')}"
                )
                parts.append(
                    "ADF treatment/outcome p="
                    f"{_format_compact_value((stationarity.get('treatment') or {}).get('p_value'))}/"
                    f"{_format_compact_value((stationarity.get('outcome') or {}).get('p_value'))}，"
                    "future-treatment placebo p="
                    f"{_format_compact_value(placebo.get('p_value'))}，"
                    "residual autocorrelation warning="
                    f"{residual.get('residual_autocorrelation_warning')}"
                )
                parts.append(
                    "动态响应="
                    f"{estimate.get('dynamic_effects')}"
                )
            heterogeneity = estimate.get("heterogeneity") or {}
            if isinstance(heterogeneity, Mapping) and heterogeneity:
                parts.append(
                    "CATE 分布 P10/Median/P90="
                    f"{_format_compact_value(heterogeneity.get('cate_p10'))}/"
                    f"{_format_compact_value(heterogeneity.get('cate_median'))}/"
                    f"{_format_compact_value(heterogeneity.get('cate_p90'))}"
                )
            if estimate.get("estimator") == "DifferenceInDifferences":
                parts.append(
                    "未调整四单元格 DiD="
                    f"{_format_compact_value(estimate.get('unadjusted_did'))}"
                )
                parts.append(f"四单元格均值={estimate.get('cell_means')}")
            if estimate.get("estimator") == "EventStudy":
                pretrend = estimate.get("pretrend_test") or {}
                parts.append(
                    "联合预趋势检验 p="
                    f"{_format_compact_value(pretrend.get('p_value'))}"
                )
                parts.append(
                    "动态效应="
                    f"{estimate.get('dynamic_effects')}"
                )
                if estimate.get("chart_path"):
                    parts.append(f"动态图路径=`{estimate.get('chart_path')}`")
            if estimate.get("estimator") == "StaggeredAdoptionDID":
                pretrend = estimate.get("pretrend_test") or {}
                configuration = estimate.get("model_configuration") or {}
                parts.append(
                    "调整模式="
                    f"{configuration.get('adjustment')}"
                )
                if configuration.get("nuisance_diagnostics"):
                    parts.append(
                        "nuisance 诊断="
                        f"{configuration.get('nuisance_diagnostics')}"
                    )
                placebo = estimate.get("placebo_test") or {}
                sensitivity = estimate.get("cohort_sensitivity") or {}
                if placebo:
                    parts.append(
                        "提前一期 placebo p="
                        f"{_format_compact_value(placebo.get('p_value'))}"
                    )
                if sensitivity:
                    parts.append(
                        "逐 cohort 剔除最大偏移="
                        f"{_format_compact_value(sensitivity.get('max_absolute_deviation'))}"
                    )
                parts.append(
                    "cohort-time ATT 单元数="
                    f"{len(estimate.get('group_time_effects') or [])}"
                )
                parts.append(
                    "联合预趋势检验 p="
                    f"{_format_compact_value(pretrend.get('p_value'))}"
                )
                parts.append(f"动态效应={estimate.get('dynamic_effects')}")
                if estimate.get("chart_path"):
                    parts.append(f"动态图路径=`{estimate.get('chart_path')}`")
        if "absolute_ate_difference" in comparison:
            parts.append(
                "估计值绝对差为 "
                + _format_compact_value(
                    comparison.get("absolute_ate_difference")
                )
            )
        if "confidence_intervals_overlap" in comparison:
            parts.append(
                "置信区间是否重叠："
                f"{comparison.get('confidence_intervals_overlap')}"
            )
        negative_controls = result.get("negative_control_summary") or {}
        if negative_controls.get("requested_outcomes"):
            parts.append(
                "负对照 outcome 通过数="
                f"{negative_controls.get('passed_count')}/"
                f"{negative_controls.get('result_count')}"
            )
        permutation = result.get("permutation_placebo") or {}
        if permutation.get("available"):
            parts.append(
                "残差化伪 treatment empirical p="
                f"{_format_compact_value(permutation.get('empirical_two_sided_p_value'))}，"
                f"status={permutation.get('signal_status')}"
            )
        evidence_grade = result.get("evidence_grade") or {}
        if evidence_grade.get("grade"):
            parts.append(
                "因果证据等级="
                f"{evidence_grade.get('label_zh')} "
                f"({evidence_grade.get('grade')})"
            )

    return (
        f"- `{tool_name}`："
        + "；".join(parts)
        + "。"
    )


def generate_markdown_report(
    state: Mapping[str, Any],
    *,
    dataset_path: str | Path | None = None,
    generated_at: datetime | None = None,
) -> str:
    """
    根据最终 AgentState 生成 Markdown 分析报告。

    本函数只负责整理和保存现有证据，
    不重新解释、补充或推断工具结果。
    """

    from causalens.reporting.chinese import time_series_report
    from causalens.reporting.planning_trace import planning_section

    chinese_report = time_series_report(state, dataset_path=dataset_path, generated_at=generated_at)
    if chinese_report is not None:
        planning = planning_section(state)
        return chinese_report + (('\n\n' + planning) if planning else '')

    report_time = (
        generated_at
        or datetime.now().astimezone()
    )

    question = str(
        state.get("question") or "未提供"
    )
    task_type = str(
        state.get("task_type") or "unknown"
    )
    workflow_status = str(
        state.get("workflow_status")
        or "unknown"
    )
    final_answer = str(
        state.get("final_answer")
        or "未生成最终回答。"
    )

    resolved_dataset_path = (
        dataset_path
        or state.get("dataset_path")
        or "未提供"
    )

    plan = state.get("plan") or []

    if not isinstance(plan, list):
        plan = [str(plan)]

    raw_tool_calls = (
        state.get("tool_calls") or []
    )
    raw_tool_results = (
        state.get("tool_results") or []
    )

    if not isinstance(
        raw_tool_calls,
        list,
    ):
        raw_tool_calls = []

    if not isinstance(
        raw_tool_results,
        list,
    ):
        raw_tool_results = []

    tool_calls = [
        _normalize_tool_call(tool_call)
        for tool_call in raw_tool_calls
    ]

    tool_results = [
        _normalize_tool_result(tool_result)
        for tool_result in raw_tool_results
    ]

    matched_tools = _match_tool_results(
        tool_calls,
        tool_results,
    )

    successful_tool_count = sum(
        tool_result["success"]
        for tool_result in tool_results
    )

    failed_tool_count = sum(
        not tool_result["success"]
        for tool_result in tool_results
    )

    lines = [
        "# CausaLens Analysis Report",
        "",
        "## Execution Summary",
        "",
        (
            f"- Generated at: "
            f"{report_time.isoformat(timespec='seconds')}"
        ),
        (
            f"- Dataset: "
            f"`{resolved_dataset_path}`"
        ),
        f"- Task type: `{task_type}`",
        (
            f"- Workflow status: "
            f"`{workflow_status}`"
        ),
        (
            f"- Step count: "
            f"{state.get('step_count', 0)}"
        ),
        (
            f"- Tool calls: "
            f"{len(tool_calls)}"
        ),
        (
            f"- Successful tools: "
            f"{successful_tool_count}"
        ),
        (
            f"- Failed tools: "
            f"{failed_tool_count}"
        ),
        "",
        "## User Question",
        "",
        question,
        "",
        "## Analysis Plan",
        "",
    ]

    if plan:
        for index, step in enumerate(
            plan,
            start=1,
        ):
            lines.append(
                f"{index}. {step}"
            )
    else:
        lines.append(
            "No analysis plan was generated."
        )

    causal_question = state.get("causal_question")
    if isinstance(causal_question, Mapping) and causal_question:
        variable_roles = causal_question.get("variable_roles")
        if not isinstance(variable_roles, Mapping):
            variable_roles = {}

        contrast = causal_question.get("treatment_contrast")
        if not isinstance(contrast, Mapping):
            contrast = {}

        missing_fields = state.get("missing_design_fields") or []
        if not isinstance(missing_fields, list):
            missing_fields = []

        lines.extend(
            [
                "",
                "## Causal Question Contract",
                "",
                f"- Treatment: `{causal_question.get('treatment')}`",
                f"- Outcome: `{causal_question.get('outcome')}`",
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
                    "- Treatment contrast: `"
                    f"{contrast.get('control', 0)} → "
                    f"{contrast.get('treated', 1)}`"
                ),
                (
                    "- Target population: "
                    f"{causal_question.get('target_population') or '未明确'}"
                ),
                (
                    "- Unit of analysis: "
                    f"{causal_question.get('unit_of_analysis') or '未明确'}"
                ),
                (
                    "- Outcome horizon: "
                    f"{causal_question.get('outcome_horizon') or '未明确'}"
                ),
                (
                    "- Confounders: "
                    + ", ".join(
                        str(item)
                        for item in variable_roles.get("confounders", [])
                    )
                ),
                (
                    "- Negative-control outcomes: "
                    + ", ".join(
                        str(item)
                        for item in variable_roles.get(
                            "negative_control_outcomes", []
                        )
                    )
                ),
                (
                    "- Instruments: "
                    + ", ".join(
                        str(item)
                        for item in variable_roles.get("instruments", [])
                    )
                ),
                (
                    "- Missing design fields: "
                    + (", ".join(str(item) for item in missing_fields) or "None")
                ),
            ]
        )

        identification = state.get("causal_identification")
        if isinstance(identification, Mapping) and identification:
            lines.extend(
                [
                    "",
                    "### Identification Assessment",
                    "",
                    f"- Status: `{identification.get('status')}`",
                    f"- DAG provided: `{identification.get('dag_provided')}`",
                    f"- Identified: `{identification.get('identified')}`",
                ]
            )
            for warning in identification.get("warnings", []) or []:
                lines.append(f"- Warning: {warning}")
            for issue in identification.get("blocking_issues", []) or []:
                lines.append(f"- Blocking issue: {issue}")

        diagnostics = state.get("causal_diagnostics")
        if (
            isinstance(diagnostics, Mapping)
            and diagnostics.get("analysis_type")
            in {
                "did_design_diagnostics",
                "event_study_design_diagnostics",
                "staggered_did_design_diagnostics",
            }
        ):
            design_summary = diagnostics.get("design_summary") or {}
            is_event_study = (
                diagnostics.get("analysis_type")
                == "event_study_design_diagnostics"
            )
            is_staggered = (
                diagnostics.get("analysis_type")
                == "staggered_did_design_diagnostics"
            )
            lines.extend(
                [
                    "",
                    (
                        "### Staggered DiD Design Diagnostics"
                        if is_staggered
                        else "### Event Study Design Diagnostics"
                        if is_event_study
                        else "### DiD Design Diagnostics"
                    ),
                    "",
                    f"- Passed: `{diagnostics.get('passed')}`",
                    f"- Cell counts: `{design_summary.get('cell_counts')}`",
                    (
                        f"- Cohorts: `{design_summary.get('cohorts')}`"
                        if is_staggered
                        else f"- Pre periods: `{design_summary.get('pre_periods')}`"
                        if is_event_study
                        else f"- Cell means: `{design_summary.get('cell_means')}`"
                    ),
                    (
                        "- Never-treated units: `"
                        f"{design_summary.get('never_treated_units')}`"
                        if is_staggered
                        else f"- Post periods: `{design_summary.get('post_periods')}`"
                        if is_event_study
                        else "- Complete two-period units: `"
                        f"{design_summary.get('complete_two_period_units')}`"
                    ),
                ]
            )
            for warning in diagnostics.get("warnings", []) or []:
                lines.append(f"- Warning: {warning}")
        elif isinstance(diagnostics, Mapping) and diagnostics:
            propensity = diagnostics.get("propensity_summary") or {}
            weights = diagnostics.get("weight_summary") or {}
            lines.extend(
                [
                    "",
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
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| {variable} | {before} | {after} |".format(
                        variable=row.get("variable"),
                        before=row.get("smd_before"),
                        after=row.get("smd_after"),
                    )
                )
            for warning in diagnostics.get("warnings", []) or []:
                lines.append(f"- Warning: {warning}")
            for issue in diagnostics.get("blocking_issues", []) or []:
                lines.append(f"- Blocking issue: {issue}")

    causal_suite = next(
        (
            item.get("result")
            for item in tool_results
            if item.get("success") is True
            and item.get("tool_name") == "run_causal_estimators"
            and isinstance(item.get("result"), Mapping)
        ),
        None,
    )
    if isinstance(causal_suite, Mapping):
        evidence_grade = causal_suite.get("evidence_grade") or {}
        if evidence_grade.get("grade"):
            lines.extend(
                [
                    "",
                    "## Causal Evidence Grade",
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
                    (
                        "> This deterministic grade summarizes available "
                        "diagnostics. It is not a probability that the causal "
                        "claim is true and is not proof of identification."
                    ),
                ]
            )
        iv_estimate = next(
            (
                estimate
                for estimate in causal_suite.get("estimates", []) or []
                if isinstance(estimate, Mapping)
                and estimate.get("estimator") == "TwoStageLeastSquares"
            ),
            None,
        )
        if isinstance(iv_estimate, Mapping):
            first_stage = iv_estimate.get("first_stage") or {}
            endogeneity = iv_estimate.get("endogeneity_test") or {}
            overidentification = (
                iv_estimate.get("overidentification_test") or {}
            )
            lines.extend(
                [
                    "",
                    "## Instrumental-Variable Diagnostics",
                    "",
                    f"- Instruments: `{causal_suite.get('instruments') or []}`",
                    (
                        "- First-stage F statistic: `"
                        f"{first_stage.get('f_statistic')}`"
                    ),
                    f"- First-stage partial R²: `{first_stage.get('partial_r2')}`",
                    (
                        "- Weak-instrument warning: `"
                        f"{first_stage.get('weak_instrument_warning')}`"
                    ),
                    (
                        "- Endogeneity-test p-value: `"
                        f"{endogeneity.get('p_value')}`"
                    ),
                    (
                        "- Hansen J available: `"
                        f"{overidentification.get('available')}`"
                    ),
                    (
                        "- Hansen J p-value: `"
                        f"{overidentification.get('p_value')}`"
                    ),
                    "",
                    (
                        "> First-stage strength is empirically testable, but "
                        "instrument independence and the exclusion restriction "
                        "cannot be proven from the observed data."
                    ),
                ]
            )
        rdd_estimate = next(
            (
                estimate
                for estimate in causal_suite.get("estimates", []) or []
                if isinstance(estimate, Mapping)
                and estimate.get("estimator") == "SharpRegressionDiscontinuity"
            ),
            None,
        )
        if isinstance(rdd_estimate, Mapping):
            density = rdd_estimate.get("density_diagnostic") or {}
            continuity = rdd_estimate.get("covariate_continuity") or {}
            sensitivity = rdd_estimate.get("bandwidth_sensitivity") or {}
            donut = rdd_estimate.get("donut_sensitivity") or {}
            support = rdd_estimate.get("local_support") or {}
            lines.extend(
                [
                    "",
                    "## Regression-Discontinuity Diagnostics",
                    "",
                    f"- Running variable: `{rdd_estimate.get('running_variable')}`",
                    f"- Cutoff: `{rdd_estimate.get('cutoff')}`",
                    f"- Bandwidth: `{rdd_estimate.get('bandwidth')}`",
                    (
                        "- Local left / right observations: `"
                        f"{support.get('left_count')} / {support.get('right_count')}`"
                    ),
                    f"- Density diagnostic method: `{density.get('method')}`",
                    f"- Density diagnostic p-value: `{density.get('p_value')}`",
                    (
                        "- Covariate continuity all passed: `"
                        f"{continuity.get('all_passed')}`"
                    ),
                    (
                        "- Bandwidth direction stable: `"
                        f"{sensitivity.get('direction_stable')}`"
                    ),
                    f"- Donut estimate: `{donut.get('estimate')}`",
                    "",
                    (
                        "> The density result is a heuristic local count test, "
                        "not a formal McCrary test. RDD identification still "
                        "requires continuity and no precise manipulation."
                    ),
                ]
            )
        synthetic_estimate = next(
            (
                estimate
                for estimate in causal_suite.get("estimates", []) or []
                if isinstance(estimate, Mapping)
                and estimate.get("estimator") == "SyntheticControl"
            ),
            None,
        )
        if isinstance(synthetic_estimate, Mapping):
            fit = synthetic_estimate.get("fit_diagnostics") or {}
            placebo = synthetic_estimate.get("placebo_inference") or {}
            weights = synthetic_estimate.get("donor_weights") or []
            lines.extend(
                [
                    "",
                    "## Synthetic-Control Diagnostics",
                    "",
                    f"- Treated unit: `{synthetic_estimate.get('treated_unit')}`",
                    (
                        "- Intervention time: `"
                        f"{synthetic_estimate.get('intervention_time')}`"
                    ),
                    f"- Pre-treatment RMSPE: `{fit.get('pre_rmspe')}`",
                    f"- Post-treatment RMSPE: `{fit.get('post_rmspe')}`",
                    (
                        "- Post/pre RMSPE ratio: `"
                        f"{fit.get('post_pre_rmspe_ratio')}`"
                    ),
                    f"- In-space placebo count: `{placebo.get('placebo_count')}`",
                    f"- Placebo rank p-value: `{placebo.get('p_value')}`",
                    f"- Non-zero donor weights: `{weights}`",
                    "",
                    (
                        "> The effect is local to the treated unit and requires "
                        "no spillovers, untreated donors, stable convex-combination "
                        "fit, and no coincident treated-unit-specific shock."
                    ),
                ]
            )
        time_series_estimate = next(
            (
                estimate
                for estimate in causal_suite.get("estimates", []) or []
                if isinstance(estimate, Mapping)
                and estimate.get("estimator") == "TimeSeriesLocalProjection"
            ),
            None,
        )
        if isinstance(time_series_estimate, Mapping):
            configuration = time_series_estimate.get("model_configuration") or {}
            stationarity = time_series_estimate.get("stationarity") or {}
            placebo = time_series_estimate.get("placebo_test") or {}
            residual = time_series_estimate.get("residual_diagnostics") or {}
            lines.extend(
                [
                    "",
                    "## Time-Series Local-Projection Diagnostics",
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
                ]
            )
        data_quality = causal_suite.get("data_quality") or {}
        if isinstance(data_quality, Mapping) and data_quality:
            quality_summary = data_quality.get("sample_summary") or {}
            lines.extend(
                [
                    "",
                    "## Causal Data Quality Gate",
                    "",
                    f"- Ready: `{data_quality.get('ready')}`",
                    (
                        "- Complete-case fraction: `"
                        f"{quality_summary.get('complete_case_fraction')}`"
                    ),
                    f"- Blocking issues: `{data_quality.get('blocking_issues') or []}`",
                    f"- Warnings: `{data_quality.get('warnings') or []}`",
                    (
                        "- Preprocessing policy: `"
                        f"{data_quality.get('preprocessing_policy') or {}}`"
                    ),
                ]
            )
        sensitivity_rows = [
            estimate
            for estimate in causal_suite.get("estimates", []) or []
            if isinstance(estimate, Mapping)
            and isinstance(estimate.get("sensitivity_analysis"), Mapping)
        ]
        if sensitivity_rows:
            lines.extend(
                [
                    "",
                    "## Unmeasured Confounding Sensitivity",
                    "",
                    "| Estimator | RV point | RV significance | Risk |",
                    "|---|---:|---:|---|",
                ]
            )
            for estimate in sensitivity_rows:
                sensitivity = estimate["sensitivity_analysis"]
                if sensitivity.get("available", True):
                    lines.append(
                        "| {estimator} | {point} | {significance} | {risk} |".format(
                            estimator=estimate.get("estimator"),
                            point=_format_compact_value(
                                sensitivity.get(
                                    "robustness_value_point_estimate"
                                )
                            ),
                            significance=_format_compact_value(
                                sensitivity.get(
                                    "robustness_value_significance"
                                )
                            ),
                            risk=sensitivity.get("unmeasured_confounding_risk"),
                        )
                    )
                else:
                    lines.append(
                        f"| {estimate.get('estimator')} | NA | NA | unavailable |"
                    )
            lines.extend(
                [
                    "",
                    (
                        "> RV is an approximate partial-R² sensitivity summary "
                        "for non-OLS or robust/bootstrapped estimators; it does "
                        "not establish that unmeasured confounding is absent."
                    ),
                ]
            )
            primary_sensitivity = sensitivity_rows[0]["sensitivity_analysis"]
            benchmark = primary_sensitivity.get(
                "observed_confounder_benchmark", {}
            )
            if benchmark.get("available"):
                lines.extend(
                    [
                        "",
                        "### Observed-confounder Benchmark",
                        "",
                        (
                            "- Strongest observed benchmark: `"
                            f"{benchmark.get('strongest_confounder')}`"
                        ),
                        (
                            "- Assessment: `"
                            f"{benchmark.get('assessment')}`"
                        ),
                        "",
                        "| Confounder | Partial R² treatment | Partial R² outcome | Equivalent RV | Ratio to point RV |",
                        "|---|---:|---:|---:|---:|",
                    ]
                )
                for row in benchmark.get("confounders", []) or []:
                    if not isinstance(row, Mapping):
                        continue
                    lines.append(
                        "| {name} | {treatment} | {outcome} | {equivalent} | {ratio} |".format(
                            name=row.get("confounder"),
                            treatment=_format_compact_value(
                                row.get("partial_r2_treatment")
                            ),
                            outcome=_format_compact_value(
                                row.get("partial_r2_outcome")
                            ),
                            equivalent=_format_compact_value(
                                row.get("equivalent_robustness_value")
                            ),
                            ratio=_format_compact_value(
                                row.get("ratio_to_point_rv")
                            ),
                        )
                    )

        negative_controls = causal_suite.get("negative_control_summary") or {}
        if negative_controls.get("requested_outcomes"):
            lines.extend(
                [
                    "",
                    "## Negative-Control Outcome Checks",
                    "",
                    f"- Estimator: `{negative_controls.get('estimator')}`",
                    f"- All passed: `{negative_controls.get('all_passed')}`",
                    f"- Assumption: {negative_controls.get('assumption')}",
                    "",
                    "| Outcome | Estimate | 95% CI | Passed | Error |",
                    "|---|---:|---|---:|---|",
                ]
            )
            for row in negative_controls.get("results", []) or []:
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| {outcome} | {estimate} | {interval} | {passed} | {error} |".format(
                        outcome=row.get("outcome"),
                        estimate=_format_compact_value(row.get("estimate")),
                        interval=row.get("confidence_interval"),
                        passed=row.get("passed"),
                        error=row.get("error") or "",
                    )
                )
        permutation = causal_suite.get("permutation_placebo") or {}
        if permutation.get("available"):
            quantiles = permutation.get("placebo_quantiles") or {}
            lines.extend(
                [
                    "",
                    "## Residualized Treatment-Permutation Diagnostic",
                    "",
                    f"- Method: `{permutation.get('method')}`",
                    (
                        "- Diagnostic statistic: `"
                        f"{permutation.get('diagnostic_statistic')}`"
                    ),
                    f"- Iterations: `{permutation.get('iterations')}`",
                    (
                        "- Pseudo treatment: `"
                        f"{permutation.get('pseudo_treatment_type')}`"
                    ),
                    (
                        "- Observed adjusted coefficient: `"
                        f"{permutation.get('observed_adjusted_coefficient')}`"
                    ),
                    f"- Placebo mean: `{permutation.get('placebo_mean')}`",
                    (
                        "- Placebo 2.5% / 50% / 97.5%: `"
                        f"{quantiles.get('q025')} / {quantiles.get('q50')} / "
                        f"{quantiles.get('q975')}`"
                    ),
                    (
                        "- Empirical two-sided p-value: `"
                        f"{permutation.get('empirical_two_sided_p_value')}`"
                    ),
                    f"- Signal status: `{permutation.get('signal_status')}`",
                    (
                        "- Placebo calibration passed: `"
                        f"{permutation.get('calibration_passed')}`"
                    ),
                    "",
                    (
                        "> This diagnostic uses a covariate-adjusted linear "
                        "projection and does not refit or replace the primary "
                        "causal estimator."
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
        ]
    )

    if tool_results:
        for tool_result in tool_results:
            lines.append(
                _summarize_tool_result(
                    tool_result
                )
            )
    else:
        lines.append(
            "No tool evidence was produced."
        )

    lines.extend(
        [
            "",
            "## Tool Execution",
            "",
        ]
    )

    if not matched_tools:
        lines.append(
            "No tools were called."
        )

    for index, (
        tool_call,
        tool_result,
    ) in enumerate(
        matched_tools,
        start=1,
    ):
        tool_name = tool_call["name"]

        lines.extend(
            [
                f"### {index}. `{tool_name}`",
                "",
            ]
        )

        if tool_call["id"]:
            lines.append(
                f"- Call ID: `{tool_call['id']}`"
            )

        if tool_result is None:
            lines.append(
                "- Status: Result not found"
            )
        elif tool_result["success"]:
            lines.append(
                "- Status: Success"
            )
        else:
            lines.append(
                "- Status: Failed"
            )

        lines.extend(
            [
                "",
                "**Arguments**",
                "",
                "```json",
                _format_json(
                    tool_call["arguments"]
                ),
                "```",
                "",
            ]
        )

        if tool_result is None:
            lines.extend(
                [
                    "**Result**",
                    "",
                    "No matching tool result was found.",
                    "",
                ]
            )
            continue

        if tool_result["result"] is not None:
            lines.extend(
                [
                    "**Result**",
                    "",
                    "```json",
                    _format_json(
                        tool_result["result"]
                    ),
                    "```",
                    "",
                ]
            )

        if tool_result["error"]:
            lines.extend(
                [
                    "**Error**",
                    "",
                    str(tool_result["error"]),
                    "",
                ]
            )

    lines.extend(
        [
            "## Final Answer",
            "",
            final_answer,
            "",
        ]
    )

    state_error = state.get("error")

    if state_error:
        lines.extend(
            [
                "## Workflow Error",
                "",
                str(state_error),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n" + planning_section(state)


def save_markdown_report(
    state: Mapping[str, Any],
    output_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """
    生成报告并写入指定 Markdown 文件。
    """

    resolved_output_path = Path(
        output_path
    )

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = generate_markdown_report(
        state,
        dataset_path=dataset_path,
        generated_at=generated_at,
    )

    resolved_output_path.write_text(
        report,
        encoding="utf-8",
    )

    return resolved_output_path

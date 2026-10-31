from datetime import datetime, timezone
from pathlib import Path

from causalens.reporting.markdown_report import (
    generate_markdown_report,
    save_markdown_report,
)


def build_mixed_state() -> dict:
    return {
        "dataset_path": (
            "data/student_performance.csv"
        ),
        "question": (
            "请解释 passed 字段含义，"
            "并统计不同 passed 值的平均 score。"
        ),
        "task_type": "mixed_analysis",
        "workflow_status": "completed",
        "step_count": 2,
        "plan": [
            "检索 passed 字段定义",
            "统计不同 passed 值的平均 score",
            "汇总工具结果",
        ],
        "tool_calls": [
            {
                "id": "call_rag",
                "tool_name": "retrieve_docs",
                "arguments": {
                    "query": (
                        "请解释 passed 字段含义"
                    ),
                    "top_k": 3,
                },
            },
            {
                "id": "call_sql",
                "tool_name": "run_readonly_sql",
                "arguments": {
                    "dataset_path": (
                        "data/student_performance.csv"
                    ),
                    "query": (
                        "SELECT passed, AVG(score) "
                        "AS avg_score FROM dataset "
                        "GROUP BY passed"
                    ),
                },
            },
        ],
        "tool_results": [
            {
                "tool_name": "retrieve_docs",
                "success": True,
                "result": {
                    "query": (
                        "请解释 passed 字段含义"
                    ),
                    "result_count": 3,
                },
                "error": None,
            },
            {
                "tool_name": "run_readonly_sql",
                "success": True,
                "result": {
                    "columns": [
                        "passed",
                        "avg_score",
                    ],
                    "rows": [
                        {
                            "passed": 0,
                            "avg_score": 62.09,
                        },
                        {
                            "passed": 1,
                            "avg_score": 84.76,
                        },
                    ],
                },
                "error": None,
            },
        ],
        "final_answer": (
            "passed=0 时平均 score 约为 62.09，"
            "passed=1 时约为 84.76。"
        ),
        "error": None,
    }


def test_generate_markdown_report_contains_core_sections() -> None:
    state = build_mixed_state()

    fixed_time = datetime(
        2026,
        7,
        28,
        16,
        0,
        0,
        tzinfo=timezone.utc,
    )

    report = generate_markdown_report(
        state,
        generated_at=fixed_time,
    )

    # 检查报告核心章节。
    assert (
        "# CausaLens Analysis Report"
        in report
    )
    assert "## Execution Summary" in report
    assert "## User Question" in report
    assert "## Analysis Plan" in report
    assert "## Evidence Summary" in report
    assert "## Tool Execution" in report
    assert "## Final Answer" in report

    # 检查执行状态。
    assert "`mixed_analysis`" in report
    assert "`completed`" in report
    assert "Step count: 2" in report

    # 检查面向用户的证据摘要。
    assert (
        "`retrieve_docs`：成功"
        in report
    )
    assert (
        "`run_readonly_sql`：成功"
        in report
    )
    assert (
        "检索问题为“请解释 passed 字段含义”"
        in report
    )
    assert "返回 3 个片段" in report
    assert "返回 2 行" in report
    assert (
        "结果字段为 passed、avg_score"
        in report
    )

    # 检查保留的原始工具调用与结果。
    assert "### 1. `retrieve_docs`" in report
    assert "### 2. `run_readonly_sql`" in report
    assert '"top_k": 3' in report
    assert "AVG(score)" in report

    assert "62.09" in report
    assert "84.76" in report

def test_save_markdown_report_creates_file(
    tmp_path: Path,
) -> None:
    state = build_mixed_state()

    output_path = (
        tmp_path
        / "nested"
        / "analysis_report.md"
    )

    saved_path = save_markdown_report(
        state,
        output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    content = output_path.read_text(
        encoding="utf-8"
    )

    assert (
        "# CausaLens Analysis Report"
        in content
    )
    assert "retrieve_docs" in content
    assert "run_readonly_sql" in content


def test_report_handles_unsupported_task() -> None:
    state = {
        "dataset_path": "data/sample.csv",
        "question": "请删除所有数据。",
        "task_type": "unsupported",
        "workflow_status": "completed",
        "step_count": 0,
        "plan": [],
        "tool_calls": [],
        "tool_results": [],
        "final_answer": (
            "当前项目只支持只读分析，"
            "不会删除原始数据。"
        ),
        "error": None,
    }

    report = generate_markdown_report(
        state
    )

    assert "`unsupported`" in report
    assert "No tools were called." in report
    assert "不会删除原始数据" in report


def test_report_records_causal_question_contract() -> None:
    state = {
        "dataset_path": "data/synthetic_causal_data.csv",
        "question": "估计广告曝光的平均处理效应。",
        "task_type": "causal_analysis",
        "workflow_status": "awaiting_approval",
        "plan": [],
        "tool_calls": [],
        "tool_results": [],
        "final_answer": "等待用户确认。",
        "causal_question": {
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "estimand": "ATE",
            "treatment_contrast": {"control": 0, "treated": 1},
            "target_population": None,
            "unit_of_analysis": "user",
            "outcome_horizon": "曝光后 30 天",
            "variable_roles": {
                "confounders": ["age"],
            },
        },
        "missing_design_fields": ["target_population"],
        "causal_identification": {
            "status": "not_assessed",
            "dag_provided": False,
            "identified": None,
            "warnings": ["未提供因果 DAG。"],
            "blocking_issues": [],
        },
        "causal_diagnostics": {
            "passed": True,
            "propensity_summary": {
                "common_support": [0.1, 0.9],
                "extreme_fraction": 0.02,
            },
            "weight_summary": {
                "effective_sample_size": 812.5,
            },
            "balance": [
                {
                    "variable": "age",
                    "smd_before": 0.24,
                    "smd_after": 0.04,
                }
            ],
            "warnings": [],
            "blocking_issues": [],
        },
    }

    report = generate_markdown_report(state)

    assert "## Causal Question Contract" in report
    assert "Estimand: `ATE`" in report
    assert "Target population: 未明确" in report
    assert "Missing design fields: target_population" in report
    assert "### Identification Assessment" in report
    assert "Status: `not_assessed`" in report
    assert "### Overlap and Balance Diagnostics" in report
    assert "Common support: `[0.1, 0.9]`" in report
    assert "| age | 0.24 | 0.04 |" in report


def test_report_labels_time_series_irf_as_continuous_shock_not_ate() -> None:
    estimate = {
        "estimator": "TimeSeriesLocalProjection",
        "ate": 0.4,
        "confidence_interval": [0.1, 0.7],
        "sample_size": 120,
        "model_configuration": {
            "response": "cumulative outcome from t through t+h",
            "max_horizon": 2,
            "covariance": "Newey-West HAC",
        },
        "stationarity": {
            "treatment": {"p_value": 0.01},
            "outcome": {"p_value": 0.02},
        },
        "placebo_test": {"p_value": 0.5, "passed": True},
        "residual_diagnostics": {
            "residual_autocorrelation_warning": False,
        },
        "dynamic_effects": [{"horizon": 0, "estimate": 0.4}],
    }
    state = {
        "dataset_path": "data/time_series.csv",
        "question": "估计连续冲击的动态响应",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "step_count": 1,
        "plan": [],
        "tool_calls": [],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "estimand": "IRF",
                    "primary_estimator": "TimeSeriesLocalProjection",
                    "sample_size": 120,
                    "estimates": [estimate],
                    "comparison": {"estimate_count": 1},
                },
                "error": None,
            }
        ],
        "final_answer": "已完成。",
    }

    report = generate_markdown_report(state)

    assert "Time-Series Local-Projection Diagnostics" in report
    assert "IRF per one source-unit shock" in report
    assert "not a binary-treatment ATE" in report
    assert "Newey-West HAC" in report


def test_report_chart_summary_uses_returned_facts_only() -> None:
    state = {
        "dataset_path": (
            "data/student_performance.csv"
        ),
        "question": (
            "请画出 study_hours "
            "和 score 的散点图。"
        ),
        "task_type": "chart_analysis",
        "workflow_status": "completed",
        "step_count": 1,
        "plan": [
            "生成散点图",
            "总结工具结果",
        ],
        "tool_calls": [
            {
                "id": "call_chart",
                "tool_name": "create_chart",
                "arguments": {
                    "dataset_path": (
                        "data/student_performance.csv"
                    ),
                    "chart_type": "scatter",
                    "x_column": "study_hours",
                    "y_column": "score",
                },
            }
        ],
        "tool_results": [
            {
                "tool_name": "create_chart",
                "success": True,
                "result": {
                    "chart_type": "scatter",
                    "plotted_row_count": 40,
                    "chart_path": (
                        "outputs/charts/"
                        "study_hours_score.png"
                    ),
                    "relationship_summary": {
                        "point_count": 40,
                        "x_min": 3.0,
                        "x_max": 18.0,
                        "y_min": 52.0,
                        "y_max": 98.0,
                        "pearson_correlation": (
                            0.9874
                        ),
                    },
                },
                "error": None,
            }
        ],
        "final_answer": (
            "散点图已经生成。"
        ),
        "error": None,
    }

    report = generate_markdown_report(
        state
    )

    assert "## Evidence Summary" in report
    assert "生成 scatter 图" in report
    assert (
        "参与绘图数据为 40 条"
        in report
    )
    assert (
        "pearson_correlation=0.9874"
        in report
    )
    assert (
        "study_hours_score.png"
        in report
    )

    # 报告摘要本身不根据相关系数追加解释。
    assert "强相关" not in report
    assert "弱相关" not in report
    assert "导致成绩提高" not in report


def test_report_summarizes_failed_tool() -> None:
    state = {
        "dataset_path": "data/sample.csv",
        "question": "统计不存在的字段。",
        "task_type": "sql_analysis",
        "workflow_status": "failed",
        "step_count": 1,
        "plan": [
            "执行只读 SQL",
        ],
        "tool_calls": [
            {
                "id": "call_sql",
                "tool_name": (
                    "run_readonly_sql"
                ),
                "arguments": {
                    "dataset_path": (
                        "data/sample.csv"
                    ),
                    "query": (
                        "SELECT unknown_column "
                        "FROM dataset"
                    ),
                },
            }
        ],
        "tool_results": [
            {
                "tool_name": (
                    "run_readonly_sql"
                ),
                "success": False,
                "result": None,
                "error": (
                    "字段 unknown_column 不存在。"
                ),
            }
        ],
        "final_answer": (
            "无法完成当前查询。"
        ),
        "error": (
            "字段 unknown_column 不存在。"
        ),
    }

    report = generate_markdown_report(
        state
    )

    assert (
        "`run_readonly_sql`：失败"
        in report
    )
    assert (
        "字段 unknown_column 不存在"
        in report
    )
    assert "## Workflow Error" in report


def test_report_summarizes_causal_estimator_comparison() -> None:
    state = {
        "dataset_path": "data/synthetic_causal_data.csv",
        "question": "估计 ATE",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "plan": [],
        "tool_calls": [
            {"tool_name": "run_causal_estimators", "arguments": {}}
        ],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "primary_estimator": "AIPW",
                    "sample_size": 1000,
                    "estimates": [
                        {
                            "estimator": "AIPW",
                            "ate": 3.05,
                            "confidence_interval": [2.7, 3.4],
                            "sensitivity_analysis": {
                                "robustness_value_point_estimate": 0.18,
                                "robustness_value_significance": 0.12,
                                "unmeasured_confounding_risk": "lower",
                                "observed_confounder_benchmark": {
                                    "available": True,
                                    "strongest_confounder": "activity",
                                    "assessment": (
                                        "stronger_than_observed_benchmark_required"
                                    ),
                                    "confounders": [
                                        {
                                            "confounder": "activity",
                                            "partial_r2_treatment": 0.04,
                                            "partial_r2_outcome": 0.09,
                                            "equivalent_robustness_value": 0.06,
                                            "ratio_to_point_rv": 0.33,
                                        }
                                    ],
                                },
                            },
                        },
                        {
                            "estimator": "LinearDML",
                            "ate": 3.14,
                            "confidence_interval": [2.8, 3.48],
                        },
                    ],
                    "comparison": {
                        "absolute_ate_difference": 0.09,
                        "confidence_intervals_overlap": True,
                    },
                    "negative_control_summary": {
                        "requested_outcomes": ["pre_treatment_score"],
                        "estimator": "AIPW",
                        "passed_count": 1,
                        "result_count": 1,
                        "all_passed": True,
                        "assumption": "Known not to be caused by treatment.",
                        "results": [
                            {
                                "outcome": "pre_treatment_score",
                                "estimate": 0.03,
                                "confidence_interval": [-0.1, 0.16],
                                "passed": True,
                            }
                        ],
                    },
                    "permutation_placebo": {
                        "available": True,
                        "method": "residualized_treatment_permutation",
                        "diagnostic_statistic": (
                            "covariate_adjusted_linear_projection"
                        ),
                        "iterations": 200,
                        "pseudo_treatment_type": (
                            "linear_fitted_plus_permuted_residual"
                        ),
                        "observed_adjusted_coefficient": 3.02,
                        "placebo_mean": 0.01,
                        "placebo_quantiles": {
                            "q025": -0.3,
                            "q50": 0.0,
                            "q975": 0.31,
                        },
                        "empirical_two_sided_p_value": 0.005,
                        "signal_status": "stronger_than_permuted_labels",
                        "calibration_passed": True,
                    },
                    "evidence_grade": {
                        "grade": "stronger_support",
                        "label_zh": "较强支持",
                        "supporting_signals": [
                            "primary_interval_excludes_zero",
                            "all_declared_negative_controls_passed",
                        ],
                        "red_flags": [],
                        "missing_checks": [],
                    },
                },
                "error": None,
            }
        ],
        "final_answer": "已完成稳健性比较。",
    }

    report = generate_markdown_report(state)

    assert "主估计器为 AIPW" in report
    assert "AIPW ATE=3.05" in report
    assert "LinearDML ATE=3.14" in report
    assert "估计值绝对差为 0.09" in report
    assert "置信区间是否重叠：True" in report
    assert "RV(point/significance)=0.18/0.12" in report
    assert "风险分级=lower" in report
    assert "## Unmeasured Confounding Sensitivity" in report
    assert "| AIPW | 0.18 | 0.12 | lower |" in report
    assert "### Observed-confounder Benchmark" in report
    assert "| activity | 0.04 | 0.09 | 0.06 | 0.33 |" in report
    assert "## Negative-Control Outcome Checks" in report
    assert "| pre_treatment_score | 0.03 | [-0.1, 0.16] | True |  |" in report
    assert "## Residualized Treatment-Permutation Diagnostic" in report
    assert "- Empirical two-sided p-value: `0.005`" in report
    assert "因果证据等级=较强支持 (stronger_support)" in report
    assert "## Causal Evidence Grade" in report
    assert "- Grade: `stronger_support`" in report
    assert "- Red flags: `[]`" in report


def test_report_summarizes_iv_diagnostics() -> None:
    state = {
        "question": "使用 encouragement 估计 treatment 的 LATE",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "plan": [],
        "tool_calls": [],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "estimand": "LATE",
                    "primary_estimator": "TwoStageLeastSquares",
                    "instruments": ["encouragement", "distance"],
                    "sample_size": 900,
                    "estimates": [
                        {
                            "estimator": "TwoStageLeastSquares",
                            "ate": 1.95,
                            "confidence_interval": [1.7, 2.2],
                            "first_stage": {
                                "f_statistic": 64.0,
                                "partial_r2": 0.28,
                                "weak_instrument_warning": False,
                            },
                            "endogeneity_test": {"p_value": 0.01},
                            "overidentification_test": {
                                "available": True,
                                "p_value": 0.35,
                                "passed": True,
                            },
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                },
                "error": None,
            }
        ],
        "final_answer": "IV 分析完成。",
    }

    report = generate_markdown_report(state)

    assert "## Instrumental-Variable Diagnostics" in report
    assert "- First-stage F statistic: `64.0`" in report
    assert "- First-stage partial R²: `0.28`" in report
    assert "- Hansen J p-value: `0.35`" in report
    assert "exclusion restriction" in report


def test_report_summarizes_rdd_diagnostics() -> None:
    state = {
        "question": "估计 cutoff 处的局部效应",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "plan": [],
        "tool_calls": [],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "estimand": "LATE",
                    "primary_estimator": "SharpRegressionDiscontinuity",
                    "sample_size": 410,
                    "estimates": [
                        {
                            "estimator": "SharpRegressionDiscontinuity",
                            "ate": 1.8,
                            "confidence_interval": [1.4, 2.2],
                            "running_variable": "risk_score",
                            "cutoff": 50.0,
                            "bandwidth": 5.0,
                            "local_support": {
                                "left_count": 200,
                                "right_count": 210,
                            },
                            "density_diagnostic": {
                                "method": "local_symmetric_count_test",
                                "p_value": 0.55,
                            },
                            "covariate_continuity": {"all_passed": True},
                            "bandwidth_sensitivity": {
                                "direction_stable": True,
                            },
                            "donut_sensitivity": {"estimate": 1.75},
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                },
                "error": None,
            }
        ],
        "final_answer": "RDD 分析完成。",
    }

    report = generate_markdown_report(state)

    assert "## Regression-Discontinuity Diagnostics" in report
    assert "- Running variable: `risk_score`" in report
    assert "- Cutoff: `50.0`" in report
    assert "- Local left / right observations: `200 / 210`" in report
    assert "not a formal McCrary test" in report


def test_report_summarizes_cate_heterogeneity() -> None:
    state = {
        "dataset_path": "data/heterogeneous.csv",
        "question": "估计 age 修饰的 CATE",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "plan": [],
        "tool_calls": [
            {"tool_name": "run_causal_estimators", "arguments": {}}
        ],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "estimand": "CATE",
                    "primary_estimator": "CausalForestDML",
                    "sample_size": 1000,
                    "estimates": [
                        {
                            "estimator": "CausalForestDML",
                            "ate": 2.1,
                            "confidence_interval": [1.7, 2.5],
                            "heterogeneity": {
                                "cate_p10": 0.8,
                                "cate_median": 2.0,
                                "cate_p90": 3.4,
                            },
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                },
                "error": None,
            }
        ],
        "final_answer": "已完成 CATE 探索。",
    }

    report = generate_markdown_report(state)

    assert "CausalForestDML CATE=2.1" in report
    assert "CATE 分布 P10/Median/P90=0.8/2/3.4" in report


def test_report_summarizes_difference_in_differences() -> None:
    state = {
        "dataset_path": "data/did.csv",
        "question": "估计政策的 DID ATT",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "plan": [],
        "tool_calls": [
            {"tool_name": "run_causal_estimators", "arguments": {}}
        ],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "estimand": "ATT",
                    "primary_estimator": "DifferenceInDifferences",
                    "sample_size": 200,
                    "estimates": [
                        {
                            "estimator": "DifferenceInDifferences",
                            "ate": 2.4,
                            "confidence_interval": [1.9, 2.9],
                            "unadjusted_did": 2.5,
                            "cell_means": {
                                "group_0_post_0": 5.0,
                                "group_0_post_1": 6.0,
                                "group_1_post_0": 5.5,
                                "group_1_post_1": 9.0,
                            },
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                },
                "error": None,
            }
        ],
        "causal_question": {
            "treatment": "treated_group",
            "outcome": "outcome",
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "post_column": "post",
            "unit_id_column": None,
            "treatment_contrast": {"control": 0, "treated": 1},
            "variable_roles": {"confounders": []},
        },
        "causal_diagnostics": {
            "analysis_type": "did_design_diagnostics",
            "passed": True,
            "design_summary": {
                "cell_counts": {"group_0_post_0": 50},
                "cell_means": {"group_0_post_0": 5.0},
            },
            "warnings": ["平行趋势需要研究设计论证。"],
        },
        "final_answer": "已完成 DID。",
    }

    report = generate_markdown_report(state)

    assert "DifferenceInDifferences ATT=2.4" in report
    assert "未调整四单元格 DiD=2.5" in report
    assert "### DiD Design Diagnostics" in report
    assert "平行趋势需要研究设计论证" in report


def test_report_summarizes_event_study_dynamic_effects() -> None:
    state = {
        "dataset_path": "data/event.csv",
        "question": "估计动态 ATT",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "plan": [],
        "tool_calls": [{"tool_name": "run_causal_estimators", "arguments": {}}],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "estimand": "ATT",
                    "primary_estimator": "EventStudy",
                    "sample_size": 240,
                    "estimates": [
                        {
                            "estimator": "EventStudy",
                            "ate": 2.0,
                            "confidence_interval": [1.7, 2.3],
                            "dynamic_effects": [
                                {"event_time": -1, "estimate": 0.0},
                                {"event_time": 0, "estimate": 1.5},
                            ],
                            "pretrend_test": {"p_value": 0.61},
                            "chart_path": "outputs/charts/event.png",
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                },
                "error": None,
            }
        ],
        "causal_question": {
            "treatment": "treated_group",
            "outcome": "outcome",
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "event_time_column": "event_time",
            "reference_period": -1,
            "unit_id_column": "unit_id",
            "variable_roles": {"confounders": []},
        },
        "causal_diagnostics": {
            "analysis_type": "event_study_design_diagnostics",
            "passed": True,
            "design_summary": {
                "pre_periods": [-3, -2, -1],
                "post_periods": [0, 1, 2],
                "cell_counts": {"group_0_event_-3": 20},
            },
        },
        "final_answer": "已完成 Event Study。",
    }

    report = generate_markdown_report(state)

    assert "EventStudy ATT=2" in report
    assert "联合预趋势检验 p=0.61" in report
    assert "### Event Study Design Diagnostics" in report
    assert "Event time column: `event_time`" in report


def test_report_summarizes_staggered_adoption_did() -> None:
    state = {
        "dataset_path": "data/staggered.csv",
        "question": "估计错位处理 ATT",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "plan": [],
        "tool_calls": [{"tool_name": "run_causal_estimators", "arguments": {}}],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "estimand": "ATT",
                    "primary_estimator": "StaggeredAdoptionDID",
                    "sample_size": 720,
                    "estimates": [
                        {
                            "estimator": "StaggeredAdoptionDID",
                            "ate": 2.1,
                            "confidence_interval": [1.8, 2.4],
                            "group_time_effects": [
                                {"cohort": 2, "time": 2, "estimate": 1.0}
                            ],
                            "dynamic_effects": [
                                {"event_time": 0, "estimate": 1.0}
                            ],
                            "pretrend_test": {"p_value": 0.55},
                            "placebo_test": {"p_value": 0.72},
                            "cohort_sensitivity": {
                                "max_absolute_deviation": 0.18,
                                "leave_one_cohort_out": [],
                            },
                            "model_configuration": {
                                "adjustment": "cross_fitted_doubly_robust",
                                "nuisance_diagnostics": {
                                    "min_weighted_control_ess": 22.0,
                                },
                            },
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                },
                "error": None,
            }
        ],
        "causal_question": {
            "treatment": "treated",
            "outcome": "outcome",
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "time_column": "period",
            "treatment_start_column": "first_treated_period",
            "unit_id_column": "unit_id",
            "variable_roles": {"confounders": []},
        },
        "causal_diagnostics": {
            "analysis_type": "staggered_did_design_diagnostics",
            "passed": True,
            "design_summary": {
                "cohorts": [2, 4],
                "never_treated_units": 30,
            },
        },
        "final_answer": "已完成错位处理 DID。",
    }

    report = generate_markdown_report(state)

    assert "StaggeredAdoptionDID ATT=2.1" in report
    assert "cohort-time ATT 单元数=1" in report
    assert "联合预趋势检验 p=0.55" in report
    assert "提前一期 placebo p=0.72" in report
    assert "逐 cohort 剔除最大偏移=0.18" in report
    assert "### Staggered DiD Design Diagnostics" in report


def test_report_summarizes_synthetic_control_and_quality_gate() -> None:
    state = {
        "dataset_path": "data/panel.csv",
        "question": "估计政策 ATT",
        "task_type": "causal_analysis",
        "workflow_status": "completed",
        "plan": [],
        "tool_calls": [{"tool_name": "run_causal_estimators", "arguments": {}}],
        "tool_results": [
            {
                "tool_name": "run_causal_estimators",
                "success": True,
                "result": {
                    "estimand": "ATT",
                    "primary_estimator": "SyntheticControl",
                    "sample_size": 108,
                    "estimates": [
                        {
                            "estimator": "SyntheticControl",
                            "ate": 2.5,
                            "confidence_interval": [2.1, 2.9],
                            "treated_unit": "city_a",
                            "intervention_time": 8,
                            "fit_diagnostics": {
                                "pre_rmspe": 0.02,
                                "post_rmspe": 2.5,
                                "post_pre_rmspe_ratio": 125.0,
                            },
                            "placebo_inference": {
                                "placebo_count": 8,
                                "p_value": 0.05,
                            },
                            "donor_weights": [
                                {"unit": "city_b", "weight": 1.0}
                            ],
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                    "data_quality": {
                        "ready": True,
                        "blocking_issues": [],
                        "warnings": [],
                        "sample_summary": {"complete_case_fraction": 1.0},
                        "preprocessing_policy": {"automatic_imputation": False},
                    },
                },
                "error": None,
            }
        ],
        "causal_question": {
            "treatment": "treated",
            "outcome": "outcome",
            "estimand": "ATT",
            "design_type": "synthetic_control",
            "unit_id_column": "unit",
            "time_column": "period",
            "intervention_time": 8,
            "treated_unit": "city_a",
            "treatment_contrast": {"control": 0, "treated": 1},
            "variable_roles": {"confounders": []},
        },
        "final_answer": "已完成合成控制。",
    }

    report = generate_markdown_report(state)

    assert "SyntheticControl ATT=2.5" in report
    assert "## Synthetic-Control Diagnostics" in report
    assert "Placebo rank p-value: `0.05`" in report
    assert "## Causal Data Quality Gate" in report
    assert "Intervention time: `8`" in report

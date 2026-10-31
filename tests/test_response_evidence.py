from causalens.response_evidence import build_evidence_rows
from causalens.schemas import ToolResult


def test_build_evidence_rows_for_successful_tools():
    tool_results = [
        ToolResult(
            tool_name="profile_dataset",
            success=True,
            result={
                "rows": 40,
                "columns": 8,
                "column_names": [
                    "student_id",
                    "major",
                    "city",
                    "study_hours",
                    "attendance_rate",
                    "score",
                    "passed",
                    "exam_date",
                ],
            },
        ),
        ToolResult(
            tool_name="run_readonly_sql",
            success=True,
            result={
                "returned_row_count": 3,
                "columns": [
                    "major",
                    "avg_score",
                    "student_count",
                ],
                "truncated": False,
            },
        ),
        ToolResult(
            tool_name="create_chart",
            success=True,
            result={
                "chart_path": (
                    "outputs/charts/"
                    "study_hours_vs_score_test.png"
                ),
                "chart_type": "scatter",
                "plotted_row_count": 40,
                "relationship_summary": {
                    "point_count": 40,
                    "x_min": 3.0,
                    "x_max": 18.0,
                    "y_min": 52.0,
                    "y_max": 98.0,
                    "pearson_correlation": 0.9874,
                },
            },
        ),
    ]

    rows = build_evidence_rows(tool_results)

    assert len(rows) == 3

    assert rows[0]["tool_name"] == "profile_dataset"
    assert rows[0]["status"] == "成功"
    assert "40 行 × 8 列" in rows[0]["evidence"]
    assert "student_id、major、city" in rows[0]["evidence"]

    assert rows[1]["tool_name"] == "run_readonly_sql"
    assert rows[1]["status"] == "成功"
    assert "SQL 返回 3 行" in rows[1]["evidence"]
    assert "major、avg_score、student_count" in rows[1]["evidence"]

    assert rows[2]["tool_name"] == "create_chart"
    assert rows[2]["status"] == "成功"
    assert "生成 scatter 图" in rows[2]["evidence"]
    assert "参与绘图数据：40 条" in rows[2]["evidence"]
    assert "pearson_correlation=0.9874" in rows[2]["evidence"]


def test_build_evidence_rows_for_failed_tool():
    tool_results = [
        ToolResult(
            tool_name="create_chart",
            success=False,
            error="无法生成散点图：x_column=major 必须是数值列。",
        ),
    ]

    rows = build_evidence_rows(tool_results)

    assert rows == [
        {
            "tool_name": "create_chart",
            "status": "失败",
            "evidence": (
                "错误：无法生成散点图："
                "x_column=major 必须是数值列。"
            ),
        }
    ]

def test_build_evidence_rows_formats_retrieve_docs_result() -> None:
    tool_results = [
        ToolResult(
            tool_name="retrieve_docs",
            success=True,
            result={
                "query": "passed 字段是什么意思？",
                "results": [
                    {
                        "source": "data_dictionary.md",
                        "chunk_id": "data_dictionary_0008",
                        "distance": 0.1234,
                        "content": "1 means passed. 0 means failed.",
                    }
                ],
            },
        )
    ]

    rows = build_evidence_rows(tool_results)

    assert len(rows) == 1
    assert rows[0]["tool_name"] == "retrieve_docs"
    assert rows[0]["status"] == "成功"
    assert "检索问题：passed 字段是什么意思？" in rows[0]["evidence"]
    assert "返回片段数：1" in rows[0]["evidence"]
    assert "data_dictionary.md" in rows[0]["evidence"]
    assert "data_dictionary_0008" in rows[0]["evidence"]


def test_build_evidence_rows_formats_causal_estimator_suite() -> None:
    tool_results = [
        ToolResult(
            tool_name="run_causal_estimators",
            success=True,
            result={
                "primary_estimator": "AIPW",
                "sample_size": 1200,
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
                                "strongest_equivalent_robustness_value": 0.08,
                                "assessment": (
                                    "stronger_than_observed_benchmark_required"
                                ),
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
                    "passed_count": 1,
                    "result_count": 1,
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
                    "observed_adjusted_coefficient": 3.02,
                    "placebo_mean": 0.01,
                    "empirical_two_sided_p_value": 0.005,
                    "signal_status": "stronger_than_permuted_labels",
                },
                "evidence_grade": {
                    "grade": "stronger_support",
                    "label_zh": "较强支持",
                    "red_flags": [],
                    "missing_checks": [],
                },
            },
        )
    ]

    rows = build_evidence_rows(tool_results)

    assert rows[0]["tool_name"] == "run_causal_estimators"
    assert "主估计器：AIPW" in rows[0]["evidence"]
    assert "AIPW: ATE=3.05" in rows[0]["evidence"]
    assert "LinearDML: ATE=3.14" in rows[0]["evidence"]
    assert "估计值绝对差：0.09" in rows[0]["evidence"]
    assert "RV(point)=0.18" in rows[0]["evidence"]
    assert "风险分级=lower" in rows[0]["evidence"]
    assert "最强已观测混杂基准=activity" in rows[0]["evidence"]
    assert "负对照 outcome：1/1 通过" in rows[0]["evidence"]
    assert "残差化伪 treatment" in rows[0]["evidence"]
    assert "empirical p=0.005" in rows[0]["evidence"]
    assert "因果证据等级：较强支持 (stronger_support)" in rows[0]["evidence"]


def test_build_evidence_rows_formats_iv_diagnostics() -> None:
    rows = build_evidence_rows(
        [
            ToolResult(
                tool_name="run_causal_estimators",
                success=True,
                result={
                    "estimand": "LATE",
                    "primary_estimator": "TwoStageLeastSquares",
                    "sample_size": 800,
                    "estimates": [
                        {
                            "estimator": "TwoStageLeastSquares",
                            "ate": 2.01,
                            "confidence_interval": [1.8, 2.22],
                            "first_stage": {
                                "f_statistic": 85.4,
                                "partial_r2": 0.31,
                                "weak_instrument_warning": False,
                            },
                            "endogeneity_test": {"p_value": 0.004},
                            "overidentification_test": {
                                "available": True,
                                "p_value": 0.42,
                                "passed": True,
                            },
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                },
            )
        ]
    )

    evidence = rows[0]["evidence"]
    assert "TwoStageLeastSquares: LATE=2.01" in evidence
    assert "IV 第一阶段：F=85.4" in evidence
    assert "partial R²=0.31" in evidence
    assert "Hansen J p=0.42" in evidence


def test_build_evidence_rows_formats_rdd_diagnostics() -> None:
    rows = build_evidence_rows(
        [
            ToolResult(
                tool_name="run_causal_estimators",
                success=True,
                result={
                    "estimand": "LATE",
                    "primary_estimator": "SharpRegressionDiscontinuity",
                    "sample_size": 420,
                    "estimates": [
                        {
                            "estimator": "SharpRegressionDiscontinuity",
                            "ate": 2.0,
                            "confidence_interval": [1.6, 2.4],
                            "running_variable": "risk_score",
                            "cutoff": 50.0,
                            "bandwidth": 6.0,
                            "local_support": {
                                "left_count": 205,
                                "right_count": 215,
                            },
                            "density_diagnostic": {"p_value": 0.6},
                            "covariate_continuity": {"all_passed": True},
                            "bandwidth_sensitivity": {
                                "direction_stable": True,
                            },
                            "donut_sensitivity": {"estimate": 1.95},
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                },
            )
        ]
    )

    evidence = rows[0]["evidence"]
    assert "SharpRegressionDiscontinuity: LATE=2" in evidence
    assert "running variable=risk_score" in evidence
    assert "cutoff=50" in evidence
    assert "断点两侧样本=205/215" in evidence
    assert "带宽方向稳定=True" in evidence


def test_build_evidence_rows_formats_synthetic_control_and_quality() -> None:
    rows = build_evidence_rows(
        [
            ToolResult(
                tool_name="run_causal_estimators",
                success=True,
                result={
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
                                "pre_rmspe": 0.03,
                                "post_pre_rmspe_ratio": 80.0,
                            },
                            "placebo_inference": {"p_value": 0.05},
                            "donor_weights": [{"unit": "city_b", "weight": 1.0}],
                        }
                    ],
                    "comparison": {"estimate_count": 1},
                    "data_quality": {
                        "ready": True,
                        "sample_summary": {"complete_case_fraction": 1.0},
                    },
                    "design_selection": {
                        "study_design": "synthetic_control_single_treated_unit",
                        "selected_estimator_names": ["SyntheticControl"],
                    },
                },
            )
        ]
    )

    evidence = rows[0]["evidence"]
    assert "SyntheticControl: ATT=2.5" in evidence
    assert "treated unit=city_a" in evidence
    assert "pre RMSPE=0.03" in evidence
    assert "数据质量门禁：ready=True" in evidence
    assert "确定性方法路由" in evidence

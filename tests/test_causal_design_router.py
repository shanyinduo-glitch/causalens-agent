from causalens.causal.design_router import select_causal_design


def test_router_selects_local_projection_for_continuous_time_series():
    result = select_causal_design(
        design_type="time_series",
        estimand="IRF",
        time_column="date",
    )

    assert result["ready"] is True
    assert result["data_structure"] == "time_series"
    assert result["selected_estimators"] == ["time_series_local_projection"]


def test_router_selects_synthetic_control_from_explicit_contract():
    result = select_causal_design(
        design_type="synthetic_control",
        estimand="ATT",
        unit_id_column="unit",
        time_column="period",
        intervention_time=8,
    )

    assert result["ready"] is True
    assert result["primary_estimator"] == "SyntheticControl"
    assert result["data_structure"] == "panel"


def test_router_selects_event_study_over_two_period_did():
    result = select_causal_design(
        design_type="difference_in_differences",
        estimand="ATT",
        post_column="post",
        event_time_column="event_time",
        unit_id_column="unit",
    )

    assert result["ready"] is True
    assert result["primary_estimator"] == "EventStudy"


def test_router_blocks_incomplete_rdd_contract():
    result = select_causal_design(
        design_type="regression_discontinuity",
        estimand="LATE",
        running_variable="score",
    )

    assert result["ready"] is False
    assert result["selected_estimators"] == []

from causalens.agent.causal_specification import (
    build_causal_question_draft,
    validate_causal_configuration_payload,
)


AVAILABLE_COLUMNS = {
    "ad_exposure",
    "retention_score",
    "age",
    "historical_activity",
    "segment",
    "pre_treatment_score",
}


def test_extended_causal_configuration_is_structured():
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "confounders": ["age", "historical_activity"],
            "effect_modifiers": ["segment"],
            "negative_control_outcomes": ["pre_treatment_score"],
            "estimand": "ATT",
            "target_population": "已注册用户",
            "unit_of_analysis": "user",
            "outcome_horizon": "曝光后 30 天",
        },
        available_columns=AVAILABLE_COLUMNS,
    )

    assert errors == []
    assert configuration is not None
    draft = build_causal_question_draft(configuration)
    assert draft.estimand == "ATT"
    assert draft.variable_roles.effect_modifiers == ["segment"]
    assert draft.variable_roles.negative_control_outcomes == [
        "pre_treatment_score"
    ]
    assert draft.missing_design_fields == []


def test_post_treatment_role_conflict_is_rejected():
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "confounders": ["historical_activity"],
            "mediators": ["historical_activity"],
        },
        available_columns=AVAILABLE_COLUMNS,
    )

    assert configuration is None
    assert any("confounder" in error for error in errors)


def test_unknown_effect_modifier_is_rejected():
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "confounders": ["age"],
            "effect_modifiers": ["missing_segment"],
        },
        available_columns=AVAILABLE_COLUMNS,
    )

    assert configuration is None
    assert any("missing_segment" in error for error in errors)


def test_explicit_dag_edges_are_preserved():
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "confounders": ["age"],
            "dag_edges": [
                {"source": "age", "target": "ad_exposure"},
                {"source": "age", "target": "retention_score"},
                {"source": "ad_exposure", "target": "retention_score"},
            ],
        },
        available_columns=AVAILABLE_COLUMNS,
    )

    assert errors == []
    assert configuration is not None
    assert configuration["causal_dag_edges"] == [
        ("age", "ad_exposure"),
        ("age", "retention_score"),
        ("ad_exposure", "retention_score"),
    ]


def test_did_configuration_preserves_design_columns() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treated_group",
            "outcome": "outcome",
            "confounders": ["baseline"],
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "post_column": "post",
            "unit_id_column": "unit_id",
        },
        available_columns={
            "treated_group",
            "outcome",
            "baseline",
            "post",
            "unit_id",
        },
    )

    assert errors == []
    assert configuration is not None
    question = configuration["causal_question"]
    assert question["design_type"] == "difference_in_differences"
    assert question["post_column"] == "post"
    assert question["unit_id_column"] == "unit_id"


def test_did_configuration_rejects_missing_post_column() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treated_group",
            "outcome": "outcome",
            "confounders": ["baseline"],
            "estimand": "ATT",
            "design_type": "difference_in_differences",
        },
        available_columns={"treated_group", "outcome", "baseline"},
    )

    assert configuration is None
    assert any("post_column" in error for error in errors)


def test_did_configuration_allows_no_optional_confounders() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treated_group",
            "outcome": "outcome",
            "confounders": [],
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "post_column": "post",
        },
        available_columns={"treated_group", "outcome", "post"},
    )

    assert errors == []
    assert configuration is not None
    assert configuration["confounders"] == []


def test_event_study_configuration_preserves_relative_time_contract() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treated_group",
            "outcome": "outcome",
            "confounders": [],
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "event_time_column": "event_time",
            "reference_period": -1,
            "unit_id_column": "unit_id",
        },
        available_columns={
            "treated_group",
            "outcome",
            "event_time",
            "unit_id",
        },
    )

    assert errors == []
    assert configuration is not None
    question = configuration["causal_question"]
    assert question["event_time_column"] == "event_time"
    assert question["reference_period"] == -1
    assert question["post_column"] is None


def test_staggered_did_configuration_preserves_timing_contract() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treated",
            "outcome": "outcome",
            "confounders": [],
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "time_column": "period",
            "treatment_start_column": "first_treated_period",
            "unit_id_column": "unit_id",
        },
        available_columns={
            "treated",
            "outcome",
            "period",
            "first_treated_period",
            "unit_id",
        },
    )

    assert errors == []
    assert configuration is not None
    question = configuration["causal_question"]
    assert question["time_column"] == "period"
    assert question["treatment_start_column"] == "first_treated_period"
    assert question["unit_id_column"] == "unit_id"


def test_iv_configuration_requires_explicit_instrument_and_late() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treatment",
            "outcome": "outcome",
            "confounders": ["age"],
            "instruments": ["encouragement"],
            "estimand": "LATE",
            "design_type": "instrumental_variable",
        },
        available_columns={
            "treatment",
            "outcome",
            "age",
            "encouragement",
        },
    )

    assert errors == []
    assert configuration is not None
    question = configuration["causal_question"]
    assert question["design_type"] == "instrumental_variable"
    assert question["estimand"] == "LATE"
    assert question["variable_roles"]["instruments"] == ["encouragement"]


def test_iv_configuration_rejects_missing_instrument() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treatment",
            "outcome": "outcome",
            "confounders": [],
            "estimand": "LATE",
            "design_type": "instrumental_variable",
        },
        available_columns={"treatment", "outcome"},
    )

    assert configuration is None
    assert any("instrument" in error for error in errors)


def test_rdd_configuration_preserves_cutoff_contract() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "eligible",
            "outcome": "score",
            "confounders": ["baseline"],
            "estimand": "LATE",
            "design_type": "regression_discontinuity",
            "running_variable": "risk_score",
            "cutoff": 50.0,
            "treatment_above_cutoff": True,
            "bandwidth": 8.0,
        },
        available_columns={
            "eligible",
            "score",
            "baseline",
            "risk_score",
        },
    )

    assert errors == []
    assert configuration is not None
    question = configuration["causal_question"]
    assert question["design_type"] == "regression_discontinuity"
    assert question["running_variable"] == "risk_score"
    assert question["cutoff"] == 50.0
    assert question["bandwidth"] == 8.0


def test_rdd_configuration_requires_cutoff() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "eligible",
            "outcome": "score",
            "confounders": [],
            "estimand": "LATE",
            "design_type": "regression_discontinuity",
            "running_variable": "risk_score",
        },
        available_columns={"eligible", "score", "risk_score"},
    )

    assert configuration is None
    assert any("cutoff" in error for error in errors)


def test_synthetic_control_configuration_preserves_panel_contract() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treated",
            "outcome": "outcome",
            "confounders": ["baseline_size"],
            "estimand": "ATT",
            "design_type": "synthetic_control",
            "unit_id_column": "unit",
            "time_column": "period",
            "intervention_time": 8,
            "treated_unit": "city_a",
        },
        available_columns={
            "treated",
            "outcome",
            "baseline_size",
            "unit",
            "period",
        },
    )

    assert errors == []
    assert configuration is not None
    question = configuration["causal_question"]
    assert question["design_type"] == "synthetic_control"
    assert question["intervention_time"] == 8.0
    assert question["treated_unit"] == "city_a"


def test_synthetic_control_configuration_requires_intervention_time() -> None:
    configuration, errors = validate_causal_configuration_payload(
        payload={
            "treatment": "treated",
            "outcome": "outcome",
            "confounders": [],
            "estimand": "ATT",
            "design_type": "synthetic_control",
            "unit_id_column": "unit",
            "time_column": "period",
        },
        available_columns={"treated", "outcome", "unit", "period"},
    )

    assert configuration is None
    assert any("intervention_time" in error for error in errors)

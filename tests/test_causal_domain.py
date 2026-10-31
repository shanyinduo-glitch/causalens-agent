import pytest
from pydantic import ValidationError

from causalens.domain import (
    CausalQuestion,
    CausalQuestionDraft,
    StudyDesign,
    TreatmentContrast,
    VariableRoles,
)


def make_question() -> CausalQuestion:
    return CausalQuestion(
        treatment="ad_exposure",
        outcome="retention_score",
        estimand="ATE",
        treatment_contrast=TreatmentContrast(
            control=0,
            treated=1,
        ),
        target_population="当前产品用户",
        unit_of_analysis="user",
        outcome_horizon="曝光后 30 天",
        variable_roles=VariableRoles(
            confounders=["age", "historical_activity"],
            effect_modifiers=["age"],
        ),
    )


def test_causal_question_captures_estimand_contract():
    question = make_question()

    assert question.estimand == "ATE"
    assert question.treatment_contrast.control == 0
    assert question.variable_roles.confounders == [
        "age",
        "historical_activity",
    ]


def test_treatment_and_outcome_must_be_distinct():
    with pytest.raises(ValidationError):
        CausalQuestion(
            treatment="score",
            outcome="score",
            treatment_contrast={"control": 0, "treated": 1},
            target_population="students",
            unit_of_analysis="student",
            outcome_horizon="semester end",
        )


def test_confounder_cannot_also_be_collider():
    with pytest.raises(ValidationError):
        VariableRoles(
            confounders=["engagement"],
            colliders=["engagement"],
        )


def test_negative_control_outcome_is_distinct_from_confounders():
    roles = VariableRoles(
        confounders=["age"],
        negative_control_outcomes=["pre_treatment_score"],
    )

    assert roles.negative_control_outcomes == ["pre_treatment_score"]
    with pytest.raises(ValidationError):
        VariableRoles(
            confounders=["age"],
            negative_control_outcomes=["age"],
        )


def test_study_design_rejects_dag_self_loop():
    with pytest.raises(ValidationError):
        StudyDesign(
            question=make_question(),
            dag_edges=[("age", "age")],
        )


def test_draft_preserves_missing_design_context():
    draft = CausalQuestionDraft(
        treatment="ad_exposure",
        outcome="retention_score",
        variable_roles=VariableRoles(confounders=["age"]),
    )

    assert draft.missing_design_fields == [
        "target_population",
        "unit_of_analysis",
        "outcome_horizon",
    ]
    with pytest.raises(ValueError, match="仍缺少研究设计字段"):
        draft.confirm()


def test_iv_question_requires_instrument_and_late_estimand() -> None:
    with pytest.raises(ValidationError, match="instruments"):
        CausalQuestion(
            treatment="treatment",
            outcome="outcome",
            estimand="LATE",
            design_type="instrumental_variable",
            treatment_contrast={"control": 0, "treated": 1},
            target_population="users",
            unit_of_analysis="user",
            outcome_horizon="30 days",
        )

    question = CausalQuestion(
        treatment="treatment",
        outcome="outcome",
        estimand="LATE",
        design_type="instrumental_variable",
        treatment_contrast={"control": 0, "treated": 1},
        target_population="users",
        unit_of_analysis="user",
        outcome_horizon="30 days",
        variable_roles=VariableRoles(instruments=["encouragement"]),
    )

    assert question.variable_roles.instruments == ["encouragement"]


def test_rdd_question_requires_running_variable_and_cutoff() -> None:
    with pytest.raises(ValidationError, match="running_variable"):
        CausalQuestion(
            treatment="eligible",
            outcome="score",
            estimand="LATE",
            design_type="regression_discontinuity",
            treatment_contrast={"control": 0, "treated": 1},
            target_population="applicants",
            unit_of_analysis="applicant",
            outcome_horizon="one year",
        )

    question = CausalQuestion(
        treatment="eligible",
        outcome="score",
        estimand="LATE",
        design_type="regression_discontinuity",
        running_variable="risk_score",
        cutoff=50.0,
        bandwidth=5.0,
        treatment_contrast={"control": 0, "treated": 1},
        target_population="applicants",
        unit_of_analysis="applicant",
        outcome_horizon="one year",
    )

    assert question.running_variable == "risk_score"
    assert question.cutoff == 50.0


def test_synthetic_control_question_requires_panel_contract() -> None:
    with pytest.raises(ValidationError, match="intervention_time"):
        CausalQuestion(
            treatment="treated",
            outcome="outcome",
            estimand="ATT",
            design_type="synthetic_control",
            unit_id_column="unit",
            time_column="period",
            treatment_contrast={"control": 0, "treated": 1},
            target_population="cities",
            unit_of_analysis="city",
            outcome_horizon="four years",
        )

    question = CausalQuestion(
        treatment="treated",
        outcome="outcome",
        estimand="ATT",
        design_type="synthetic_control",
        unit_id_column="unit",
        time_column="period",
        intervention_time=8,
        treatment_contrast={"control": 0, "treated": 1},
        target_population="cities",
        unit_of_analysis="city",
        outcome_horizon="four years",
    )
    assert question.intervention_time == 8


def test_design_columns_cannot_also_be_confounders() -> None:
    with pytest.raises(ValidationError, match="设计专属字段"):
        CausalQuestion(
            treatment="shock",
            outcome="return",
            estimand="IRF",
            design_type="time_series",
            time_column="date",
            treatment_contrast={"control": 0, "treated": 1},
            target_population="trading days",
            unit_of_analysis="day",
            outcome_horizon="five days",
            variable_roles=VariableRoles(confounders=["date"]),
        )

    with pytest.raises(ValidationError, match="设计专属字段"):
        CausalQuestionDraft(
            treatment="treated",
            outcome="outcome",
            estimand="ATT",
            design_type="synthetic_control",
            unit_id_column="unit",
            time_column="period",
            intervention_time=8,
            variable_roles=VariableRoles(confounders=["unit"]),
        )

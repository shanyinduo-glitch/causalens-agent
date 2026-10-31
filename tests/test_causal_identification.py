from causalens.causal.identification import assess_causal_identification


def test_identification_is_not_assessed_without_dag():
    result = assess_causal_identification(
        treatment="treatment",
        outcome="outcome",
        adjustment_set=["confounder"],
        dag_edges=[],
    )

    assert result.status == "not_assessed"
    assert result.identified is None


def test_valid_adjustment_set_blocks_backdoor_path():
    result = assess_causal_identification(
        treatment="treatment",
        outcome="outcome",
        adjustment_set=["confounder"],
        dag_edges=[
            ("confounder", "treatment"),
            ("confounder", "outcome"),
            ("treatment", "outcome"),
        ],
    )

    assert result.status == "identified"
    assert result.identified is True
    assert result.diagnostics["open_backdoor_path"] is False
    assert result.diagnostics["minimal_adjustment_sets"] == [["confounder"]]
    assert result.diagnostics["adjustment_set_minimal"] is True


def test_unblocked_backdoor_path_is_rejected():
    result = assess_causal_identification(
        treatment="treatment",
        outcome="outcome",
        adjustment_set=["measured_but_irrelevant"],
        dag_edges=[
            ("confounder", "treatment"),
            ("confounder", "outcome"),
            ("treatment", "outcome"),
            ("measured_but_irrelevant", "outcome"),
        ],
    )

    assert result.status == "blocked"
    assert result.diagnostics["open_backdoor_path"] is True


def test_treatment_descendant_cannot_be_adjusted_for():
    result = assess_causal_identification(
        treatment="treatment",
        outcome="outcome",
        adjustment_set=["mediator"],
        dag_edges=[
            ("treatment", "mediator"),
            ("mediator", "outcome"),
        ],
    )

    assert result.status == "blocked"
    assert any("后代" in issue for issue in result.blocking_issues)


def test_cycle_is_rejected():
    result = assess_causal_identification(
        treatment="treatment",
        outcome="outcome",
        adjustment_set=["confounder"],
        dag_edges=[
            ("confounder", "treatment"),
            ("treatment", "outcome"),
            ("outcome", "confounder"),
        ],
    )

    assert result.status == "blocked"
    assert result.dag_valid is False
    assert any("有向环" in issue for issue in result.blocking_issues)


def test_collider_path_remains_closed_without_adjustment():
    result = assess_causal_identification(
        treatment="treatment",
        outcome="outcome",
        adjustment_set=[],
        dag_edges=[
            ("treatment", "collider"),
            ("outcome", "collider"),
            ("treatment", "outcome"),
        ],
    )

    assert result.status == "identified"
    assert result.diagnostics["open_backdoor_path"] is False
    assert result.diagnostics["minimal_adjustment_sets"] == [[]]


def test_identification_warns_when_valid_adjustment_set_is_not_minimal():
    result = assess_causal_identification(
        treatment="treatment",
        outcome="outcome",
        adjustment_set=["confounder", "precision_variable"],
        dag_edges=[
            ("confounder", "treatment"),
            ("confounder", "outcome"),
            ("precision_variable", "outcome"),
            ("treatment", "outcome"),
        ],
    )

    assert result.status == "identified"
    assert result.diagnostics["adjustment_set_minimal"] is False
    assert ["confounder"] in result.diagnostics["minimal_adjustment_sets"]
    assert any("not minimal" in warning for warning in result.warnings)

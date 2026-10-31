from __future__ import annotations

from causalens.causal.benchmark import (
    SIMULATION_SCENARIOS,
    SimulationObservation,
    SimulationScenario,
    evaluate_simulation_expectation,
    run_simulation_scenario,
    summarize_simulation_observations,
)


def test_simulation_summary_reports_bias_rmse_coverage_and_failures() -> None:
    observations = [
        SimulationObservation(1, 1, 2.0, True, 2.1, 0.2, [1.7, 2.5]),
        SimulationObservation(2, 2, 2.0, True, 1.9, 0.2, [1.5, 2.3]),
        SimulationObservation(3, 3, 2.0, False, error="blocked"),
    ]

    summary = summarize_simulation_observations(
        observations,
        max_failure_rate=0.5,
        max_absolute_bias=0.2,
    )

    assert summary["successful_replications"] == 2
    assert summary["failure_rate"] == 1 / 3
    assert abs(summary["bias"]) < 1e-12
    assert summary["rmse"] > 0
    assert summary["coverage_95"] == 1.0
    assert summary["failure_reasons"] == {"blocked": 1}
    assert summary["passed"] is True


def test_benchmark_registry_covers_current_primary_design_families() -> None:
    assert set(SIMULATION_SCENARIOS) == {
        "aipw_linear",
        "aipw_nonlinear",
        "did_two_period",
        "event_study_common_timing",
        "staggered_dr",
        "iv_strong_first_stage",
        "rdd_sharp",
        "synthetic_control_single_unit",
        "aipw_hidden_confounding",
        "event_study_pretrend_violation",
        "staggered_weak_overlap",
        "staggered_anticipation",
    }


def test_expectation_evaluation_distinguishes_blocks_diagnostics_and_bias() -> None:
    def unused_generator(_rng):
        raise AssertionError("not called")

    def unused_estimator(_path, _seed):
        raise AssertionError("not called")

    base = dict(
        name="stress",
        description="stress",
        estimand="ATE",
        generator=unused_generator,
        estimator=unused_estimator,
        minimum_sample_size=1,
    )
    blocked = [
        SimulationObservation(1, 1, 2.0, False, error="overlap"),
        SimulationObservation(2, 2, 2.0, False, error="overlap"),
    ]
    blocked_summary = summarize_simulation_observations(
        blocked, max_failure_rate=0.05, max_absolute_bias=0.2
    )
    result = evaluate_simulation_expectation(
        SimulationScenario(
            **base,
            expectation="expected_block",
            expected_failure_substring="overlap",
        ),
        blocked,
        blocked_summary,
    )
    assert result["passed"] is True

    detected = [
        SimulationObservation(
            1, 1, 2.0, True, 3.0, 0.2, [2.6, 3.4], detected_violation=True
        ),
        SimulationObservation(
            2, 2, 2.0, True, 3.0, 0.2, [2.6, 3.4], detected_violation=True
        ),
    ]
    detected_summary = summarize_simulation_observations(
        detected, max_failure_rate=0.05, max_absolute_bias=0.2
    )
    result = evaluate_simulation_expectation(
        SimulationScenario(**base, expectation="diagnostic_detection"),
        detected,
        detected_summary,
    )
    assert result["detection_rate"] == 1.0
    assert result["passed"] is True

    result = evaluate_simulation_expectation(
        SimulationScenario(**base, expectation="expected_bias"),
        detected,
        detected_summary,
    )
    assert result["absolute_bias"] == 1.0
    assert result["passed"] is True


def test_two_period_did_simulation_smoke_test(tmp_path) -> None:
    result = run_simulation_scenario(
        SIMULATION_SCENARIOS["did_two_period"],
        replications=2,
        seed=7,
        temporary_root=tmp_path,
    )

    assert result["summary"]["successful_replications"] == 2
    assert result["summary"]["absolute_bias"] < 0.25


def test_pretrend_violation_simulation_is_detected(tmp_path) -> None:
    result = run_simulation_scenario(
        SIMULATION_SCENARIOS["event_study_pretrend_violation"],
        replications=1,
        seed=11,
        temporary_root=tmp_path,
    )

    assert result["summary"]["detection_rate"] == 1.0
    assert result["summary"]["passed"] is True


def test_weak_overlap_simulation_is_safely_blocked(tmp_path) -> None:
    result = run_simulation_scenario(
        SIMULATION_SCENARIOS["staggered_weak_overlap"],
        replications=1,
        seed=13,
        temporary_root=tmp_path,
    )

    assert result["summary"]["failure_rate"] == 1.0
    assert result["summary"]["passed"] is True


def test_iv_rdd_and_synthetic_control_simulation_smoke_tests(tmp_path) -> None:
    for name in (
        "iv_strong_first_stage",
        "rdd_sharp",
        "synthetic_control_single_unit",
    ):
        result = run_simulation_scenario(
            SIMULATION_SCENARIOS[name],
            replications=1,
            seed=17,
            temporary_root=tmp_path,
        )
        assert result["summary"]["successful_replications"] == 1
        assert result["summary"]["passed"] is True

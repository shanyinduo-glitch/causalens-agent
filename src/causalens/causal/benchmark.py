from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import numpy as np
import pandas as pd

from causalens.tools.causal_tool import run_aipw
from causalens.tools.did_tool import run_difference_in_differences
from causalens.tools.event_study_tool import run_event_study
from causalens.tools.iv_tool import run_two_stage_least_squares
from causalens.tools.rdd_tool import run_sharp_regression_discontinuity
from causalens.tools.staggered_did_tool import run_staggered_did
from causalens.tools.synthetic_control_tool import run_synthetic_control


Generator = Callable[[np.random.Generator], tuple[pd.DataFrame, float]]
Estimator = Callable[[str, int], dict[str, Any]]
Detector = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    description: str
    estimand: str
    generator: Generator
    estimator: Estimator
    minimum_sample_size: int
    max_failure_rate: float = 0.05
    max_absolute_bias: float = 0.35
    expectation: str = "accurate_estimation"
    detector: Detector | None = None
    minimum_detection_rate: float = 0.80
    minimum_expected_failure_rate: float = 0.80
    minimum_expected_absolute_bias: float = 0.50
    expected_failure_substring: str | None = None


@dataclass(frozen=True)
class SimulationObservation:
    replication: int
    seed: int
    truth: float
    success: bool
    estimate: float | None = None
    standard_error: float | None = None
    confidence_interval: list[float] | None = None
    error: str | None = None
    detected_violation: bool | None = None


def summarize_simulation_observations(
    observations: list[SimulationObservation],
    *,
    max_failure_rate: float,
    max_absolute_bias: float,
) -> dict[str, Any]:
    """计算估计偏差、RMSE、区间覆盖率与失败率。"""

    if not observations:
        raise ValueError("observations cannot be empty.")
    successful = [row for row in observations if row.success]
    attempted = len(observations)
    failed = attempted - len(successful)
    failure_rate = failed / attempted
    failure_reasons: dict[str, int] = {}
    for row in observations:
        if row.success:
            continue
        reason = row.error or "Unknown error"
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    summary: dict[str, Any] = {
        "replications": attempted,
        "successful_replications": len(successful),
        "failed_replications": failed,
        "failure_rate": float(failure_rate),
        "max_failure_rate": max_failure_rate,
        "max_absolute_bias": max_absolute_bias,
        "failure_reasons": failure_reasons,
    }
    if not successful:
        summary.update(
            {
                "mean_estimate": None,
                "bias": None,
                "absolute_bias": None,
                "rmse": None,
                "mean_standard_error": None,
                "coverage_95": None,
                "mean_ci_width": None,
                "passed": False,
            }
        )
        return summary

    estimates = np.asarray([row.estimate for row in successful], dtype=float)
    truths = np.asarray([row.truth for row in successful], dtype=float)
    errors = estimates - truths
    standard_errors = np.asarray(
        [row.standard_error for row in successful], dtype=float
    )
    intervals = np.asarray(
        [row.confidence_interval for row in successful], dtype=float
    )
    coverage = (intervals[:, 0] <= truths) & (truths <= intervals[:, 1])
    bias = float(errors.mean())
    absolute_bias = abs(bias)
    summary.update(
        {
            "truth": float(truths.mean()),
            "mean_estimate": float(estimates.mean()),
            "bias": bias,
            "absolute_bias": absolute_bias,
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mean_standard_error": float(standard_errors.mean()),
            "coverage_95": float(coverage.mean()),
            "mean_ci_width": float(np.mean(intervals[:, 1] - intervals[:, 0])),
            "passed": bool(
                failure_rate <= max_failure_rate
                and absolute_bias <= max_absolute_bias
            ),
        }
    )
    return summary


def evaluate_simulation_expectation(
    scenario: SimulationScenario,
    observations: list[SimulationObservation],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """按场景意图判定通过，避免把正确拦截误报为估计器失败。"""

    expectation = scenario.expectation
    result = dict(summary)
    result["expectation"] = expectation
    result["detection_rate"] = None
    result["expected_block_rate"] = None
    if expectation == "accurate_estimation":
        return result
    if expectation == "expected_block":
        expected_blocks = sum(
            row.success is False
            and scenario.expected_failure_substring is not None
            and scenario.expected_failure_substring in (row.error or "")
            for row in observations
        )
        expected_block_rate = expected_blocks / len(observations)
        result["expected_blocked_replications"] = expected_blocks
        result["expected_block_rate"] = float(expected_block_rate)
        result["minimum_expected_failure_rate"] = (
            scenario.minimum_expected_failure_rate
        )
        result["expected_failure_substring"] = scenario.expected_failure_substring
        result["passed"] = bool(
            scenario.expected_failure_substring
            and expected_block_rate >= scenario.minimum_expected_failure_rate
        )
        return result
    if expectation == "expected_bias":
        result["minimum_expected_absolute_bias"] = (
            scenario.minimum_expected_absolute_bias
        )
        result["passed"] = bool(
            result["successful_replications"] > 0
            and result["failure_rate"] <= scenario.max_failure_rate
            and result["absolute_bias"] is not None
            and result["absolute_bias"] >= scenario.minimum_expected_absolute_bias
        )
        return result
    if expectation == "diagnostic_detection":
        detected = sum(row.detected_violation is True for row in observations)
        detection_rate = detected / len(observations)
        result["detected_replications"] = detected
        result["detection_rate"] = float(detection_rate)
        result["minimum_detection_rate"] = scenario.minimum_detection_rate
        result["passed"] = bool(detection_rate >= scenario.minimum_detection_rate)
        return result
    raise ValueError(f"Unsupported simulation expectation: {expectation}")


def _aipw_linear_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    sample_size = 600
    x1 = rng.normal(size=sample_size)
    x2 = rng.normal(size=sample_size)
    propensity = 1 / (1 + np.exp(-0.7 * x1 + 0.4 * x2))
    treatment = rng.binomial(1, propensity)
    truth = 2.0
    outcome = truth * treatment + 1.2 * x1 - 0.8 * x2 + rng.normal(size=sample_size)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome, "x1": x1, "x2": x2}), truth


def _aipw_nonlinear_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    sample_size = 700
    x1 = rng.normal(size=sample_size)
    x2 = rng.normal(size=sample_size)
    propensity = 1 / (1 + np.exp(-(0.5 * x1 - 0.35 * x2)))
    treatment = rng.binomial(1, propensity)
    truth = 2.0
    outcome = (
        truth * treatment
        + 1.5 * np.sin(x1)
        + 0.7 * np.square(x2)
        + rng.normal(size=sample_size)
    )
    return pd.DataFrame({"treatment": treatment, "outcome": outcome, "x1": x1, "x2": x2}), truth


def _did_data(rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    truth = 1.5
    rows: list[dict[str, float | int]] = []
    for group in (0, 1):
        for post in (0, 1):
            for _ in range(200):
                baseline = rng.normal()
                outcome = 4 + 0.8 * group + 0.6 * post + truth * group * post
                outcome += 0.4 * baseline + rng.normal(scale=0.5)
                rows.append(
                    {
                        "group": group,
                        "post": post,
                        "baseline": baseline,
                        "outcome": outcome,
                    }
                )
    return pd.DataFrame(rows), truth


def _event_study_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    effects = {0: 1.0, 1: 2.0, 2: 3.0}
    rows: list[dict[str, float | int]] = []
    for unit in range(120):
        treated = int(unit >= 60)
        unit_effect = rng.normal(scale=0.7)
        for period in (-3, -2, -1, 0, 1, 2):
            rows.append(
                {
                    "unit_id": unit,
                    "treated": treated,
                    "event_time": period,
                    "outcome": (
                        5
                        + unit_effect
                        + 0.4 * period
                        + treated * effects.get(period, 0.0)
                        + rng.normal(scale=0.3)
                    ),
                }
            )
    return pd.DataFrame(rows), 2.0


def _staggered_dr_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    unit_count = 150
    x = rng.normal(size=unit_count)
    logits = np.column_stack([0.65 * x, np.zeros(unit_count), -0.65 * x])
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    categories = [rng.choice(3, p=probabilities[index]) for index in range(unit_count)]
    starts = [2.0 if category == 0 else 4.0 if category == 1 else np.nan for category in categories]
    effects = {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.0, 4: 3.0, 5: 3.0}
    rows: list[dict[str, float | int]] = []
    for unit, (start, risk) in enumerate(zip(starts, x)):
        unit_effect = rng.normal(scale=0.6)
        for period in range(8):
            treated = int(not np.isnan(start) and period >= start)
            event_time = period - int(start) if not np.isnan(start) else None
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "treatment_start": start,
                    "treated": treated,
                    "baseline_risk": risk,
                    "outcome": (
                        5
                        + unit_effect
                        + 0.3 * period
                        + 0.6 * risk * period
                        + (effects.get(event_time, 0.0) if event_time is not None else 0.0)
                        + rng.normal(scale=0.2)
                    ),
                }
            )
    return pd.DataFrame(rows), 2.4


def _hidden_confounding_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    sample_size = 800
    x1 = rng.normal(size=sample_size)
    x2 = rng.normal(size=sample_size)
    unobserved = rng.normal(size=sample_size)
    propensity = 1 / (
        1 + np.exp(-(0.2 * x1 - 0.2 * x2 + 1.25 * unobserved))
    )
    treatment = rng.binomial(1, propensity)
    truth = 2.0
    outcome = (
        truth * treatment
        + 0.5 * x1
        - 0.4 * x2
        + 2.0 * unobserved
        + rng.normal(size=sample_size)
    )
    return pd.DataFrame(
        {"treatment": treatment, "outcome": outcome, "x1": x1, "x2": x2}
    ), truth


def _iv_data(rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    sample_size = 900
    instrument = rng.binomial(1, 0.5, sample_size)
    confounder = rng.normal(size=sample_size)
    latent = rng.normal(size=sample_size)
    treatment = 0.9 * instrument + 0.6 * confounder + 0.8 * latent
    truth = 2.0
    outcome = truth * treatment + 0.7 * confounder + latent + rng.normal(
        scale=0.7, size=sample_size
    )
    return pd.DataFrame(
        {
            "instrument": instrument,
            "treatment": treatment,
            "outcome": outcome,
            "confounder": confounder,
        }
    ), truth


def _rdd_data(rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    sample_size = 900
    running = rng.uniform(-2.0, 2.0, size=sample_size)
    treatment = (running >= 0).astype(int)
    baseline = rng.normal(size=sample_size)
    truth = 1.75
    outcome = (
        truth * treatment
        + 0.8 * running
        + 0.2 * treatment * running
        + 0.3 * baseline
        + rng.normal(scale=0.5, size=sample_size)
    )
    return pd.DataFrame(
        {
            "running": running,
            "treatment": treatment,
            "outcome": outcome,
            "baseline": baseline,
        }
    ), truth


def _synthetic_control_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    truth = 2.5
    donor_count = 8
    times = list(range(12))
    donor_paths = []
    rows: list[dict[str, float | int | str]] = []
    for donor in range(donor_count):
        path = np.asarray(
            [
                4
                + 0.2 * time
                + 0.25 * donor
                + 0.1 * np.sin(time + donor)
                + rng.normal(scale=0.02)
                for time in times
            ]
        )
        donor_paths.append(path)
        for time, outcome in zip(times, path):
            rows.append(
                {
                    "unit": f"donor_{donor}",
                    "time": time,
                    "treated": 0,
                    "outcome": outcome,
                }
            )
    weights = np.asarray([0.5, 0.3, 0.2, 0, 0, 0, 0, 0])
    untreated = np.average(np.vstack(donor_paths), axis=0, weights=weights)
    for time, outcome in zip(times, untreated):
        rows.append(
            {
                "unit": "treated",
                "time": time,
                "treated": int(time >= 8),
                "outcome": outcome + (truth if time >= 8 else 0.0),
            }
        )
    return pd.DataFrame(rows), truth


def _event_study_pretrend_violation_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, float | int]] = []
    for unit in range(140):
        treated = int(unit >= 70)
        unit_effect = rng.normal(scale=0.5)
        for period in (-3, -2, -1, 0, 1, 2):
            treatment_effect = 2.0 if period >= 0 else 0.0
            rows.append(
                {
                    "unit_id": unit,
                    "treated": treated,
                    "event_time": period,
                    "outcome": (
                        5
                        + unit_effect
                        + 0.3 * period
                        + treated * 0.75 * period
                        + treated * treatment_effect
                        + rng.normal(scale=0.2)
                    ),
                }
            )
    return pd.DataFrame(rows), 2.0


def _staggered_weak_overlap_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    del rng
    starts = [2.0] * 40 + [4.0] * 40 + [np.nan] * 40
    risks = [4.0] * 40 + [0.0] * 40 + [-4.0] * 40
    rows: list[dict[str, float | int]] = []
    for unit, (start, risk) in enumerate(zip(starts, risks)):
        for period in range(8):
            treated = int(not np.isnan(start) and period >= start)
            event_time = period - int(start) if not np.isnan(start) else None
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "treatment_start": start,
                    "treated": treated,
                    "baseline_risk": risk,
                    "outcome": (
                        5
                        + 0.25 * period
                        + 0.5 * risk * period
                        + (2.0 if event_time is not None and event_time >= 0 else 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows), 2.0


def _staggered_anticipation_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    starts = [2.0] * 35 + [4.0] * 35 + [np.nan] * 35
    rows: list[dict[str, float | int]] = []
    for unit, start in enumerate(starts):
        unit_effect = rng.normal(scale=0.4)
        for period in range(8):
            treated = int(not np.isnan(start) and period >= start)
            event_time = period - int(start) if not np.isnan(start) else None
            effect = 0.0
            if event_time == -1:
                effect = 1.25
            elif event_time is not None and event_time >= 0:
                effect = 2.0
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "treatment_start": start,
                    "treated": treated,
                    "outcome": 5 + unit_effect + 0.3 * period + effect + rng.normal(scale=0.15),
                }
            )
    return pd.DataFrame(rows), 2.0


def _run_aipw(path: str, seed: int) -> dict[str, Any]:
    return run_aipw(
        dataset_path=path,
        treatment="treatment",
        outcome="outcome",
        confounders=["x1", "x2"],
        random_state=seed,
        cv=3,
    )


def _run_did(path: str, seed: int) -> dict[str, Any]:
    del seed
    return run_difference_in_differences(
        dataset_path=path,
        group_column="group",
        post_column="post",
        outcome="outcome",
        confounders=["baseline"],
    )


def _run_event_study(path: str, seed: int) -> dict[str, Any]:
    del seed
    return run_event_study(
        dataset_path=path,
        group_column="treated",
        event_time_column="event_time",
        outcome="outcome",
        unit_id_column="unit_id",
    )


def _run_staggered(path: str, seed: int) -> dict[str, Any]:
    return run_staggered_did(
        dataset_path=path,
        treatment_column="treated",
        time_column="period",
        treatment_start_column="treatment_start",
        outcome="outcome",
        unit_id_column="unit_id",
        confounders=["baseline_risk"],
        cross_fitting_folds=3,
        bootstrap_iterations=30,
        random_state=seed,
    )


def _run_staggered_unadjusted(path: str, seed: int) -> dict[str, Any]:
    return run_staggered_did(
        dataset_path=path,
        treatment_column="treated",
        time_column="period",
        treatment_start_column="treatment_start",
        outcome="outcome",
        unit_id_column="unit_id",
        bootstrap_iterations=30,
        random_state=seed,
    )


def _run_iv(path: str, seed: int) -> dict[str, Any]:
    del seed
    return run_two_stage_least_squares(
        dataset_path=path,
        treatment="treatment",
        outcome="outcome",
        instruments=["instrument"],
        confounders=["confounder"],
    )


def _run_rdd(path: str, seed: int) -> dict[str, Any]:
    del seed
    return run_sharp_regression_discontinuity(
        dataset_path=path,
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff=0.0,
        covariates=["baseline"],
    )


def _run_synthetic_control(path: str, seed: int) -> dict[str, Any]:
    del seed
    return run_synthetic_control(
        dataset_path=path,
        treatment="treated",
        outcome="outcome",
        unit_id_column="unit",
        time_column="time",
        intervention_time=8,
    )


def _pretrend_rejected(result: dict[str, Any]) -> bool:
    return (result.get("pretrend_test") or {}).get("passed") is False


def _placebo_rejected(result: dict[str, Any]) -> bool:
    return (result.get("placebo_test") or {}).get("passed") is False


SIMULATION_SCENARIOS: dict[str, SimulationScenario] = {
    "aipw_linear": SimulationScenario(
        name="aipw_linear",
        description="Binary treatment, measured linear confounding, good overlap.",
        estimand="ATE",
        generator=_aipw_linear_data,
        estimator=_run_aipw,
        minimum_sample_size=600,
        max_absolute_bias=0.25,
    ),
    "aipw_nonlinear": SimulationScenario(
        name="aipw_nonlinear",
        description="Binary treatment with nonlinear outcome nuisance functions.",
        estimand="ATE",
        generator=_aipw_nonlinear_data,
        estimator=_run_aipw,
        minimum_sample_size=700,
        max_absolute_bias=0.30,
    ),
    "did_two_period": SimulationScenario(
        name="did_two_period",
        description="Repeated cross-section two-period parallel-trends DiD.",
        estimand="ATT",
        generator=_did_data,
        estimator=_run_did,
        minimum_sample_size=800,
        max_absolute_bias=0.20,
    ),
    "event_study_common_timing": SimulationScenario(
        name="event_study_common_timing",
        description="Balanced panel Event Study with common treatment timing.",
        estimand="ATT",
        generator=_event_study_data,
        estimator=_run_event_study,
        minimum_sample_size=720,
        max_absolute_bias=0.25,
    ),
    "staggered_dr": SimulationScenario(
        name="staggered_dr",
        description="Staggered cohorts with measured baseline trend confounding.",
        estimand="ATT",
        generator=_staggered_dr_data,
        estimator=_run_staggered,
        minimum_sample_size=1200,
        max_failure_rate=0.15,
        max_absolute_bias=0.35,
    ),
    "iv_strong_first_stage": SimulationScenario(
        name="iv_strong_first_stage",
        description="Valid strong instrument with one endogenous treatment.",
        estimand="LATE",
        generator=_iv_data,
        estimator=_run_iv,
        minimum_sample_size=900,
        max_absolute_bias=0.25,
    ),
    "rdd_sharp": SimulationScenario(
        name="rdd_sharp",
        description="Sharp cutoff assignment with a smooth conditional outcome.",
        estimand="LATE",
        generator=_rdd_data,
        estimator=_run_rdd,
        minimum_sample_size=900,
        max_absolute_bias=0.35,
    ),
    "synthetic_control_single_unit": SimulationScenario(
        name="synthetic_control_single_unit",
        description="One treated unit generated by a convex donor combination.",
        estimand="ATT",
        generator=_synthetic_control_data,
        estimator=_run_synthetic_control,
        minimum_sample_size=108,
        max_absolute_bias=0.20,
    ),
    "aipw_hidden_confounding": SimulationScenario(
        name="aipw_hidden_confounding",
        description="AIPW stress test with a strong omitted common cause.",
        estimand="ATE",
        generator=_hidden_confounding_data,
        estimator=_run_aipw,
        minimum_sample_size=800,
        expectation="expected_bias",
        minimum_expected_absolute_bias=0.50,
    ),
    "event_study_pretrend_violation": SimulationScenario(
        name="event_study_pretrend_violation",
        description="Event Study stress test with treatment-group differential trends.",
        estimand="ATT",
        generator=_event_study_pretrend_violation_data,
        estimator=_run_event_study,
        minimum_sample_size=840,
        expectation="diagnostic_detection",
        detector=_pretrend_rejected,
        minimum_detection_rate=0.80,
    ),
    "staggered_weak_overlap": SimulationScenario(
        name="staggered_weak_overlap",
        description="Near-separated cohorts expected to trigger overlap/ESS blocking.",
        estimand="ATT",
        generator=_staggered_weak_overlap_data,
        estimator=_run_staggered,
        minimum_sample_size=960,
        expectation="expected_block",
        minimum_expected_failure_rate=0.80,
        expected_failure_substring="overlap/ESS diagnostics did not pass",
    ),
    "staggered_anticipation": SimulationScenario(
        name="staggered_anticipation",
        description="One-period anticipation expected to fail placebo timing.",
        estimand="ATT",
        generator=_staggered_anticipation_data,
        estimator=_run_staggered_unadjusted,
        minimum_sample_size=840,
        expectation="diagnostic_detection",
        detector=_placebo_rejected,
        minimum_detection_rate=0.80,
    ),
}


def run_simulation_scenario(
    scenario: SimulationScenario,
    *,
    replications: int,
    seed: int,
    temporary_root: str | Path | None = None,
) -> dict[str, Any]:
    """运行单个 DGP；每次复制使用独立 CSV，且不调用任何 LLM。"""

    if replications < 1:
        raise ValueError("replications must be positive.")
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(replications)
    observations: list[SimulationObservation] = []
    with TemporaryDirectory(
        prefix="causalens-benchmark-",
        dir=temporary_root,
    ) as temporary_directory:
        directory = Path(temporary_directory)
        for replication, child_sequence in enumerate(child_sequences, start=1):
            replication_seed = int(child_sequence.generate_state(1)[0])
            rng = np.random.default_rng(child_sequence)
            dataframe, truth = scenario.generator(rng)
            if len(dataframe) < scenario.minimum_sample_size:
                raise ValueError(
                    f"Scenario {scenario.name!r} generated {len(dataframe)} rows; "
                    f"expected at least {scenario.minimum_sample_size}."
                )
            dataset_path = directory / f"{scenario.name}_{replication}.csv"
            dataframe.to_csv(dataset_path, index=False)
            response = scenario.estimator(str(dataset_path), replication_seed)
            if response.get("success") is not True:
                observations.append(
                    SimulationObservation(
                        replication=replication,
                        seed=replication_seed,
                        truth=truth,
                        success=False,
                        error=str(response.get("error") or "Unknown error"),
                    )
                )
                continue
            result = response["result"]
            detected_violation = (
                scenario.detector(result) if scenario.detector is not None else None
            )
            observations.append(
                SimulationObservation(
                    replication=replication,
                    seed=replication_seed,
                    truth=truth,
                    success=True,
                    estimate=float(result["ate"]),
                    standard_error=float(result["standard_error"]),
                    confidence_interval=[
                        float(result["confidence_interval"][0]),
                        float(result["confidence_interval"][1]),
                    ],
                    detected_violation=detected_violation,
                )
            )
    summary = evaluate_simulation_expectation(
        scenario,
        observations,
        summarize_simulation_observations(
            observations,
            max_failure_rate=scenario.max_failure_rate,
            max_absolute_bias=scenario.max_absolute_bias,
        ),
    )
    return {
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "estimand": scenario.estimand,
            "minimum_sample_size": scenario.minimum_sample_size,
            "expectation": scenario.expectation,
        },
        "summary": summary,
        "observations": [asdict(row) for row in observations],
    }


def run_simulation_benchmark(
    *,
    scenario_names: list[str] | None = None,
    replications: int = 10,
    seed: int = 20261014,
    temporary_root: str | Path | None = None,
) -> dict[str, Any]:
    selected_names = scenario_names or list(SIMULATION_SCENARIOS)
    unknown = sorted(set(selected_names) - set(SIMULATION_SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown simulation scenarios: {unknown}")
    results = [
        run_simulation_scenario(
            SIMULATION_SCENARIOS[name],
            replications=replications,
            seed=seed + index * 100_003,
            temporary_root=temporary_root,
        )
        for index, name in enumerate(selected_names)
    ]
    return {
        "benchmark": "causal_estimator_simulation",
        "replications_per_scenario": replications,
        "seed": seed,
        "scenario_count": len(results),
        "passed_scenarios": sum(
            result["summary"]["passed"] is True for result in results
        ),
        "all_passed": all(result["summary"]["passed"] for result in results),
        "results": results,
    }

import numpy as np
import pandas as pd

from causalens.causal.diagnostics import run_causal_diagnostics


def test_synthetic_dataset_has_structured_diagnostics():
    response = run_causal_diagnostics(
        dataset_path="data/synthetic_causal_data.csv",
        treatment="ad_exposure",
        outcome="retention_score",
        confounders=[
            "age",
            "log_income",
            "historical_activity",
            "channel_score",
        ],
    )

    assert response["success"] is True
    result = response["result"]
    assert result["sample_size"] == 3000
    assert result["propensity_summary"]["cross_fitting_folds"] == 5
    assert len(result["balance"]) == 4
    assert result["weight_summary"]["effective_sample_size"] > 0


def test_no_common_support_blocks_diagnostics(tmp_path):
    rows = 200
    treatment = np.repeat([0, 1], rows // 2)
    separated_feature = np.where(treatment == 1, 20.0, -20.0)
    dataframe = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": treatment + np.linspace(0, 1, rows),
            "separated_feature": separated_feature,
        }
    )
    dataset_path = tmp_path / "no_overlap.csv"
    dataframe.to_csv(dataset_path, index=False)

    response = run_causal_diagnostics(
        dataset_path=str(dataset_path),
        treatment="treatment",
        outcome="outcome",
        confounders=["separated_feature"],
    )

    assert response["success"] is True
    assert response["result"]["passed"] is False
    assert response["result"]["propensity_summary"][
        "has_common_support"
    ] is False


def test_diagnostics_rejects_nonbinary_treatment(tmp_path):
    dataframe = pd.DataFrame(
        {
            "treatment": [0, 1, 2, 0, 1, 2],
            "outcome": [1, 2, 3, 4, 5, 6],
            "confounder": [1, 2, 3, 4, 5, 6],
        }
    )
    dataset_path = tmp_path / "nonbinary.csv"
    dataframe.to_csv(dataset_path, index=False)

    response = run_causal_diagnostics(
        dataset_path=str(dataset_path),
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder"],
    )

    assert response["success"] is False
    assert "0/1" in response["error"]

from __future__ import annotations

import numpy as np
import pandas as pd

from causalens.causal.data_quality import assess_causal_data_quality


def test_data_quality_reports_policy_and_identifier_like_adjustment(tmp_path):
    size = 40
    frame = pd.DataFrame(
        {
            "treatment": [0, 1] * (size // 2),
            "outcome": np.linspace(0, 1, size),
            "row_id": np.arange(size),
        }
    )
    path = tmp_path / "quality.csv"
    frame.to_csv(path, index=False)

    response = assess_causal_data_quality(
        dataset_path=str(path),
        required_columns=["treatment", "outcome", "row_id"],
        numeric_columns=["treatment", "outcome", "row_id"],
        adjustment_columns=["row_id"],
    )

    assert response["success"] is True
    result = response["result"]
    assert result["ready"] is True
    assert result["preprocessing_policy"]["automatic_imputation"] is False
    assert result["column_diagnostics"]["adjustment_columns"]["row_id"][
        "possible_identifier"
    ] is True


def test_data_quality_does_not_mislabel_continuous_return_as_identifier(tmp_path):
    size = 40
    frame = pd.DataFrame(
        {
            "treatment": [0, 1] * (size // 2),
            "outcome": np.linspace(0, 1, size),
            "market_return": np.linspace(-0.03, 0.04, size),
        }
    )
    path = tmp_path / "continuous_adjustment.csv"
    frame.to_csv(path, index=False)

    response = assess_causal_data_quality(
        dataset_path=str(path),
        required_columns=["treatment", "outcome", "market_return"],
        numeric_columns=["treatment", "outcome", "market_return"],
        adjustment_columns=["market_return"],
    )

    diagnostic = response["result"]["column_diagnostics"]["adjustment_columns"][
        "market_return"
    ]
    assert diagnostic["possible_identifier"] is False
    assert not any("market_return" in warning for warning in response["result"]["warnings"])


def test_data_quality_blocks_low_complete_case_retention(tmp_path):
    frame = pd.DataFrame(
        {
            "treatment": [0, 1] * 20,
            "outcome": [np.nan] * 25 + list(np.arange(15, dtype=float)),
            "x": np.arange(40, dtype=float),
        }
    )
    path = tmp_path / "missing.csv"
    frame.to_csv(path, index=False)

    response = assess_causal_data_quality(
        dataset_path=str(path),
        required_columns=["treatment", "outcome", "x"],
        numeric_columns=["treatment", "outcome", "x"],
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert any(
        "Complete-case retention" in issue
        for issue in response["result"]["blocking_issues"]
    )


def test_data_quality_blocks_duplicate_panel_key(tmp_path):
    rows = [
        {"unit": unit, "time": time, "treatment": unit % 2, "outcome": unit + time}
        for unit in range(10)
        for time in range(3)
    ]
    frame = pd.DataFrame([*rows, rows[0]])
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)

    response = assess_causal_data_quality(
        dataset_path=str(path),
        required_columns=["unit", "time", "treatment", "outcome"],
        numeric_columns=["time", "treatment", "outcome"],
        panel_key=("unit", "time"),
    )

    assert response["result"]["ready"] is False
    assert response["result"]["duplicate_diagnostics"][
        "duplicate_panel_keys"
    ] == 1

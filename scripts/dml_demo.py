from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from econml.inference import StatsModelsInference
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_causal_data.csv"
)

TREATMENT = "ad_exposure"
OUTCOME = "retention_score"

CONFOUNDERS = [
    "age",
    "log_income",
    "historical_activity",
    "channel_score",
]

KNOWN_TRUE_ATE = 3.0
RANDOM_STATE = 42
CV_FOLDS = 5


def as_float(value: object) -> float:
    """把 NumPy 标量或单元素数组转换成 float。"""
    array = np.asarray(value)

    if array.size != 1:
        raise ValueError(
            "Expected a scalar result, "
            f"but received shape {array.shape}."
        )

    return float(array.reshape(-1)[0])


def main() -> None:
    dataframe = pd.read_csv(DATASET_PATH)

    required_columns = [
        TREATMENT,
        OUTCOME,
        *CONFOUNDERS,
    ]

    selected_data = (
        dataframe[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    treatment_values = (
        selected_data[TREATMENT]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if treatment_values != [0, 1]:
        raise ValueError(
            "This demo requires treatment coded as 0 and 1. "
            f"Observed values: {treatment_values}"
        )

    y = (
        selected_data[OUTCOME]
        .to_numpy(dtype=float)
    )

    treatment = (
        selected_data[TREATMENT]
        .to_numpy(dtype=int)
    )

    controls = (
        selected_data[CONFOUNDERS]
        .astype(float)
    )

    # 结果变量 nuisance model：
    # 允许 E[Y | W] 存在非线性关系。
    model_y = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 二元 treatment nuisance model：
    # StandardScaler 和 LogisticRegression 被放进同一个
    # Pipeline，避免在交叉拟合前对完整数据提前拟合 scaler。
    model_t = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    estimator = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
    )

    estimator.fit(
        y,
        treatment,
        X=None,
        W=controls,
        inference=StatsModelsInference(
            cov_type="HC1"
        ),
    )

    estimated_ate = as_float(
        estimator.ate(
            X=None,
            T0=0,
            T1=1,
        )
    )

    lower_bound, upper_bound = (
        estimator.ate_interval(
            X=None,
            T0=0,
            T1=1,
            alpha=0.05,
        )
    )

    lower_bound = as_float(lower_bound)
    upper_bound = as_float(upper_bound)

    treated_mean = float(
        selected_data.loc[
            selected_data[TREATMENT] == 1,
            OUTCOME,
        ].mean()
    )

    control_mean = float(
        selected_data.loc[
            selected_data[TREATMENT] == 0,
            OUTCOME,
        ].mean()
    )

    naive_difference = (
        treated_mean - control_mean
    )

    print("=" * 80)
    print("LinearDML Demo")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Sample size: {len(selected_data)}")
    print(f"Treatment: {TREATMENT}")
    print(f"Outcome: {OUTCOME}")
    print(f"Confounders: {CONFOUNDERS}")
    print()
    print(
        "Treatment counts:",
        selected_data[TREATMENT]
        .value_counts()
        .sort_index()
        .to_dict(),
    )
    print()
    print(
        f"Naive treated-control difference: "
        f"{naive_difference:.4f}"
    )
    print(
        f"Estimated ATE: "
        f"{estimated_ate:.4f}"
    )
    print(
        f"95% confidence interval: "
        f"[{lower_bound:.4f}, "
        f"{upper_bound:.4f}]"
    )
    print(
        f"Known true ATE: "
        f"{KNOWN_TRUE_ATE:.4f}"
    )
    print(
        f"Absolute estimation error: "
        f"{abs(estimated_ate - KNOWN_TRUE_ATE):.4f}"
    )
    print(
        "True ATE inside interval:",
        lower_bound
        <= KNOWN_TRUE_ATE
        <= upper_bound,
    )


if __name__ == "__main__":
    main()
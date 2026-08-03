from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic_causal_data.csv"


def generate_causal_data(
    n: int = 3000,
    true_ate: float = 3.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    生成一份带有已知真实平均处理效应的模拟因果数据。

    Treatment:
        ad_exposure

    Outcome:
        retention_score

    Observed confounders:
        age
        income
        historical_activity
        channel_score

    True ATE:
        true_ate
    """
    if n < 100:
        raise ValueError("n must be at least 100.")

    rng = np.random.default_rng(random_state)

    age = rng.normal(loc=35, scale=8, size=n)
    income = rng.lognormal(mean=10.5, sigma=0.4, size=n)
    historical_activity = rng.normal(loc=50, scale=12, size=n)
    channel_score = rng.normal(loc=0, scale=1, size=n)

    # 对偏态的 income 取对数并标准化，避免原始金额尺度过大。
    log_income = np.log(income)
    income_z = (
        log_income - log_income.mean()
    ) / log_income.std(ddof=0)

    # treatment 同时受到多个背景变量影响，因此不是随机分配。
    treatment_logit = (
        -1.0
        + 0.02 * (age - 35)
        + 0.03 * (historical_activity - 50)
        + 0.50 * channel_score
        + 0.25 * income_z
    )

    treatment_probability = 1 / (1 + np.exp(-treatment_logit))
    ad_exposure = rng.binomial(
        n=1,
        p=treatment_probability,
        size=n,
    )

    noise = rng.normal(loc=0, scale=4, size=n)

    retention_score = (
        20
        + true_ate * ad_exposure
        + 0.08 * age
        + 0.12 * historical_activity
        + 1.50 * channel_score
        + 0.80 * income_z
        + noise
    )

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "age": age,
            "income": income,
            "log_income": log_income,
            "historical_activity": historical_activity,
            "channel_score": channel_score,
            "ad_exposure": ad_exposure,
            "retention_score": retention_score,
        }
    )


def validate_generated_data(
    dataframe: pd.DataFrame,
    expected_rows: int,
) -> None:
    """对生成结果执行最基础的确定性检查。"""
    required_columns = {
        "user_id",
        "age",
        "income",
        "historical_activity",
        "channel_score",
        "ad_exposure",
        "retention_score",
        "log_income",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Generated data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"but got {len(dataframe)}."
        )

    treatment_values = set(
        dataframe["ad_exposure"].dropna().unique().tolist()
    )

    if treatment_values != {0, 1}:
        raise ValueError(
            "ad_exposure must contain both treatment "
            "and control observations."
        )

    if dataframe["retention_score"].nunique() < 2:
        raise ValueError(
            "retention_score must have variation."
        )


def save_causal_data(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    n: int = 3000,
    true_ate: float = 3.0,
    random_state: int = 42,
) -> pd.DataFrame:
    dataframe = generate_causal_data(
        n=n,
        true_ate=true_ate,
        random_state=random_state,
    )

    validate_generated_data(
        dataframe=dataframe,
        expected_rows=n,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return dataframe


if __name__ == "__main__":
    number_of_rows = 3000
    known_true_ate = 3.0

    df = save_causal_data(
        n=number_of_rows,
        true_ate=known_true_ate,
        random_state=42,
    )

    treatment_counts = (
        df["ad_exposure"]
        .value_counts()
        .sort_index()
    )

    naive_group_difference = (
        df.loc[
            df["ad_exposure"] == 1,
            "retention_score",
        ].mean()
        - df.loc[
            df["ad_exposure"] == 0,
            "retention_score",
        ].mean()
    )

    print("Synthetic causal data generated successfully.")
    print(f"Output path: {DEFAULT_OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print()
    print("Treatment counts:")
    print(treatment_counts)
    print()
    print(
        f"Treatment rate: "
        f"{df['ad_exposure'].mean():.4f}"
    )
    print(f"Known true ATE: {known_true_ate:.4f}")
    print(
        f"Naive treated-control difference: "
        f"{naive_group_difference:.4f}"
    )
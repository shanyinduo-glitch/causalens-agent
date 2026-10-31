from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from causalens.schemas import CausalDataQualityResult


def assess_causal_data_quality(
    *,
    dataset_path: str,
    required_columns: list[str],
    numeric_columns: list[str],
    adjustment_columns: list[str] | None = None,
    panel_key: tuple[str, str] | None = None,
    minimum_complete_rows: int = 20,
    minimum_complete_fraction: float = 0.50,
    maximum_core_missing_fraction: float = 0.50,
) -> dict[str, Any]:
    """只读检查进入估计器的数据；不静默插补、编码或删除异常值。"""

    normalized_required = list(dict.fromkeys(required_columns))
    normalized_numeric = list(dict.fromkeys(numeric_columns))
    normalized_adjustments = list(dict.fromkeys(adjustment_columns or []))
    blocking: list[str] = []
    warnings: list[str] = []
    try:
        frame = pd.read_csv(dataset_path)
    except FileNotFoundError:
        return {
            "success": False,
            "result": None,
            "error": f"Dataset file does not exist: {dataset_path}",
        }
    except Exception as exc:
        return {
            "success": False,
            "result": None,
            "error": f"Failed to read dataset for data-quality checks: {exc}",
        }
    if frame.empty:
        blocking.append("Dataset has no rows.")
    missing_columns = sorted(set(normalized_required) - set(frame.columns))
    if missing_columns:
        blocking.append(f"Required columns are missing: {missing_columns}")

    available = [column for column in normalized_required if column in frame.columns]
    selected = frame[available].replace([np.inf, -np.inf], np.nan)
    missing_fraction = {
        column: float(selected[column].isna().mean())
        for column in available
    }
    for column, fraction in missing_fraction.items():
        if fraction > maximum_core_missing_fraction:
            blocking.append(
                f"Column {column!r} is missing in {fraction:.1%} of rows, above "
                f"the {maximum_core_missing_fraction:.0%} safety threshold."
            )
        elif fraction > 0.10:
            warnings.append(
                f"Column {column!r} is missing in {fraction:.1%} of rows."
            )

    non_numeric = [
        column
        for column in normalized_numeric
        if column in frame.columns
        and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        blocking.append(
            "The selected estimator requires numeric analysis columns: "
            f"{non_numeric}. Explicit encoding is required."
        )
    constant_columns = [
        column
        for column in available
        if selected[column].dropna().nunique() <= 1
    ]
    if constant_columns:
        blocking.append(f"Required columns have no variation: {constant_columns}")

    complete_rows = int(selected.dropna().shape[0]) if available else 0
    total_rows = int(len(frame))
    complete_fraction = float(complete_rows / total_rows) if total_rows else 0.0
    if available and complete_rows < minimum_complete_rows:
        blocking.append(
            f"Only {complete_rows} complete rows remain; at least "
            f"{minimum_complete_rows} are required by the general quality gate."
        )
    if available and complete_fraction < minimum_complete_fraction:
        blocking.append(
            f"Complete-case retention is {complete_fraction:.1%}, below the "
            f"{minimum_complete_fraction:.0%} safety threshold."
        )

    exact_duplicates = int(frame.duplicated().sum())
    if exact_duplicates:
        warnings.append(
            f"Dataset contains {exact_duplicates} exact duplicate rows; verify "
            "whether they represent repeated observations."
        )
    duplicate_panel_keys = 0
    if panel_key and all(column in frame.columns for column in panel_key):
        duplicate_panel_keys = int(frame.duplicated(list(panel_key)).sum())
        if duplicate_panel_keys:
            blocking.append(
                f"Panel key {list(panel_key)} has {duplicate_panel_keys} duplicate "
                "unit-time rows."
            )

    adjustment_diagnostics: dict[str, Any] = {}
    for column in normalized_adjustments:
        if column not in frame.columns:
            continue
        unique_count = int(frame[column].nunique(dropna=True))
        unique_fraction = float(unique_count / max(frame[column].notna().sum(), 1))
        normalized_name = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        name_tokens = set(normalized_name.split("_"))
        identifier_name = bool(
            name_tokens.intersection({"id", "identifier", "uuid", "key"})
            or normalized_name.endswith("_code")
        )
        non_numeric_identifier = not pd.api.types.is_numeric_dtype(frame[column])
        id_like = bool(
            unique_count > 20
            and unique_fraction > 0.90
            and (identifier_name or non_numeric_identifier)
        )
        near_constant = bool(
            not frame[column].dropna().empty
            and frame[column].value_counts(normalize=True, dropna=True).iloc[0] > 0.99
        )
        adjustment_diagnostics[column] = {
            "unique_count": unique_count,
            "unique_fraction": unique_fraction,
            "possible_identifier": id_like,
            "identifier_name_hint": identifier_name,
            "near_constant": near_constant,
        }
        if id_like:
            warnings.append(
                f"Adjustment column {column!r} looks identifier-like; confirm it is "
                "a pre-treatment cause rather than an ID."
            )
        if near_constant:
            warnings.append(
                f"Adjustment column {column!r} is nearly constant and may add "
                "numerical instability without useful adjustment."
            )

    result = CausalDataQualityResult(
        ready=not blocking,
        required_columns=normalized_required,
        numeric_columns=normalized_numeric,
        blocking_issues=list(dict.fromkeys(blocking)),
        warnings=list(dict.fromkeys(warnings)),
        sample_summary={
            "total_rows": total_rows,
            "complete_rows": complete_rows,
            "dropped_if_complete_case": total_rows - complete_rows,
            "complete_case_fraction": complete_fraction,
        },
        column_diagnostics={
            "missing_fraction": missing_fraction,
            "non_numeric_columns": non_numeric,
            "constant_columns": constant_columns,
            "adjustment_columns": adjustment_diagnostics,
        },
        duplicate_diagnostics={
            "exact_duplicate_rows": exact_duplicates,
            "panel_key": list(panel_key) if panel_key else None,
            "duplicate_panel_keys": duplicate_panel_keys,
        },
        preprocessing_policy={
            "mutates_source_data": False,
            "automatic_imputation": False,
            "automatic_categorical_encoding": False,
            "automatic_outlier_removal": False,
            "infinite_values_treated_as_missing": True,
            "estimation_missing_policy": "explicit complete-case analysis",
            "reason": (
                "Potentially consequential preprocessing requires an explicit "
                "research decision and is never silently inferred."
            ),
        },
    )
    return {"success": True, "result": result.model_dump(), "error": None}

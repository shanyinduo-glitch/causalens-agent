from __future__ import annotations

import pandas as pd
import pytest

from causalens.causal.runtime import (
    AnalysisResultCache,
    WorkflowCheckpointStore,
    build_analysis_cache_key,
)


def test_analysis_cache_key_changes_when_dataset_changes(tmp_path):
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(dataset, index=False)
    first = build_analysis_cache_key(
        dataset_path=dataset,
        request={"estimand": "ATE"},
    )
    pd.DataFrame({"x": [1, 3]}).to_csv(dataset, index=False)
    second = build_analysis_cache_key(
        dataset_path=dataset,
        request={"estimand": "ATE"},
    )

    assert first != second


def test_analysis_result_cache_round_trip(tmp_path):
    cache = AnalysisResultCache(tmp_path / "cache")
    key = "a" * 64
    value = {"success": True, "result": {"ate": 2.0}}

    path = cache.store(key, value)

    assert path.is_file()
    assert cache.load(key) == value


def test_workflow_checkpoint_round_trip_and_safe_run_id(tmp_path):
    store = WorkflowCheckpointStore(tmp_path / "checkpoints")
    state = {"workflow_status": "awaiting_approval", "approved": None}

    store.save("analysis-001", state)

    assert store.load("analysis-001") == state
    with pytest.raises(ValueError, match="run_id"):
        store.save("../escape", state)

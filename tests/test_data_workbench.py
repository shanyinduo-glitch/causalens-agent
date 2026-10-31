import numpy as np
import pandas as pd
import pytest
from causalens.ui.data_workbench import transform_thesis, LOGS, DIFFS, LEVELS, lookup_knowledge, bound_metadata
from causalens.ui.result_comparison import compare


def frame():
    data = pd.DataFrame({c: np.arange(1., 31.) for c in [*LOGS, *DIFFS, *LEVELS]})
    data['date'] = pd.date_range('2020-01-01', periods=30)
    return data


def test_transform_reproducible_and_preserves_source():
    data = frame()
    original = data.copy(deep=True)
    result = transform_thesis(data)
    assert len(result) == 28
    assert result.time_trend.iloc[0] == 3
    assert result.t_diff.iloc[0] == 1
    assert result.y_ret.iloc[0] == pytest.approx(np.log(3) - np.log(2))
    assert result.y_ret_lag1.iloc[0] == pytest.approx(np.log(2))
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize('bad', ['zero', 'missing', 'duplicate', 'reverse'])
def test_invalid_transform_rejected(bad):
    data = frame()
    if bad == 'zero': data.loc[0, 'y'] = 0
    if bad == 'missing': data.loc[0, 't'] = np.nan
    if bad == 'duplicate': data.loc[1, 'date'] = data.loc[0, 'date']
    if bad == 'reverse': data = data.iloc[::-1]
    with pytest.raises(ValueError): transform_thesis(data)


def test_unknown_data_never_inherits_dictionary():
    assert bound_metadata(b'unrelated,data\n1,2') is None
    assert lookup_knowledge(None, 'y_ret') == []
    record = {'field': 'y_ret', 'text': '对数差分', 'source': 'script'}
    assert lookup_knowledge({'knowledge': [record]}, 'y_ret') == [record]
    assert lookup_knowledge({'knowledge': [record]}, '不存在') == []


def test_comparison_rejects_other_data_and_response():
    estimate = {'analysis_type': 'time_series_local_projection', 'model_configuration': {'response': 'cumulative'}, 'dynamic_effects': []}
    record = {'format': 'causalens-comparison-v1', 'dataset_sha256': 'a', 'suite': {'estimates': [estimate]}}
    assert compare(record, record) == []
    with pytest.raises(ValueError): compare(record, {**record, 'dataset_sha256': 'b'})


def test_case_workbench_ui():
    from pathlib import Path
    from causalens.ui.data_workbench import ROOT
    from streamlit.testing.v1 import AppTest
    if not (ROOT / 'data/private/gold_tips_thesis/aligned/gold_tips_daily_levels.csv').exists():
        pytest.skip('private case data not installed')
    app = AppTest.from_file(str(Path(__file__).parent / 'fixtures/data_workbench_app.py'), default_timeout=30).run()
    assert not app.exception
    assert app.button[0].disabled
    app.checkbox[0].check().run()
    assert not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception
    assert len(app.get('download_button')) == 2
    app.text_input[0].set_value('y').run()
    assert not app.exception
    assert len(app.get('download_button')) == 2

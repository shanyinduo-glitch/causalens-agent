import numpy as np
import pandas as pd
import pytest

from causalens.ui.research_advisor import advise_design, inspect_stationarity


def frame():
    rng = np.random.default_rng(42)
    return pd.DataFrame({'date': pd.date_range('2020-01-01', periods=100), 't': rng.normal(size=100), 'y': rng.normal(size=100)})


@pytest.mark.parametrize('structure,mechanism,expected', [
    ('不确定', '连续变化的冲击', None),
    ('同一对象随时间变化', '不确定', None),
    ('同一对象随时间变化', '连续变化的冲击', 'time_series'),
    ('多个对象重复观测', '连续变化的冲击', None),
    ('不同对象各一行', '只能依据处理前共同原因调整', None),
    ('多个对象重复观测', '政策实施前后有处理组与对照组', 'difference_in_differences'),
    ('同一对象随时间变化', '政策实施前后有处理组与对照组', None),
    ('多个对象重复观测', '一个受影响对象与多个未受影响对象', 'synthetic_control'),
    ('不同对象各一行', '按明确分数阈值分配', 'regression_discontinuity'),
    ('不同对象各一行', '有外生工具变量依据', 'instrumental_variable'),
])
def test_design_requires_declared_background(structure, mechanism, expected):
    assert advise_design(frame(), treatment='t', outcome='y', structure=structure, mechanism=mechanism)['design'] == expected


def test_binary_cross_section_and_invalid_roles():
    data = frame()
    data['t'] = [0, 1] * 50
    kwargs = dict(structure='不同对象各一行', mechanism='随机分组')
    assert advise_design(data, treatment='t', outcome='y', **kwargs)['design'] == 'cross_sectional'
    assert advise_design(data, treatment='t', outcome='t', **kwargs)['design'] is None


def test_screen_does_not_modify_data():
    data = frame().iloc[::-1].copy()
    original = data.copy(deep=True)
    result = inspect_stationarity(data, treatment='t', outcome='y', time_column='date')
    assert result['y']['available']
    pd.testing.assert_frame_equal(original, data)


def test_ui_candidate_requires_confirmation():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parent / 'fixtures/research_advisor_app.py')).run()
    assert not app.exception
    app.selectbox[0].select('同一对象随时间变化')
    app.selectbox[1].select('连续变化的冲击')
    app.run()
    assert not app.exception
    assert app.session_state['test_result'] == (None, None, None)
    app.selectbox(key='advisor_time').select('date')
    next(w for w in app.selectbox if w.label == '希望研究哪种关系？').select('短期变化或动态响应')
    app.run()
    app.button[0].click().run()
    assert not app.exception
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state['test_result'] == ('time_series', 'IRF', 'date')
    assert any('ADF p值' in w.value for w in app.markdown)
    next(w for w in app.selectbox if w.label == '希望研究哪种关系？').select('长期水平关系').run()
    assert app.session_state['test_result'] == (None, None, None)


def test_pending_design_is_not_ready():
    from causalens.ui.causal_guidance import assess_guided_compatibility
    assert not assess_guided_compatibility(frame(), treatment='t', outcome='y', design_type=None)['ready']


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'infinite', 'invalid_date'])
def test_screen_rejects_invalid_input(problem):
    data = frame()
    if problem == 'duplicate':
        data.loc[1, 'date'] = data.loc[0, 'date']
    elif problem == 'invalid_date':
        data.loc[0, 'date'] = pd.NaT
    else:
        data.loc[0, 't'] = np.nan if problem == 'missing' else np.inf
    with pytest.raises(ValueError):
        inspect_stationarity(data, treatment='t', outcome='y', time_column='date')

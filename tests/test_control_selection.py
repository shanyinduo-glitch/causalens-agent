from pathlib import Path
import pytest
from causalens.ui.control_selection import BASE, EXTRA, case_eligible
from causalens.ui.data_workbench import ROOT


def test_unknown_data_cannot_use_case_recipe():
    assert not case_eligible(b'unknown', 't_diff', 'y_ret', 'time_series')


def test_case_roles_and_ui_confirmation():
    path = ROOT / 'data/private/gold_tips_thesis/processed/gold_tips_daily_stationary.csv'
    if not path.exists():
        pytest.skip('Private case not installed')
    assert not case_eligible(path.read_bytes(), 'y_ret', 't_diff', 'time_series')
    assert not case_eligible(path.read_bytes(), 't_diff', 'y_ret', 'cross_sectional')
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parent / 'fixtures/control_selection_app.py')).run()
    assert not app.exception
    assert app.session_state['control_result'] == ([], False)
    app.radio[0].set_value('基础案例方案（4项）').run()
    assert app.session_state['control_result'] == (BASE, False)
    app.checkbox[0].check().run()
    assert app.session_state['control_result'] == (BASE, True)
    app.radio[0].set_value('扩展案例方案（9项）').run()
    assert app.session_state['control_result'] == (BASE + EXTRA, False)
    assert app.warning
    app.radio[0].set_value('自定义').run()
    assert app.session_state['control_result'] == ([], False)
    assert 'date' not in app.multiselect[0].options
    assert not app.exception

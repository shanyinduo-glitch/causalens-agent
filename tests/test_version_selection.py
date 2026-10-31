from pathlib import Path
from causalens.ui.data_workbench import clear_active_analysis


def test_switch_clears_analysis_but_preserves_preparation():
    state = {'active_data_fingerprint': 'a', 'final_state': {'old': True}, 'current_contract': {}, 'prepared_a': b'keep'}
    clear_active_analysis(state, 'a')
    assert 'final_state' in state
    clear_active_analysis(state, 'b')
    assert 'final_state' not in state and 'current_contract' not in state
    assert state['prepared_a'] == b'keep'


def test_version_switch_round_trip():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parent / 'fixtures/version_selection_app.py')).run()
    assert not app.exception
    original = app.session_state['selected_bytes']
    app.session_state['final_state'] = {'old_result': True}
    app.radio[0].set_value('转换版本（短期变化）').run()
    assert not app.exception
    assert b't_diff' in app.session_state['selected_bytes']
    assert app.session_state['selected_name'] == 'gold_tips_prepared.csv'
    assert 'final_state' not in app.session_state
    app.radio[0].set_value('原始上传版本').run()
    assert not app.exception
    assert app.session_state['selected_bytes'] == original
    assert app.session_state['selected_name'] == 'source.csv'

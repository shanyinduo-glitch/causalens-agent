from pathlib import Path
import pytest
from causalens.ui.demo_mode import load_demo, DATA


def test_missing_demo_data_is_explicit(tmp_path):
    with pytest.raises(ValueError, match='缺失'):
        load_demo(tmp_path / 'missing.csv')


def test_demo_real_execution_and_stale_guard():
    if not DATA.exists():
        pytest.skip('Private demo data unavailable')
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parent / 'fixtures/demo_app.py'), default_timeout=60).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.button[1].disabled
    next(w for w in app.checkbox if w.label.startswith('我已了解')).check().run()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state['demo_result']['state']['workflow_status'] == 'completed'
    assert len(app.get('download_button')) >= 1
    app.number_input(key='demo_tlags').set_value(2).run()
    assert not app.exception
    assert any('旧结果暂停展示' in w.value for w in app.warning)
    assert len(app.get('download_button')) == 0
    assert app.button[1].disabled

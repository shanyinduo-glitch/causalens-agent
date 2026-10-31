import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from causalens.ui.causal_guidance import suggest_causal_roles_with_model


def test_plan_evidence_and_invalid_method(monkeypatch):
    import causalens.agent.qwen_client as client
    captured = []
    def fake(messages, **kwargs):
        captured.extend(messages)
        return SimpleNamespace(model='test-model', usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({'design_candidate': 'invented',
                'treatment_candidates': ['t', 'missing'], 'rationale': '需要时间背景'})))])
    monkeypatch.setattr(client, 'create_chat_completion', fake)
    value = suggest_causal_roles_with_model('影响？', pd.DataFrame({'t': [987654, 123456]}),
                                           {'knowledge': [{'field': 't', 'text': '利率'}], 'risks': ['风险']})
    assert value['design_candidate'] == 'unknown'
    assert value['treatment'] == ['t']
    assert value['model'] == 'test-model'
    assert value['knowledge_bound'] and not value['confirmed']
    assert '987654' not in captured[1]['content']
    assert '利率' in captured[1]['content']


def test_plan_requires_consent_and_hides_stale(monkeypatch):
    import causalens.ui.ai_planning as panel
    from streamlit.testing.v1 import AppTest
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return dict(research_question='问题', design_candidate='time_series', rationale='待核实',
                    treatment=['t'], outcome=['y'], time=[], adjustment=[], alternatives=[],
                    clarifying_questions=['频率？'], warning='', model='test-model', usage=None,
                    knowledge_bound=False, confirmed=False)
    monkeypatch.setattr(panel, 'suggest_causal_roles_with_model', fake)
    app = AppTest.from_file(str(Path(__file__).parent / 'fixtures/ai_planning_app.py')).run()
    assert not app.exception and app.button[0].disabled and not calls
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception and len(calls) == 1
    assert len(app.get('download_button')) == 1
    app.text_input[0].set_value('新的问题').run()
    assert not app.exception and app.button[0].disabled
    assert len(app.get('download_button')) == 0 and len(calls) == 1


def test_clarification_revision_and_unknown_rag_guard(monkeypatch):
    import causalens.ui.ai_planning as panel
    from streamlit.testing.v1 import AppTest
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(research_question='问题', design_candidate='unknown', rationale='待核实',
                    treatment=['t'], outcome=['y'], time=[], adjustment=[], alternatives=[],
                    clarifying_questions=['频率？'], warning='', model='test', usage=None,
                    knowledge_bound=False, confirmed=False, evidence_ids=[],
                    raw_draft={'design_candidate': 'unknown'},
                    validation={'status': 'needs_revision', 'issues': ['方法未定']})
    monkeypatch.setattr(panel, 'suggest_causal_roles_with_model', fake)
    monkeypatch.setattr(panel, 'bound_metadata', lambda raw: None)
    app = AppTest.from_file(str(Path(__file__).parent / 'fixtures/ai_planning_app.py')).run()
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception and len(calls) == 1 and calls[0][0][2] is None
    app.text_area[0].set_value('每日观测，未来5期').run()
    assert app.button[0].disabled and len(calls) == 1
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception and len(calls) == 2
    assert calls[1][1]['clarification'] == '每日观测，未来5期'
    assert calls[1][1]['previous_plan'] == {'design_candidate': 'unknown'}
    assert len(app.session_state['guided_ai_plan']['revisions']) == 1
    app.radio[0].set_value('论文背景增强（RAG）').run()
    app.checkbox[0].check().run()
    assert app.button[0].disabled and len(calls) == 2
    assert len(app.get('download_button')) == 0

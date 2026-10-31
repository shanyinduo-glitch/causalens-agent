import json
from types import SimpleNamespace
import pandas as pd
from causalens.ui.method_knowledge import method_cards, comparison_rows, validate_method_options
from causalens.ui.causal_guidance import suggest_causal_roles_with_model


def option(key):
    return dict(method_id=key, reason='背景支持作为候选', missing_evidence='外生性仍待核实',
                target_alignment='短期动态；其他目标需另行确认')


def test_cards_have_sources_and_separate_theory_from_code():
    cards = {c['id']: c for c in method_cards()}
    assert len(cards) == 8
    assert all(c['url'].startswith('https://') and c['assumptions'] and c['diagnostics'] for c in cards.values())
    assert '已有执行器' in cards['M-LP']['implementation']
    assert '尚未接入' in cards['M-DML-C']['implementation']
    assert '尚未接入' in cards['M-ECM']['implementation']


def test_untrusted_execution_claim_not_used():
    payload = option('M-DML-C') | {'implementation': '可以立即执行', 'executable': True}
    assert '尚未接入' in comparison_rows([payload])[0]['系统实现状态（代码核对）']
    assert comparison_rows([None, {'method_id': []}, {'method_id': 'invented'}]) == []


def test_options_validation():
    assert not validate_method_options({'design_candidate': 'time_series', 'method_options': [option('M-LP'), option('M-DML-C')]})
    for options in (None, [], [option('fake')], [option('M-LP')]*2, [{'method_id': 'M-LP'}]):
        assert validate_method_options({'method_options': options})
    assert validate_method_options({'design_candidate': 'time_series', 'method_options': [option('M-DML-C')]})


def test_boolean_alignment_from_copied_draft_remains_invalid_and_is_explained():
    # Reproduce the types in the user's copied viewer output, not raw API JSON.
    options = [option('M-LP') | {'target_alignment': True},
               option('M-DML-C') | {'target_alignment': False}]
    issues = validate_method_options({'design_candidate': 'time_series', 'method_options': options})
    assert len(issues) == 2
    assert 'M-LP' in issues[0] and 'M-DML-C' in issues[1]
    assert all('target_alignment' in issue and '布尔值' in issue for issue in issues)
    rows = comparison_rows(options)
    assert all('待修订' in row['与本次目标的关系'] for row in rows)
    assert options[0]['target_alignment'] is True
    assert options[1]['target_alignment'] is False


def test_model_receives_cards_and_returns_checked_comparison(monkeypatch):
    import causalens.agent.qwen_client as client
    captured = []
    payload = dict(design_candidate='time_series', treatment_candidates=['t'], outcome_candidates=['y'],
                   time_candidates=['date'], confounder_candidates=[], evidence_ids=['M-LP'],
                   method_options=[option('M-LP'), option('M-DML-C')])
    def fake(messages, **kwargs):
        captured.extend(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
    monkeypatch.setattr(client, 'create_chat_completion', fake)
    result = suggest_causal_roles_with_model('未来5期动态', pd.DataFrame({'t':[.1,.2], 'y':[1,2], 'date':[1,2]}))
    assert result['validation']['status'] == 'candidate_only'
    notes = json.loads(captured[1]['content'])['retrieved_evidence']
    assert len([c for c in notes if c['id'] == 'M-LP']) == 1
    assert result['method_options'][1]['method_id'] == 'M-DML-C'
    assert not result['confirmed']
    assert '不能返回布尔值' in captured[0]['content']

    payload['method_options'][0]['target_alignment'] = True
    rejected = suggest_causal_roles_with_model('未来5期动态', pd.DataFrame({'t':[.1,.2], 'y':[1,2], 'date':[1,2]}))
    assert rejected['validation']['status'] == 'needs_revision'
    assert rejected['raw_draft']['method_options'][0]['target_alignment'] is True

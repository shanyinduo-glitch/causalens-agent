import pandas as pd
from causalens.ui.planning_evidence import capabilities, retrieve_case_evidence, validate_plan


def test_catalog_is_backed_by_registry():
    catalog = capabilities()
    assert 'time_series' in catalog['design_labels']
    assert any('time_series' in s['data_structures'] for s in catalog['registered_estimators'])


def test_retrieval_is_scoped_and_traceable():
    assert retrieve_case_evidence(None, '黄金 TIPS') == []
    hits = retrieve_case_evidence({'knowledge': [], 'risks': []}, '黄金 TIPS 机会成本')
    assert hits and all(h['id'].startswith('E-') and h['source'] for h in hits)
    assert hits == retrieve_case_evidence({'knowledge': [], 'risks': []}, '黄金 TIPS 机会成本')
    assert retrieve_case_evidence({'knowledge': []}, 'zzzznotfound') == []


def test_bad_plan_not_silently_repaired():
    frame = pd.DataFrame({'t': [.1, .2], 'y': [3, 4], 'date': [1, 2]})
    invalid = {'design_candidate': 'unknown', 'treatment_candidates': ['t'],
               'outcome_candidates': ['t'], 'confounder_candidates': ['t'], 'evidence_ids': ['fake']}
    result = validate_plan(invalid, frame, [])
    assert result['status'] == 'needs_revision' and len(result['issues']) >= 4
    assert invalid['design_candidate'] == 'unknown'
    valid = {'design_candidate': 'time_series', 'treatment_candidates': ['t'],
             'outcome_candidates': ['y'], 'time_candidates': ['date'], 'evidence_ids': ['E-1']}
    assert validate_plan(valid, frame, [{'id': 'E-1'}])['status'] == 'candidate_only'
    valid['design_candidate'] = 'cross_sectional'
    assert validate_plan(valid, frame, [{'id': 'E-1'}])['status'] == 'needs_revision'


def test_bad_shapes_are_reviewed():
    result = validate_plan({'design_candidate': [], 'evidence_ids': None,
                            'treatment_candidates': {}, 'outcome_candidates': 'x'}, pd.DataFrame({'x':[1]}), [])
    assert result['status'] == 'needs_revision'


def test_known_capability_contradiction_is_flagged():
    result = validate_plan({'design_candidate': 'time_series', 'treatment_candidates': ['t'],
                            'outcome_candidates': ['y'], 'time_candidates': ['date'],
                            'alternatives': ['time_series: 不在允许的设计选项中']},
                           pd.DataFrame({'t': [.1], 'y': [.2], 'date': [1]}), [])
    assert any('能力清单矛盾' in x for x in result['issues'])

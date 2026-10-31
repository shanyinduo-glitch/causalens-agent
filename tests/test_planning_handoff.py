from pathlib import Path
from causalens.ui.planning_handoff import text_items, explicit_role_issues, missing_context, attach_planning
from causalens.reporting.planning_trace import planning_section


def test_lists_and_explicit_roles():
    assert text_items("['第一项', '第二项']") == ['第一项', '第二项']
    assert text_items(['第一项', None]) == ['第一项']
    assert text_items('普通说明') == ['普通说明']
    assert explicit_role_issues('结果变量明确为 y_ret', {'outcome_candidates': ['comex_gold_ret', 'y_ret']})
    assert not explicit_role_issues('结果变量明确为 y_ret', {'outcome_candidates': ['y_ret']})
    assert not explicit_role_issues('黄金的影响', {})


def test_missing_context_is_chinese():
    assert missing_context({'target_population': '样本', 'unit_of_analysis': ' ', 'outcome_horizon': ''}) == ['每一行代表什么', '观察多长时间后的结果']


def test_snapshot_is_frozen_and_trace_explains_unexecuted_dml():
    snapshot = {'identity': 'version1', 'question': '研究问题', 'route': '论文背景增强',
                'value': {'model': 'test', 'method_options': [{'method_id': 'M-LP'}, {'method_id':'M-DML-C'}],
                          'evidence_ids':['E-1'], 'design_candidate':'time_series'},
                'evidence':[{'id':'E-1', 'source':'论文摘述', 'text':'变量定义'}],
                'final_contract': {'design_type':'time_series', 'treatment':'t_diff', 'outcome':'y_ret'}}
    state = attach_planning({}, snapshot)
    snapshot['value']['model'] = 'changed'
    text = planning_section(state)
    assert '模型：test' in text and 'changed' not in text
    assert '连续处理DML仅参与方法比较，本次未执行' in text
    assert '变量定义' in text and 't_diff' in text
    assert planning_section({}) == ''


def test_report_records_revision_and_control_difference():
    state = {'planning_trace': {
        'value': {'adjustment': ['dollar_ret', 'infl_exp5y_level'],
                  'validation': {'status': 'candidate_only'}},
        'final_contract': {'confounders': ['dollar_ret']},
        'revisions': [{'time': ['date', 'time_trend'],
                       'validation': {'status': 'needs_revision', 'issues': ['时间候选不一致']}}]}}
    report = planning_section(state)
    assert 'AI 建议但最终未选：infl_exp5y_level' in report
    assert '时间候选不一致' in report
    assert 'candidate_only' in report


def test_web_report_includes_submitted_trace():
    # Isolate the real report function without starting the interactive app.
    import ast
    source = Path('streamlit_app.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'build_markdown_report')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'streamlit_app.py', 'exec'), namespace)
    from unittest.mock import patch
    with patch('causalens.reporting.chinese.time_series_report', return_value='统计正文'):
        report = namespace['build_markdown_report']({'planning_trace': {'value': {'model': 'test-model'}}})
    assert '统计正文' in report and '模型：test-model' in report


def test_context_only_filled_after_confirmation():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parent / 'fixtures/context_handoff_app.py')).run()
    assert not app.exception
    assert all(w.value == '' for w in app.text_input)
    app.button[0].click().run()
    assert not app.exception
    assert [w.value for w in app.text_input] == ['样本期', '日度观测', '未来5个观测期']
    app.text_input[0].set_value('修改后的范围').run()
    assert app.text_input[0].value == '修改后的范围'

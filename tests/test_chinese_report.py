from copy import deepcopy
from datetime import datetime

from causalens.reporting.markdown_report import generate_markdown_report
from causalens.reporting.chinese import time_series_report


def test_chinese_time_series_report_preserves_evidence():
    state = {
        'workflow_status': 'completed',
        'question': '研究连续冲击',
        'causal_question': {'treatment': 't_diff'},
        'tool_results': [{'success': True, 'result': {
            'design_type': 'time_series', 'treatment': 't_diff', 'outcome': 'y_ret',
            'estimates': [{
                'analysis_type': 'time_series_local_projection',
                'model_configuration': {'response': 'cumulative outcome from t through t+h'},
                'dynamic_effects': [{'horizon': 0, 'estimate': -0.022475671,
                                    'confidence_interval': [-0.0385, -0.00644]}],
                'residual_diagnostics': {'residual_autocorrelation_warning': True},
            }],
        }}],
    }
    original = deepcopy(state)
    timestamp = datetime(2026, 10, 21)
    report = generate_markdown_report(state, generated_at=timestamp)
    assert report == time_series_report(state, generated_at=timestamp)
    assert '# CausaLens 因果分析报告' in report
    assert '工作流状态：已完成' in report
    assert '累计响应' in report
    assert '| 0 | -0.0224757 |' in report
    assert '残差仍存在自相关' in report
    assert '研究配置与检查记录' in report
    assert '-0.022475671' in report
    assert state == original


def test_non_time_series_falls_back():
    assert time_series_report({'tool_results': []}) is None

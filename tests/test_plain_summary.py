from causalens.reporting.plain_summary import summarize_time_series


def state(value=-0.02, ci=None):
    return {'tool_results': [{'success': True, 'result': {
        'design_type': 'time_series', 'estimates': [{
            'analysis_type': 'time_series_local_projection',
            'model_configuration': {'response': 'cumulative outcome from t through t+h'},
            'dynamic_effects': [{'horizon': 0, 'estimate': value, 'confidence_interval': ci}],
            'residual_diagnostics': {'residual_autocorrelation_warning': True},
        }],
    }}]}


def test_summary_explains_limits_and_never_infers_units():
    text = summarize_time_series(state(ci=[-0.03, -0.01]))
    assert '均为负值' in text
    assert '不包含零' in text
    assert '不自动换算' in text
    assert '不能把不同期数的结果再次相加' in text
    assert '残差存在时间规律' in text


def test_uncertainty_is_not_hidden():
    assert '包含零，方向仍不明确' in summarize_time_series(state(ci=[-1, 1]))
    assert '未提供可用' in summarize_time_series(state())
    assert '缺少可用' in summarize_time_series(state(value=float('nan')))
    assert summarize_time_series({}) is None


def test_bound_units_scale_interval_and_use_plain_period(monkeypatch):
    import causalens.reporting.plain_summary as module
    monkeypatch.setattr(module, 'metadata_for_state', lambda s: {'verified_transform': True, 'kind': 'stationary', 'risks': ['CPI时点待核实']})
    data = state(ci=[-0.03, -0.01])
    data['tool_results'][0]['result'].update(treatment='t_diff', outcome='y_ret')
    text = summarize_time_series(data)
    assert '**当期**' in text and '第0期' not in text
    assert '-0.200%' in text and '-0.300% 至 -0.100%' in text
    assert '1个原始单位' not in text
    assert '不代表所有相关字段都进入了本次模型' in text

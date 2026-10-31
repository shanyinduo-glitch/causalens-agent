from causalens.reporting.customer_report import customer_report


def test_customer_report_keeps_summary_not_audit(monkeypatch):
    monkeypatch.setattr('causalens.reporting.customer_report.summarize_time_series',
                        lambda state: '结果仅供探索。原始系数见技术详情')
    result = customer_report({'question': '研究目标', 'planning_trace': {'secret_marker': 'audit-only'}})
    assert '研究目标' in result and '仅供探索' in result
    assert '单独的技术记录' in result
    assert 'audit-only' not in result


def test_missing_summary_does_not_invent_result():
    assert '未生成可用的客户摘要' in customer_report({})

from causalens.reporting.tool_fallback import tool_fallback
from causalens.tools.chart_tool import display_label


def test_fallback_uses_facts_without_raw_payload():
    result = tool_fallback([{'success': True, 'tool_name': 'profile_dataset', 'result': {
        'rows': 4538, 'columns': 15, 'missing_values': {'t_diff': 0},
        'numeric_summary': {'t_diff': {'mean': 0.1, 'std': 0.2}}}}])
    assert '4538' in result and '缺失数量：0' in result and '均值 0.1' in result
    assert '不是模型解读' in result and 'numeric_summary' not in result


def test_chart_labels_are_human_readable():
    assert display_label('t_diff') == 'TIPS 实际收益率变化'
    assert display_label('unknown_column') == 'unknown_column'

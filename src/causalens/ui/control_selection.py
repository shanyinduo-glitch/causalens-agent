"""Explicit case-specific adjustment choices; not automatic confounder discovery."""
from causalens.ui.data_workbench import bound_metadata, digest

BASE = ['dollar_ret', 'sp500_ret', 'fed_funds_diff', 'vix_level']
EXTRA = ['comex_gold_ret', 'global_holdings_tonnes_ret', 'cpi_yoy_level', 'infl_exp5y_level', 'gprh_level']
DESCRIPTIONS = {
    'dollar_ret': ('美元与货币', '美元指数变化', '描述美元相关共同波动；仍需判断是否处于传导路径。'),
    'fed_funds_diff': ('美元与货币', '联邦基金利率变化', '描述货币政策环境；不等同于外生政策冲击。'),
    'sp500_ret': ('市场风险', '标普500收益率', '描述风险资产环境；可能与研究变量共同响应信息。'),
    'vix_level': ('市场风险', 'VIX指数', '描述市场波动情绪；需确认测量时间。'),
    'comex_gold_ret': ('实物与持仓', 'COMEX库存变化', '是库存而非金价；可能有反馈或中介关系。'),
    'global_holdings_tonnes_ret': ('实物与持仓', 'ETF持仓变化', '月度填充到日度，发布日期尚未核实。'),
    'cpi_yoy_level': ('通胀', 'CPI同比', '上游月份对齐存在疑问，当前未修复。'),
    'infl_exp5y_level': ('通胀', '5年盈亏平衡通胀率', '通胀预期代理；可能与实际利率共同变化。'),
    'gprh_level': ('地缘政治', 'GPR指数', '月度填充，真实可获得时间尚未核实。'),
    'time_trend': ('动态设置', '观测序号趋势', '确定性趋势不是已证明的混杂变量。'),
    'y_ret_lag1': ('动态设置', '黄金收益率预制滞后', '若控制变量再滞后1期，会引入更早一期的信息。'),
    't_diff_lag1': ('动态设置', 'TIPS变化预制滞后', '模型已有处理滞后；另加本字段会改变动态设定。'),
}


def case_eligible(raw, treatment, outcome, design):
    meta = bound_metadata(raw)
    return bool(meta and meta.get('verified_transform') and meta.get('kind') == 'stationary'
                and treatment == 't_diff' and outcome == 'y_ret' and design == 'time_series')


def render_control_selection(raw, columns, reserved, treatment, outcome, design):
    import streamlit as st
    eligible = case_eligible(raw, treatment, outcome, design)
    scope = digest(raw)[:16] + '_' + digest(repr((treatment, outcome, design, sorted(reserved))).encode())[:12]
    available = [c for c in columns if c not in reserved]
    st.markdown('##### 选择要比较的控制方案')
    mode = st.radio('控制方案', ['待选择', '基础案例方案（4项）', '扩展案例方案（9项）', '自定义'] if eligible else ['待选择', '自定义'], key='controls_mode_' + scope)
    if eligible:
        st.caption('案例方案来自已有研究配置，不是系统已证明的共同原因集合。两种方案均不自动加入趋势和预制滞后项。')
    else:
        st.info('当前数据或变量方向不符合已核验案例，不套用黄金/TIPS控制方案。请依据数据说明选择处理前共同原因。')
    if mode == '待选择':
        st.info('下一步：选择一个方案查看说明。暂不生成可执行契约。')
        return [], False
    if mode.startswith('基础'):
        selected = list(BASE)
    elif mode.startswith('扩展'):
        selected = BASE + EXTRA
    else:
        selected = st.multiselect('自定义控制字段（不默认全选）', available,
                                  format_func=lambda c: f'{DESCRIPTIONS[c][1]}（{c}）' if eligible and c in DESCRIPTIONS else c,
                                  key='controls_custom_' + scope)
    if any(c not in available for c in selected):
        st.error('方案字段与当前变量角色冲突，不能采用。')
        return [], False
    if eligible and selected:
        st.dataframe([{'类别': DESCRIPTIONS.get(c, ('待确认', '', ''))[0], '字段': c,
                       '含义': DESCRIPTIONS.get(c, ('', c, ''))[1],
                       '选择理由与限制': DESCRIPTIONS.get(c, ('', '', '需要数据说明'))[2]} for c in selected], hide_index=True)
    if eligible and any(c in selected for c in ['cpi_yoy_level', 'global_holdings_tonnes_ret', 'gprh_level']):
        st.warning('本方案包含尚未解决的信息时点风险字段，只适合带明确限制的探索与配置比较，不能当作已修复数据。')
    if any(c.endswith('_lag1') for c in selected):
        st.warning('预制滞后字段还会受到模型“控制变量滞后”设置影响，请检查实际阶数，避免无意改变方案。')
    st.caption('时间序列模型使用控制字段的滞后值，具体阶数见高级参数。是否真是处理前信息仍需确认；增加控制不能保证消除混杂。')
    selection_id = digest(repr((mode, selected)).encode())[:12]
    label = '我已查看选择理由与风险，确认采用这组控制变量' if selected else '我确认不使用控制变量，并了解未调整混杂的限制'
    approved = st.checkbox(label, key='controls_approved_' + scope + '_' + selection_id)
    return selected, approved

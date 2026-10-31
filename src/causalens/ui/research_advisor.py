"""Offline research-design advice. Suggestions never constitute identification approval."""
from __future__ import annotations

import numpy as np
import pandas as pd

from causalens.tools.time_series_tool import _stationarity


STRUCTURES = ['不确定', '不同对象各一行', '同一对象随时间变化', '多个对象重复观测']
MECHANISMS = ['不确定', '随机分组', '政策实施前后有处理组与对照组', '按明确分数阈值分配',
              '有外生工具变量依据', '一个受影响对象与多个未受影响对象', '连续变化的冲击', '只能依据处理前共同原因调整']


def advise_design(frame, *, treatment, outcome, structure, mechanism):
    """Map user-declared context to provisional designs, without guessing from correlations."""
    result = {'design': None, 'reason': '', 'questions': [], 'warnings': [], 'facts': {}}
    missing = [c for c in (treatment, outcome) if not c or c not in frame.columns]
    if missing or treatment == outcome:
        result['questions'] = ['请先选择两个不同且真实存在的处理、结果字段，并确认其含义与单位。']
        return result
    values = frame[treatment].dropna()
    binary = set(values.unique()) == {0, 1}
    result['facts'] = {'样本行数': len(frame), '处理变量不同取值数': int(values.nunique()),
                       '处理变量是否为0/1': binary, '处理或结果缺失行数': int(frame[[treatment, outcome]].isna().any(axis=1).sum())}
    if structure not in STRUCTURES or structure == '不确定':
        result['questions'].append('每行代表一个对象还是一个时间点？同一对象是否重复出现？')
        return result
    candidates = {
        '政策实施前后有处理组与对照组': ('difference_in_differences', '政策前后与组间对比可考虑双重差分。', '请确认处理前趋势、同期政策、对照组溢出，并填写处理时点字段。'),
        '按明确分数阈值分配': ('regression_discontinuity', '明确资格阈值可考虑断点回归。', '请提供运行变量和阈值；检查阈值操纵与连续性，当前实现为清晰断点。'),
        '有外生工具变量依据': ('instrumental_variable', '有外部识别依据时可考虑工具变量。', '请提供工具变量，说明相关性、排除限制和外生性依据；不能用相关系数替代。'),
        '一个受影响对象与多个未受影响对象': ('synthetic_control', '单个处理对象与供体面板可考虑合成控制。', '请确认干预时点、足够长的处理前记录、供体未受影响与处理前拟合。'),
        '连续变化的冲击': ('time_series', '单条时间序列的连续冲击可考虑局部投影，研究动态响应。', '请解释冲击为何近似外生，确认响应时间窗、变量单位和控制变量的测量时点。'),
        '随机分组': ('cross_sectional', '0/1随机分组可使用当前横截面估计器；随机化依据需用户确认。', '请确认随机化、失访、依从性和干扰；当前不自动处理聚类随机实验。'),
        '只能依据处理前共同原因调整': ('cross_sectional', '可考虑基于可观测混杂调整的估计。', '请说明共同原因是否充分、均在处理前测量；不可观测混杂无法由表格排除。'),
    }
    if mechanism not in candidates:
        result['questions'].append('是什么造成处理差异？请提供实验、政策、阈值、外生冲击或处理前共同原因的背景。')
        return result
    design, reason, question = candidates[mechanism]
    result.update(reason=reason, questions=[question])
    if design == 'time_series' and (structure != '同一对象随时间变化' or binary):
        result['warnings'].append('当前局部投影入口面向单条连续冲击序列；面板或二元处理不应直接套用。')
    elif design == 'synthetic_control' and structure != '多个对象重复观测':
        result['warnings'].append('合成控制需要多个对象的处理前后面板。')
    elif design == 'difference_in_differences' and structure == '同一对象随时间变化':
        result['warnings'].append('单条序列未提供处理组与对照组，暂不能推荐双重差分。')
    elif design == 'cross_sectional' and (not binary or structure != '不同对象各一行'):
        result['warnings'].append('当前横截面估计器要求0/1处理与横截面结构；不能把连续值自动二分后使用DML。')
    elif not pd.api.types.is_numeric_dtype(frame[outcome]):
        result['warnings'].append('当前结果变量不是数值字段，请先确认编码。')
    else:
        result['design'] = design
    return result


def inspect_stationarity(frame, *, treatment, outcome, time_column):
    """Read-only ADF screen; no dropping gaps, transformation or causal estimation."""
    required = [time_column, treatment, outcome]
    if len(set(required)) != 3 or any(c not in frame for c in required):
        raise ValueError('请选择三个不同且存在的时间、处理和结果字段。')
    time = frame[time_column]
    ordered = pd.to_numeric(time, errors='coerce') if pd.api.types.is_numeric_dtype(time) else pd.to_datetime(time, errors='coerce')
    if ordered.isna().any() or ordered.duplicated().any():
        raise ValueError('时间存在缺失、无效值或重复。面板应先明确对象，不能拼接成单条序列。')
    results = {}
    for column in [treatment, outcome]:
        values = frame.loc[ordered.sort_values().index, column]
        if not pd.api.types.is_numeric_dtype(values) or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'{column} 包含非数值、缺失或无穷值；请先确认处理方案，不自动填补。')
        results[column] = _stationarity(values)
    return results


def render_research_advisor(frame, treatment, outcome, design_labels):
    import streamlit as st
    import hashlib

    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes() + str(list(frame.columns)).encode()).hexdigest()
    scope = (fingerprint, treatment, outcome)
    if st.session_state.get('advisor_scope') != scope:
        for key in ('advisor_accepted', 'advisor_screen', 'advisor_screen_error', 'advisor_time', 'advisor_manual_design'):
            st.session_state.pop(key, None)
        st.session_state['advisor_scope'] = scope

    with st.expander('不知道选什么方法？先描述研究场景', expanded=True):
        st.caption('本地规则辅助，不消耗模型额度。建议不是因果识别结论，也不替你选择控制变量。')
        structure = st.selectbox('数据中的每一行属于哪种情况？', STRUCTURES)
        mechanism = st.selectbox('你了解的处理差异来源是什么？', MECHANISMS)
        advice = advise_design(frame, treatment=treatment, outcome=outcome, structure=structure, mechanism=mechanism)
        if advice['facts']:
            st.caption(f'已读取 {advice["facts"]["样本行数"]} 行数据。字段类型只能帮助检查适配性，不能确定因果机制。')
        for text in advice['questions']:
            st.info(text)
        for text in advice['warnings']:
            st.warning(text)
        time_column = None
        goal = None
        if structure == '同一对象随时间变化':
            time_column = st.selectbox('时间字段（预检与执行共用）', [None, *frame.columns], key='advisor_time', format_func=lambda x: '请选择时间字段' if x is None else x)
            goal = st.selectbox('希望研究哪种关系？', ['待确认', '短期变化或动态响应', '长期水平关系'])
            if goal == '长期水平关系':
                st.warning('当前局部投影入口不提供协整或误差修正模型。请先制定长期关系识别方案；不能以差分分析替代原问题。')
        context = (scope, structure, mechanism, time_column, goal)
        if st.session_state.get('advisor_context') != context:
            st.session_state.pop('advisor_manual_design', None)
            st.session_state.pop('advisor_accepted', None)
            st.session_state.pop('advisor_screen', None)
            st.session_state.pop('advisor_screen_error', None)
            st.session_state['advisor_context'] = context
        if advice['design']:
            label = next(label for label, key in design_labels.items() if key == advice['design'])
            st.write(f'候选设计：{label}。{advice["reason"]}')
            st.caption('未声明的政策、阈值、工具变量等依据不会从数值相关性中推断；执行前仍需完整配置与检查。')
        else:
            st.caption('暂不推荐模型。补充背景后再判断；也可由熟悉研究设计的用户手动配置。')
        if structure == '同一对象随时间变化':
            if st.button('只检查水平数据的平稳性，不执行估计', disabled=not all([treatment, outcome, time_column])):
                try:
                    diagnostics = inspect_stationarity(frame, treatment=treatment, outcome=outcome, time_column=time_column)
                except ValueError as exc:
                    st.session_state['advisor_screen_error'] = str(exc)
                    st.session_state.pop('advisor_screen', None)
                else:
                    st.session_state['advisor_screen'] = diagnostics
                    st.session_state.pop('advisor_screen_error', None)
            if st.session_state.get('advisor_screen_error'):
                st.warning(st.session_state['advisor_screen_error'])
            if st.session_state.get('advisor_screen'):
                    for column, test in st.session_state['advisor_screen'].items():
                        if not test.get('available'):
                            st.warning(f'{column}：样本量或变化不足，无法给出ADF诊断。')
                        else:
                            st.write(f'{column}：ADF p值 = {test["p_value"]:.6g}（常数项，AIC选滞后，10%阈值）')
                            if not test['reject_unit_root_at_10_percent']:
                                st.warning('这列数据可能存在持续趋势，当前短期响应方案暂不能直接使用。下一步先确认是否研究变化量，或改为制定长期关系方案；不要仅为通过检验而转换。技术含义：不能拒绝单位根假设。')
                    st.info('ADF不是平稳性的证明，也不证明因果识别。差分或对数收益率会改变研究问题；本次未转换数据。检验按排序后的观测间隔进行，需另行确认频率对齐与缺失处理。')
        screen = st.session_state.get('advisor_screen')
        risk = bool(st.session_state.get('advisor_screen_error')) or bool(screen and any(not t.get('available') or not t.get('reject_unit_root_at_10_percent') for t in screen.values()))
        blocked = structure == '同一对象随时间变化' and (goal != '短期变化或动态响应' or not time_column or not screen or risk)
        if blocked:
            st.info('暂不能确认方案：请明确短期目标、时间字段并完成预检。若存在单位根风险，请先决定转换或其他建模方案，再上传相应数据；系统不会自动转换。')
        if not advice['design']:
            st.info('下一步：回答上方的研究背景问题。没有候选时不能确认；平稳性检查通过并不自动产生研究设计。')
        if st.button('采用建议并确认研究设计', disabled=blocked or not advice['design']):
            st.session_state['advisor_accepted'] = advice['design']
            st.session_state['advisor_manual'] = False

    with st.expander('高级设置：手动选择研究设计', expanded=False):
        manual = st.checkbox('我要手动配置（仍须通过检查）', key='advisor_manual')
        manual_design = st.selectbox('手动研究设计', [None, *design_labels], key='advisor_manual_design', format_func=lambda x: x or '待确认') if manual else None
    design = design_labels.get(manual_design) if manual else st.session_state.get('advisor_accepted')
    # A manual choice must not bypass a known time-series risk or unresolved goal.
    if design == 'time_series' and (structure != '同一对象随时间变化' or blocked):
        design = None
    if design is None:
        st.info('研究方案：待确认。尚未生成可执行研究契约。')
        return None, None, None
    estimands = ['ATT'] if design in {'difference_in_differences', 'synthetic_control'} else ['LATE'] if design in {'instrumental_variable', 'regression_discontinuity'} else ['IRF'] if design == 'time_series' else ['ATE', 'CATE']
    estimand = estimands[0]
    if len(estimands) > 1:
        estimand = st.selectbox('效应目标', estimands, help='ATE：平均效应；CATE：不同群体的效应。')
    if design in {'difference_in_differences', 'synthetic_control'} and structure != '同一对象随时间变化':
        time_column = st.selectbox('执行时间字段', [None, *frame.columns])
    label = next(label for label, value in design_labels.items() if value == design)
    st.success(f'已确认方案：{label}；效应目标：{estimand}；时间字段：{time_column or "未指定"}')
    st.caption('下面仅需补充控制变量、研究语境及设计专属字段。技术适配不代表因果假设成立。')
    return design, estimand, time_column

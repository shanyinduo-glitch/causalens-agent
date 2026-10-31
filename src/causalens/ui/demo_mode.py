"""A thin presentation layer over the production contract runner."""
import json
from pathlib import Path

from causalens.causal.contract_runner import run_explicit_causal_contract
from causalens.ui.causal_guidance import contract_result_to_agent_state
from causalens.ui.data_workbench import ROOT, bound_metadata, digest
from causalens.ui.analysis_identity import analysis_identity
from causalens.reporting import generate_markdown_report
from causalens.reporting.plain_summary import summarize_time_series
from causalens.ui.result_comparison import render_comparison

DATA = ROOT / 'data/private/gold_tips_thesis/processed/gold_tips_daily_stationary.csv'
CONTRACT = ROOT / 'examples/case_studies/gold_tips_thesis/lp_core.json'
QUESTION = 'TIPS实际收益率变化与黄金收益率在当期及随后观测期的动态关系'


def load_demo(data_path=DATA, contract_path=CONTRACT):
    if not data_path.is_file() or not contract_path.is_file():
        raise ValueError('本地演示数据或研究配置缺失。请按案例指南安装数据；系统不会自动下载私人数据。')
    raw = data_path.read_bytes()
    meta = bound_metadata(raw)
    if not meta or not meta.get('verified_transform') or meta.get('kind') != 'stationary':
        raise ValueError('数据未匹配核验记录。请先运行 scripts/audit_thesis_data.py，不自动采用同名文件。')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract['unit_of_analysis'] = '一个保留的交易日观测'
    contract['outcome_horizon'] = '当期至未来5个观测期'
    return raw, contract, meta


def render_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    st.header('论文案例演示')
    st.caption('预设研究配置，不是LLM自动推断。复用正式检查与统计估计流程，无模型调用、无历史结果替代。')
    if st.button('加载论文演示案例', type='primary'):
        st.session_state['demo_loaded'] = True
        st.session_state.pop('demo_result', None)
    if not st.session_state.get('demo_loaded'):
        st.info('第一步：加载本地已核验的案例数据，无需手动上传。')
        return
    try:
        raw, contract, meta = load_demo()
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    st.subheader('分析方案确认卡')
    st.write('研究问题：' + QUESTION)
    st.write('数据：论文转换版；影响因素：TIPS收益率变化；观察结果：黄金对数收益率。')
    st.write('方法：局部投影；基础控制：美元、标普500、联邦基金利率变化与VIX；效应目标：动态响应。')
    st.caption('数据标识：' + digest(raw)[:12])
    st.warning('这是探索性研究。配置与转换核验不代表已证明因果关系；尚有上游时序与样本选择风险。')
    with st.expander('查看数据风险与字段依据'):
        for risk in meta['risks']:
            st.write('- ' + risk)
        st.dataframe(meta['knowledge'], hide_index=True)
    with st.expander('修改专业设置'):
        contract['treatment_lags'] = st.number_input('处理变量滞后阶数', 1, 10, 1, key='demo_tlags')
        contract['outcome_lags'] = st.number_input('结果变量滞后阶数', 1, 10, 1, key='demo_ylags')
        contract['control_lag'] = st.number_input('控制变量滞后阶数', 1, 10, 1, key='demo_clags')
        contract['max_horizon'] = st.number_input('最大响应期数', 1, 20, 5, key='demo_horizon')
        contract['cumulative_outcome'] = st.checkbox('累计响应', value=True, key='demo_cumulative')
        st.caption('更多研究设计与控制方案请使用“自定义分析”，演示入口不自动搜索最佳参数。')
    contract['outcome_horizon'] = f'当期至未来{contract["max_horizon"]}个观测期'
    st.write(f'当前设置：处理/结果/控制滞后 {contract["treatment_lags"]}/{contract["outcome_lags"]}/{contract["control_lag"]} 期；最大响应 {contract["max_horizon"]} 期；' + ('累计响应' if contract['cumulative_outcome'] else '单期响应'))
    identity = analysis_identity(raw, 'demo', QUESTION, contract)
    agreed = st.checkbox('我已了解预设方案和风险，确认执行本次分析', key='demo_agree_' + identity['signature'])
    if st.button('确认并运行（实时统计）', disabled=not agreed):
        st.session_state.pop('demo_result', None)
        with st.spinner('正在校验研究契约并运行统计估计…'):
            try:
                result = run_explicit_causal_contract(dataset_path=DATA, contract=contract, approved=True, result_cache_dir=None)
                state = contract_result_to_agent_state(result, dataset_path=str(DATA), question=QUESTION)
            except Exception as exc:
                st.error('本次计算失败，没有使用历史结果替代。')
                with st.expander('错误详情'):
                    st.write(str(exc))
            else:
                st.session_state['demo_result'] = {'identity': identity, 'state': state}
    saved = st.session_state.get('demo_result')
    if not saved:
        return
    if saved['identity'] != identity:
        st.warning('参数或数据已修改，旧结果暂停展示。请重新确认并运行。')
        return
    state = saved['state']
    if state.get('workflow_status') != 'completed':
        st.error('本次检查或计算未完成。没有使用历史示例替代。')
        st.write(state.get('final_answer'))
        return
    st.success('本次实时统计已完成；当前参数与结果一致。')
    st.markdown(summarize_time_series(state) or '未返回可解释的时间序列结果。')
    for tool in state.get('tool_results', []):
        suite = tool.get('result') or {}
        for estimate in suite.get('estimates', []):
            if estimate.get('analysis_type') != 'time_series_local_projection':
                continue
            rows = estimate.get('dynamic_effects') or []
            chart_data = pd.DataFrame([{'观测期': r['horizon'], '估计': r['estimate'] * 10, '下界': r['confidence_interval'][0] * 10, '上界': r['confidence_interval'][1] * 10} for r in rows])
            if not chart_data.empty:
                base = alt.Chart(chart_data).encode(x=alt.X('观测期:Q', axis=alt.Axis(tickMinStep=1)))
                band = base.mark_area(opacity=0.2).encode(y=alt.Y('下界:Q', title='黄金收益率响应（%，10基点冲击）'), y2='上界:Q')
                line = base.mark_line(point=True).encode(y='估计:Q', tooltip=list(chart_data.columns))
                st.altair_chart(band + line, width='stretch')
                st.caption('阴影为各期95%置信区间，不是整条曲线的同时置信带；百分比为对数变化的小幅近似。')
    st.download_button('下载本次中文报告', generate_markdown_report(state), 'demo_report.md', 'text/markdown')
    render_comparison(state)
    with st.expander('查看完整配置与原始证据'):
        st.json(state)

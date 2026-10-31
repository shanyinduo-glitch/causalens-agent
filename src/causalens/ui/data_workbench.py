"""Version-bound local knowledge, explicit transformations and reproducibility audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'docs/case_studies/gold_tips_thesis/verified_data.json'
LOGS = {'y': 'y_ret', 'dollar': 'dollar_ret', 'sp500': 'sp500_ret', 'comex_gold': 'comex_gold_ret', 'global_holdings_tonnes': 'global_holdings_tonnes_ret'}
DIFFS = {'t': 't_diff', 'fed_funds': 'fed_funds_diff'}
LEVELS = ['vix', 'cpi_yoy', 'infl_exp5y', 'gprh']
RISKS = ['CPI月份匹配方向与源代码注释相反，尚未修复。', 'GPR与ETF按源日期填充，真实发布日期尚未核实。', '删行后的相邻观测不保证是连续交易日；转换不能修复上游时序风险。']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def transform_thesis(frame):
    required = ['date', *LOGS, *DIFFS, *LEVELS]
    if any(c not in frame for c in required):
        raise ValueError('缺少论文转换所需字段，不能套用此配方。')
    dates = pd.to_datetime(frame['date'], errors='coerce')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('日期必须有效、唯一且已升序排列。请先核对排序和重复记录。')
    numeric = frame[required[1:]]
    if not all(pd.api.types.is_numeric_dtype(numeric[c]) for c in numeric) or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('存在缺失、非数值或无穷值；不会自动填补或删除原始异常。')
    if (frame[list(LOGS)] <= 0).any().any():
        raise ValueError('对数转换要求所有相关原始值大于零。')
    out = pd.DataFrame({'date': frame['date']})
    out['y_ret'] = np.log(frame['y']).diff()
    out['t_diff'] = frame['t'].diff()
    for source, target in LOGS.items():
        out[target] = np.log(frame[source]).diff()
    for source, target in DIFFS.items():
        out[target] = frame[source].diff()
    for source in LEVELS:
        out[source + '_level'] = frame[source]
    out['time_trend'] = np.arange(1, len(frame) + 1)
    out['y_ret_lag1'] = out['y_ret'].shift(1)
    out['t_diff_lag1'] = out['t_diff'].shift(1)
    order = ['date', 'y_ret', 't_diff', 'dollar_ret', 'sp500_ret', 'fed_funds_diff', 'comex_gold_ret', 'global_holdings_tonnes_ret', 'vix_level', 'cpi_yoy_level', 'infl_exp5y_level', 'gprh_level', 'time_trend', 'y_ret_lag1', 't_diff_lag1']
    return out[order].dropna().reset_index(drop=True)


def bound_metadata(raw):
    if not MANIFEST.exists():
        return None
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    return manifest['datasets'].get(digest(raw))


def lookup_knowledge(metadata, query):
    """Dataset-scoped lexical retrieval, not a general-purpose semantic RAG."""
    records = metadata.get('knowledge', []) if metadata else []
    terms = query.lower().split()
    scored = [(sum(term in (r['text'] + ' ' + r['field']).lower() for term in terms), r) for r in records]
    return [r for score, r in sorted(scored, key=lambda x: -x[0]) if score][:5]


def render_data_workbench(raw):
    import streamlit as st
    metadata = bound_metadata(raw)
    with st.expander('数据说明、处理与知识依据', expanded=bool(metadata)):
        if not metadata:
            st.info('未匹配已核验的数据版本；不会套用论文的单位或转换公式。仍可按普通流程分析。')
            return
        st.write(metadata['name'])
        for risk in metadata['risks']:
            st.warning(risk)
        st.dataframe(pd.DataFrame(metadata['knowledge']), hide_index=True)
        query = st.text_input('查阅本数据说明（输入字段名或关键词，例如 y_ret）')
        if query:
            matches = lookup_knowledge(metadata, query)
            if not matches:
                st.info('当前绑定资料没有匹配依据，不推测回答。')
            for record in matches:
                st.write(record['text'])
                st.caption('来源：' + record['source'])
        if metadata['kind'] == 'levels':
            st.write('可选：复现论文的短期变化配方。价格/库存/持仓取对数差分，利率取差分，指定指标保留水平，并生成趋势与滞后。通常损失开头两行。此操作改变问题，不适用于长期水平关系。')
            approved = st.checkbox('我确认研究短期变化，接受该配方和上述未解决风险；保留原文件', key='prepare_' + digest(raw))
            if st.button('生成独立转换文件', disabled=not approved):
                from io import BytesIO
                try:
                    transformed = transform_thesis(pd.read_csv(BytesIO(raw)))
                    from causalens.ui.research_advisor import inspect_stationarity
                    check = inspect_stationarity(transformed, treatment='t_diff', outcome='y_ret', time_column='date')
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.write(f'生成 {len(transformed)} 行；原数据未修改。可在下方直接使用转换版本继续分析，也可下载留档。')
                    record = {'source_sha256': digest(raw), 'recipe': 'thesis_v1', 'output_rows': len(transformed), 'risks': metadata['risks'], 'checks': check}
                    content = transformed.to_csv(index=False).encode('utf-8')
                    record['output_sha256'] = digest(content)
                    record['dropped_initial_rows'] = 2
                    st.session_state['prepared_' + digest(raw)] = (content, record)
            prepared = st.session_state.get('prepared_' + digest(raw))
            if prepared:
                st.json(prepared[1]['checks'])
                st.download_button('下载转换后的CSV', prepared[0], 'gold_tips_prepared.csv', 'text/csv')
                st.download_button('下载转换记录', json.dumps(prepared[1], ensure_ascii=False, indent=2), 'preparation_record.json')


def metadata_for_state(state):
    path = state.get('dataset_path')
    if not path or not Path(path).is_file():
        return None
    return bound_metadata(Path(path).read_bytes())


def clear_active_analysis(session, fingerprint):
    """Clear only analysis state when the bytes/source of the active dataset change."""
    if session.get('active_data_fingerprint') == fingerprint:
        return
    for key in ('final_state', 'current_dataset_path', 'current_question', 'current_contract',
                'current_execution_mode', 'guided_ai_suggestions', 'advisor_accepted',
                'advisor_screen', 'advisor_screen_error', 'advisor_context', 'advisor_scope',
                'advisor_manual_design', 'advisor_time', 'comparison_upload'):
        session.pop(key, None)
    session['active_data_fingerprint'] = fingerprint


def select_analysis_version(uploaded):
    """Return a BytesIO-compatible upload consumed by both preview and execution."""
    import streamlit as st
    from io import BytesIO
    raw = uploaded.getvalue()
    source_hash = digest(raw)
    render_data_workbench(raw)
    prepared = st.session_state.get('prepared_' + source_hash)
    options = ['原始上传版本']
    if prepared:
        options.append('转换版本（短期变化）')
    choice = st.radio('本次分析使用哪个版本？', options, key='active_version_' + source_hash,
                      help='选择转换版本即可继续，无需下载后重新上传。切换会清除旧分析结果并要求重新确认方案。')
    selected = prepared[0] if prepared and choice == options[-1] and len(options) > 1 else raw
    clear_active_analysis(st.session_state, source_hash + ':' + digest(selected))
    result = BytesIO(selected)
    result.name = 'gold_tips_prepared.csv' if selected is not raw else uploaded.name
    st.caption(f'当前版本：{choice}；校验标识：{digest(selected)[:12]}。原始上传文件保持不变。')
    if selected is not raw:
        st.info('已切换到转换数据。下方预览、变量选择及执行均使用该版本；请重新确认处理变量与结果变量。原有数据时序风险仍保留。')
    return result

"""Explicit saved-run comparison without automatically searching for significance."""
import json
from pathlib import Path
from causalens.ui.data_workbench import digest


def snapshot(state):
    path = Path(state.get('dataset_path') or '')
    if not path.is_file() or state.get('workflow_status') != 'completed':
        return None
    for tool in state.get('tool_results', []):
        if hasattr(tool, 'model_dump'):
            tool = tool.model_dump()
        suite = tool.get('result') or {}
        if tool.get('success') and suite.get('design_type') == 'time_series':
            return {'format': 'causalens-comparison-v1', 'dataset_sha256': digest(path.read_bytes()), 'contract': state.get('causal_question', {}), 'suite': suite}
    return None


def compare(left, right):
    if not isinstance(right, dict) or right.get('format') != 'causalens-comparison-v1':
        raise ValueError('不是本项目导出的方案记录。')
    if left['dataset_sha256'] != right.get('dataset_sha256'):
        raise ValueError('数据版本不同，不能作为同数据稳健性比较。')
    for key in ['treatment', 'outcome', 'estimand', 'design_type']:
        if left['suite'].get(key) != right.get('suite', {}).get(key):
            raise ValueError('变量或效应目标不同，不能直接比较。')
    rows = []
    configs = []
    for name, record in [('当前方案', left), ('已保存方案', right)]:
        estimate = next(e for e in record['suite']['estimates'] if e.get('analysis_type') == 'time_series_local_projection')
        configs.append(estimate.get('model_configuration', {}))
        for row in estimate.get('dynamic_effects', []):
            rows.append({'方案': name, '期数': row['horizon'], '估计': row['estimate'], '置信区间': row.get('confidence_interval'), '样本量': row.get('sample_size')})
    if configs[0].get('response') != configs[1].get('response'):
        raise ValueError('累计与单期响应口径不同，不能直接比较。')
    return rows


def render_comparison(state):
    import streamlit as st
    current = snapshot(state)
    if not current:
        return
    with st.expander('保存方案并比较结果'):
        st.caption('先保存本次方案；按事先确定的控制变量或滞后配置重新分析，再上传旧方案比较。不要按显著性挑选结果。')
        st.download_button('保存本次方案与结果', json.dumps(current, ensure_ascii=False, indent=2, default=str), 'causalens_run.json', 'application/json')
        uploaded = st.file_uploader('上传另一个已保存方案', type=['json'], key='comparison_upload')
        if uploaded:
            try:
                other = json.loads(uploaded.getvalue())
                rows = compare(current, other)
            except (ValueError, KeyError, TypeError, StopIteration) as exc:
                st.warning(f'无法比较：{exc}')
            else:
                st.dataframe(rows, hide_index=True)
                st.write('两份配置（用于核对控制变量、滞后和研究假设）：')
                st.json({'当前': current['contract'], '已保存': other['contract']})
                st.caption('方向一致不能证明因果识别；不同有效样本量也可能影响比较。')

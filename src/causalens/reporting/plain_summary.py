"""Evidence-only summaries; no model calls or inferred measurement units."""
import math
from causalens.ui.data_workbench import metadata_for_state


def summarize_time_series(state):
    for item in state.get('tool_results', []) or []:
        if hasattr(item, 'model_dump'):
            item = item.model_dump()
        suite = item.get('result') or {}
        if not item.get('success') or not isinstance(suite, dict) or suite.get('design_type') != 'time_series':
            continue
        estimate = next((e for e in suite.get('estimates', []) if e.get('analysis_type') == 'time_series_local_projection'), None)
        if not estimate:
            continue
        rows = estimate.get('dynamic_effects') or []
        usable = [r for r in rows if isinstance(r.get('estimate'), (int, float)) and math.isfinite(r['estimate'])]
        lines = ['## 先看结论', '', '### 发现了什么？', '']
        metadata = metadata_for_state(state)
        known_units = bool(metadata and metadata.get('verified_transform') and metadata.get('kind') == 'stationary' and suite.get('treatment') == 't_diff' and suite.get('outcome') == 'y_ret')
        if not usable:
            lines.append('缺少可用的动态响应数值，暂时不能概括影响方向。')
        else:
            signs = {1 if r['estimate'] > 0 else -1 if r['estimate'] < 0 else 0 for r in usable}
            direction = '均为负值' if signs == {-1} else '均为正值' if signs == {1} else '随观察期变化或包含零值'
            lines.append(f'在当前数据和模型设置下，已返回的响应估计{direction}。这是模型中的条件响应，不是已证明的因果关系。')
            lines += ['', '### 影响大约多大？', '']
            cumulative = (estimate.get('model_configuration') or {}).get('response') == 'cumulative outcome from t through t+h'
            for row in [usable[0], *([usable[-1]] if len(usable) > 1 else [])]:
                horizon = row.get('horizon', '未知')
                period = '当期' if horizon == 0 else (f'当期至第{horizon}个观测期累计' if cumulative else f'第{horizon}个观测期单期')
                ci = row.get('confidence_interval')
                uncertainty = '未提供可用的95%置信区间，不能判断方向是否明确。'
                if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in ci):
                    uncertainty = f'95%置信区间为 [{ci[0]:.6g}, {ci[1]:.6g}]；' + ('包含零，方向仍不明确。' if ci[0] <= 0 <= ci[1] else '不包含零，但这不证明因果关系。')
                if known_units:
                    interval = '未提供可用的95%置信区间。'
                    if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in ci):
                        interval = f'95%置信区间：{ci[0] * 10:.3f}% 至 {ci[1] * 10:.3f}%。' + ('包含零，方向仍不明确。' if ci[0] <= 0 <= ci[1] else '')
                    lines.append(f'- **{period}**：TIPS收益率上升10个基点，对应黄金收益率约变化 **{row["estimate"] * 10:.3f}%**。{interval}')
                else:
                    lines.append(f'- {period}：处理变量增加1个原始单位，结果变量的估计响应为 {row["estimate"]:.6g} 个原始单位。{uncertainty}')
            lines += ['', ('换算依据：已核验数据字典；百分比为对数变化的小幅近似，原始系数见技术详情。' if known_units else '未确认单位定义，因此不自动换算为百分比或基点。') + '期数按数据观测顺序计算，不一定等于自然日。']
            if cumulative:
                lines.append('这里是累计响应，不能把不同期数的结果再次相加，也不能把最后一期解释为当天的单日变化。')
        lines += ['', '### 可以下什么结论？', '', '可以把结果作为研究线索；仅凭本次输出不能认定“处理变量导致结果变化”，也不应当作预测。其他共同因素与冲击的外生性仍需研究依据。']
        residual = estimate.get('residual_diagnostics') or {}
        if residual.get('residual_autocorrelation_warning'):
            lines.append('本次还发现：模型残差存在时间规律，说明动态设定仍需检查。稳健标准误不能修复遗漏因素。')
        placebo = estimate.get('placebo_test') or {}
        if placebo.get('passed') is False:
            lines.append('未来冲击检查未通过，存在额外识别风险，应先调查再解释结果。')
        if metadata:
            lines += ['', '### 数据来源仍有哪些限制？', '', '以下为上游数据风险，不代表所有相关字段都进入了本次模型。即使未选入，上游合并删行仍可能影响样本。']
            lines += ['- ' + risk for risk in metadata['risks']]
        lines += ['', '### 下一步做什么？', '', '保留本次配置和报告。先核实变量单位与控制变量的时间顺序，再用事先确定的替代滞后或控制方案检查结果是否稳定；不要以获得显著结果为目标调参。', '']
        return '\n'.join(lines)
    return None

"""中文时间序列报告；保留原始证据供审计，不调用语言模型。"""
import json
from datetime import datetime
from causalens.reporting.plain_summary import summarize_time_series
from pathlib import Path
from causalens.ui.analysis_identity import analysis_identity


def time_series_report(state, *, dataset_path=None, generated_at=None):
    suite = None
    for item in state.get('tool_results', []):
        if hasattr(item, 'model_dump'):
            item = item.model_dump()
        result = item.get('result') or {}
        if item.get('success') and isinstance(result, dict) and result.get('design_type') == 'time_series':
            suite = result
            break
    if not suite:
        return None
    estimates = suite.get('estimates') or []
    estimate = next((e for e in estimates if e.get('analysis_type') == 'time_series_local_projection'), None)
    if not estimate:
        return None
    config = estimate.get('model_configuration') or {}
    grade = suite.get('evidence_grade') or {}
    placebo = estimate.get('placebo_test') or {}
    residual = estimate.get('residual_diagnostics') or {}
    stationarity = estimate.get('stationarity') or {}
    status = state.get('workflow_status', '未提供')
    status = {'completed': '已完成', 'failed': '失败', 'awaiting_approval': '等待确认'}.get(status, status)
    def number(value):
        return '未提供' if value is None else f'{value:.6g}'
    def verdict(value):
        return {True: '是', False: '否', None: '未提供'}.get(value, str(value))
    rows = estimate.get('dynamic_effects') or []
    path = Path(dataset_path or state.get('dataset_path') or '')
    identity = analysis_identity(path.read_bytes(), 'report', state.get('question') or '', state.get('causal_question')) if path.is_file() else None
    lines = [
        '# CausaLens 因果分析报告', '',
        summarize_time_series(state) or '',
        '## 技术详情与复现记录', '',
        '## 执行概况', '',
        ('- 数据 SHA256：`' + identity['data_sha256'] + '`；报告输入标识：`' + identity['signature'] + '`') if identity else '- 数据校验值：未提供可读取的数据文件。',
        f'- 生成时间：{(generated_at or datetime.now().astimezone()).isoformat()}',
        f'- 数据文件：`{dataset_path or state.get("dataset_path", "未提供")}`',
        f'- 工作流状态：{status}',
        '- 分析方法：时间序列局部投影（Local Projection）',
        '## 研究问题与变量', '', str(state.get('question') or '执行已确认的因果研究契约'), '',
        f'- 处理变量：`{suite.get("treatment")}`',
        f'- 结果变量：`{suite.get("outcome")}`',
        f'- 控制变量：{", ".join(suite.get("confounders") or []) or "无"}',
        '- 估计目标：连续处理变量增加一个原始单位后的脉冲响应（IRF）。',
        '- 效应单位由输入数据决定，不能直接解释为二元处理的平均处理效应。',
        '## 模型设置', '',
        f'- 最大响应期数：{config.get("max_horizon", "未提供")}',
        f'- 处理变量滞后阶数：{config.get("treatment_lags", "未提供")}',
        f'- 结果变量滞后阶数：{config.get("outcome_lags", "未提供")}',
        f'- 控制变量滞后阶数：{config.get("control_lag", "未提供")}',
        '- 响应口径：' + ('从冲击当期至各期的累计响应' if config.get('response') == 'cumulative outcome from t through t+h' else '各响应期的单期响应'),
        '- 标准误：Newey-West 异方差与自相关稳健标准误（HAC）',
        '## 动态响应结果', '',
        '| 响应期 h | 效应估计 | 标准误 | 95%置信区间 | p值 | 有效样本量 |',
        '| ---: | ---: | ---: | --- | ---: | ---: |',
    ]
    for row in rows:
        ci = row.get('confidence_interval') or [None, None]
        lines.append(f'| {row.get("horizon")} | {number(row.get("estimate"))} | {number(row.get("standard_error"))} | [{number(ci[0])}, {number(ci[1])}] | {number(row.get("p_value"))} | {row.get("sample_size")} |')
    lines += ['', '## 诊断与证据等级', '',
        f'- 处理变量 ADF 单位根检验 p值：{number((stationarity.get("treatment") or {}).get("p_value"))}',
        f'- 结果变量 ADF 单位根检验 p值：{number((stationarity.get("outcome") or {}).get("p_value"))}',
        f'- 未来处理变量安慰剂检验 p值：{number(placebo.get("p_value"))}',
        f'- 安慰剂检验通过：{verdict(placebo.get("passed"))}',
        f'- Ljung-Box 残差自相关检验 p值：{number(residual.get("p_value"))}',
        f'- 存在残差自相关警告：{verdict(residual.get("residual_autocorrelation_warning"))}',
        f'- 证据等级：{grade.get("label_zh") or grade.get("grade", "未提供")}',
        '', '证据等级汇总当前可用诊断，并不表示因果结论为真的概率。安慰剂检验通过也不能证明不存在混杂。',
        '', '## 识别假设与结果限制', '',
        '- 在控制声明的滞后变量后，处理冲击仍需满足相对于未来潜在结果近似外生的假设。',
        '- 滞后设定应覆盖重要动态关系，分析区间内关系应足够稳定。',
        '- 同时影响处理变量和结果变量的遗漏冲击仍可能造成偏差。',
        '- HAC 调整标准误，但不能建立因果识别。',
    ]
    if residual.get('residual_autocorrelation_warning'):
        lines.append('- 本次残差仍存在自相关，应检查滞后长度与模型动态设定，并将结果作为探索性证据解读。')
    lines += ['', '## 原始证据附录', '', '以下保留原始字段、诊断和警告，方便程序读取与结果核验；英文标识不影响正文阅读。', '', '```json', json.dumps(suite, ensure_ascii=False, indent=2, default=str), '```', '']
    context = {key: state[key] for key in ('causal_question', 'causal_identification', 'causal_data_quality', 'causal_readiness', 'causal_diagnostics') if key in state}
    if context:
        lines += ['### 研究配置与检查记录', '', '```json', json.dumps(context, ensure_ascii=False, indent=2, default=str), '```', '']
    return '\n'.join(lines)

"""Deterministic readable evidence when the narrative model is unavailable."""
def tool_fallback(results):
    lines = ['统计工具结果已保留；AI 文字解读暂不可用。以下为程序直接整理的计算事实，不是模型解读。', '']
    fields = set()
    for item in results:
        result = item.get('result') or {}
        if item.get('success') and item.get('tool_name') == 'create_chart':
            fields.update(x for x in (result.get('x_column'), result.get('y_column')) if x)
    for item in results:
        if not item.get('success'):
            lines.append('- 有工具未成功，请查看技术记录；本摘要不代表任务全部完成。')
            continue
        result = item.get('result') or {}
        if item.get('tool_name') == 'profile_dataset':
            lines.append(f"- 数据规模：{result.get('rows', '未知')} 行，{result.get('columns', '未知')} 个字段。")
            missing = result.get('missing_values', {})
            summaries = result.get('numeric_summary', {})
            for field in sorted(fields or summaries.keys()):
                if field in missing:
                    lines.append(f'- `{field}` 缺失数量：{missing[field]}。')
                summary = summaries.get(field, {})
                values = '；'.join(f'{label} {summary[key]:.6g}' for key, label in
                                  [('mean', '均值'), ('std', '标准差'), ('min', '最小值'), ('max', '最大值')]
                                  if isinstance(summary.get(key), (int, float)))
                if values:
                    lines.append(f'- `{field}`：{values}（原始数据单位）。')
        elif item.get('tool_name') == 'create_chart':
            lines.append(f"- 已生成图表：`{result.get('x_column', '')}` 与 `{result.get('y_column', '')}`。")
    lines.extend(['', '统计差异或相关性不证明因果关系。原始输出保存在技术记录中；可选择“只解读上一轮结果”重新请求文字解读，无需重跑计算。'])
    return '\n\n'.join(lines)

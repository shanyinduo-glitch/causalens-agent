"""Report the submitted planning snapshot, never a later session draft."""
from causalens.ui.method_knowledge import comparison_rows
from causalens.ui.planning_handoff import text_items, CONTEXT_LABELS


def planning_section(state):
    trace = state.get('planning_trace')
    if not isinstance(trace, dict):
        return ''
    value = trace.get('value', {})
    contract = trace.get('final_contract', {})
    lines = ['## 研究规划与人工确认记录', '',
             '以下为提交分析时的规划快照；AI 草案与引用不构成因果识别证明。', '',
             f"- 规划路径：{trace.get('route', '未记录')}",
             f"- 模型：{value.get('model', '未记录')}",
             f"- 草案标识：`{trace.get('identity', '')}`",
             f"- 用户需求：{trace.get('question', '')}",
             f"- 用户补充：{trace.get('clarification') or '无'}", '', '### AI 候选与最终方案', '']
    for row in comparison_rows(value.get('method_options', [])):
        lines.extend([f"- {row['方法']}：{row['回答的问题']}；{row['系统实现状态（代码核对）']}。",
                      f"  AI 理由：{row['AI候选理由']}；待核实：{row['待核实依据']}。"])
    lines.extend(['', f"最终提交设计：`{contract.get('design_type', '未记录')}`；处理变量 `{contract.get('treatment', '')}`；结果变量 `{contract.get('outcome', '')}`。",
                  '最终配置由用户表单提交，实际执行状态与方法见统计记录；不能将所有候选视为已运行。'])
    lines.append('最终控制变量：' + '、'.join(contract.get('confounders', [])))
    proposed = value.get('adjustment', [])
    selected = contract.get('confounders', [])
    lines.append('AI 建议控制变量：' + ('、'.join(proposed) or '未记录'))
    lines.append('AI 建议但最终未选：' + ('、'.join(x for x in proposed if x not in selected) or '无'))
    lines.append('最终新增控制变量：' + ('、'.join(x for x in selected if x not in proposed) or '无'))
    lines.extend(['', '### 澄清与修订记录', ''])
    for index, revision in enumerate(trace.get('revisions', []), 1):
        validation = revision.get('validation', {})
        lines.append(f"- 历史草案 {index}：{validation.get('status', '未记录')}；时间候选：{'、'.join(revision.get('time', []))}。")
        for issue in validation.get('issues', []):
            lines.append('  校验问题：' + str(issue))
    lines.append('当前草案校验状态：' + str(value.get('validation', {}).get('status', '未记录')) + '（仅结构检查，不代表因果识别通过）。')
    lines.append(f"AI 主候选：`{value.get('design_candidate', '未记录')}`；是否与最终设计一致：{'是' if value.get('design_candidate') == contract.get('design_type') else '否'}。未单独采集人工选择理由，不能把AI理由当作用户理由。")
    for item in text_items(value.get('rationale')):
        lines.append('- AI候选理由（未验证）：' + item)
    for item in value.get('validation', {}).get('issues', []):
        lines.append('- 草案校验问题：' + str(item))
    for k, label in CONTEXT_LABELS.items():
        lines.append(f"- {label}：{contract.get(k) or '未记录'}")
    if any(x.get('method_id') == 'M-DML-C' for x in value.get('method_options', []) if isinstance(x, dict)):
        lines.extend(['', '连续处理DML仅参与方法比较，本次未执行；LP估计的是各期动态/累计响应，不能称为论文DML复现或将两者显著性直接比较。'])
    lines.extend(['', '### 规划引用依据', ''])
    refs = value.get('evidence_ids', [])
    for item in trace.get('evidence', []):
        if item.get('id') in refs:
            lines.append(f"- [{item['id']}] {item.get('source', '')} · {item.get('section', '')}：{item.get('text', '')}")
            if item.get('url'):
                lines.append(f"  来源：{item['url']}")
    lines.extend(['', '引用编号和来源可追溯，不代表已经验证引用支持每项AI判断。', ''])
    return '\n'.join(lines)

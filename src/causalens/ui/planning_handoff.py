"""Explicit handoff helpers: no new model requests or statistical decisions."""
import ast
import copy
import re

CONTEXT_LABELS = {'target_population': '结果希望适用于谁', 'unit_of_analysis': '每一行代表什么',
                  'outcome_horizon': '观察多长时间后的结果'}


def text_items(value):
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                value = parsed
        except (ValueError, SyntaxError):
            pass
    if isinstance(value, list):
        return [str(v).strip()[:3000] for v in value[:10] if isinstance(v, str) and v.strip()]
    return [value.strip()[:3000]] if isinstance(value, str) and value.strip() else []


def explicit_role_issues(text, parsed):
    issues = []
    # Only explicit declarations are treated as constraints, not inferred semantics.
    for label, field in [('处理变量', 'treatment_candidates'), ('结果变量', 'outcome_candidates'), ('时间字段', 'time_candidates')]:
        names = re.findall(label + r'\s*(?:明确为|为|是|[:：])\s*`?([A-Za-z_][A-Za-z_0-9]*)', text)
        if names and parsed.get(field) != [names[-1]]:
            issues.append(f'用户明确指定{label}为 {names[-1]}，模型候选不一致；请修订，不能自动改写。')
    return issues


def missing_context(contract):
    return [label for key, label in CONTEXT_LABELS.items() if not str((contract or {}).get(key) or '').strip()]


def attach_planning(state, snapshot):
    if snapshot:
        state['planning_trace'] = copy.deepcopy(snapshot)
    return state


def render_context_inputs(ai_value, identity):
    import streamlit as st
    proposed = (ai_value or {}).get('research_context', {})
    proposed = {k: v for k, v in proposed.items() if k in CONTEXT_LABELS and isinstance(v, str) and v.strip()}
    keys = {k: f'context_{identity}_{k}' for k in CONTEXT_LABELS}
    if proposed:
        st.caption('AI 提取的研究背景尚未确认；请检查后填入，也可以手动修改。')
        st.table([{'项目': CONTEXT_LABELS[k], '候选内容': v} for k, v in proposed.items()])
        if st.button('确认以上背景并填入表单', key='fill_context_' + identity):
            for k, v in proposed.items():
                st.session_state[keys[k]] = v
    values = {}
    for col, (k, label) in zip(st.columns(3), CONTEXT_LABELS.items()):
        with col:
            values[k] = st.text_input(label + '？', key=keys[k])
    return values['target_population'], values['unit_of_analysis'], values['outcome_horizon']

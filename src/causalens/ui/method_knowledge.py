"""Curated method evidence, separate from executable capability and case facts."""
import json
from pathlib import Path
from causalens.causal.estimators import ESTIMATOR_REGISTRY


def method_cards():
    cards = json.loads(Path(__file__).with_name('method_cards.json').read_text(encoding='utf-8'))
    for card in cards:
        card['implementation'] = ('已有执行器，仍需研究契约与适配检查'
                                  if card['estimator_key'] in ESTIMATOR_REGISTRY else '研究候选，尚未接入执行器')
        card['source'] = card['source_title']
        card['section'] = card['name']
        card['text'] = '；'.join(card[k] for k in ('target', 'when', 'assumptions', 'diagnostics', 'limits', 'implementation'))
    return cards


def validate_method_options(parsed):
    """Do not trust the model's claimed implementation/readiness status."""
    options = parsed.get('method_options')
    if not isinstance(options, list) or not 1 <= len(options) <= 3:
        return ['请返回1至3个有依据的方法候选，不要为了数量凑出不适用的方法。']
    known = {c['id']: c for c in method_cards()}
    issues, seen = [], set()
    for item in options:
        if not isinstance(item, dict) or not isinstance(item.get('method_id'), str) or item['method_id'] not in known:
            issues.append('方法候选引用了未知的方法卡片。')
            continue
        key = item['method_id']
        if key in seen:
            issues.append('方法候选重复。')
        seen.add(key)
        for field, label in (('reason', '建议理由'), ('missing_evidence', '待核实依据'),
                             ('target_alignment', '与研究目标的关系')):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                detail = ('返回了布尔值，不能代替文字解释' if isinstance(value, bool)
                          else '缺少非空文字说明')
                issues.append(f'{key} 的“{label}”（{field}）{detail}。')
    design = parsed.get('design_candidate')
    if design and design != 'unknown' and not any(known[k]['design'] == design for k in seen):
        issues.append('主设计与方法候选清单不一致。')
    return issues


def comparison_rows(options):
    cards = {c['id']: c for c in method_cards()}
    rows = []
    for item in options if isinstance(options, list) else []:
        if not isinstance(item, dict):
            continue
        key = item.get('method_id')
        if not isinstance(key, str) or key not in cards:
            continue
        card = cards[key]
        alignment = item.get('target_alignment', '')
        if isinstance(alignment, bool):
            alignment = ('模型标记为一致' if alignment else '模型标记为不一致') + '，但未提供文字解释（待修订）'
        rows.append({'方法': card['name'], '回答的问题': card['target'],
                     'AI候选理由': str(item.get('reason', ''))[:1500],
                     '与本次目标的关系': str(alignment)[:1000],
                     '待核实依据': str(item.get('missing_evidence', ''))[:1500],
                     '系统实现状态（代码核对）': card['implementation'],
                     '方法依据': card['id']})
    return rows

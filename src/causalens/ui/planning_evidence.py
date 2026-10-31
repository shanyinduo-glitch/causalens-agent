"""Bounded local retrieval and structural review for untrusted model drafts."""
import hashlib
import json
import re
from pathlib import Path
from causalens.causal.estimators import ESTIMATOR_REGISTRY

LABELS = {
    'cross_sectional': '横截面混杂调整（当前实现要求0/1处理）',
    'difference_in_differences': '双重差分 / 事件研究',
    'instrumental_variable': '工具变量（2SLS）',
    'regression_discontinuity': '清晰断点回归',
    'synthetic_control': '合成控制',
    'time_series': '时间序列局部投影（LP，动态响应）',
    'unknown': '尚不能确定，需要补充背景',
}


def capabilities():
    return {'design_labels': LABELS, 'registered_estimators': [s.to_dict() for s in ESTIMATOR_REGISTRY.values()],
            'rules': ['time_series 是已支持的设计，指单条有序序列的连续冲击局部投影，不是VAR，也不是横截面回归。',
                      '字段中有滞后列不代表已满足识别假设。局部投影不限于高频或短面板。',
                      '当前横截面DML只支持0/1处理，不代表DML理论不支持连续处理。',
                      'DID需要组别与处理时点，IV需要外生工具依据，RDD需要分配阈值；未提供依据应说尚无法确认，不能推断不存在。',
                      '支持某方法不代表当前数据适合；不可声称检验已通过，不可替用户确认外生性。']}


def _terms(text):
    text = text.lower()
    terms = set(re.findall(r'[a-z][a-z0-9_]+', text))
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        terms.update(run[i:i+2] for i in range(len(run)-1))
    return terms


def retrieve_case_evidence(metadata, query, limit=6):
    if not metadata:
        return []
    records = []
    for note in metadata.get('knowledge', []):
        records.append({'source': '已核验案例数据说明', 'section': str(note.get('field', '')),
                        'text': str(note.get('text', ''))})
    for risk in metadata.get('risks', []):
        records.append({'source': '已核验案例数据风险', 'section': '上游风险', 'text': risk})
    path = Path(__file__).resolve().parents[3] / 'docs/case_studies/gold_tips_thesis/planning_knowledge.json'
    if path.exists():
        records.extend(json.loads(path.read_text(encoding='utf-8')))
    terms = _terms(query)
    hits = []
    for record in records:
        score = len(terms & _terms(record['section'] + ' ' + record['text']))
        if score:
            key = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
            hits.append({**record, 'id': 'E-' + key, 'score': score})
    return sorted(hits, key=lambda r: (-r['score'], r['id']))[:limit]


def validate_plan(parsed, frame, evidence):
    issues = []
    fields = ('treatment_candidates', 'outcome_candidates', 'time_candidates', 'confounder_candidates')
    for name in fields:
        values = parsed.get(name, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or v not in frame.columns for v in values):
            issues.append(f'{name} 含无效格式或不存在的字段，不能采用。')
    t = parsed.get('treatment_candidates', [])
    y = parsed.get('outcome_candidates', [])
    t = t if isinstance(t, list) else []
    y = y if isinstance(y, list) else []
    if not t or not y:
        issues.append('尚未提供完整的处理与结果候选。')
    if t and y and t[0] == y[0]:
        issues.append('首选处理变量与结果变量相同。')
    controls = parsed.get('confounder_candidates', [])
    if isinstance(controls, list) and any(v in controls for v in t[:1] + y[:1]):
        issues.append('首选处理/结果变量与控制候选冲突。')
    design = parsed.get('design_candidate')
    if not isinstance(design, str) or design not in LABELS or design == 'unknown':
        issues.append('方法尚未明确或返回了未支持的方法标识；不会自动改成其他方法。')
    if design == 'cross_sectional' and t and isinstance(t[0], str) and t[0] in frame:
        if set(frame[t[0]].dropna().unique()) != {0, 1}:
            issues.append('当前横截面实现要求0/1处理，与首选字段不符。')
    if design == 'time_series' and not parsed.get('time_candidates'):
        issues.append('局部投影尚缺时间字段候选。')
    explanation = str(parsed.get('rationale', '')) + '\n' + str(parsed.get('alternatives', ''))
    if re.search(r'time_series[^。\n]{0,100}(?:不在允许|不在.*选项|系统不支持)', explanation):
        issues.append('方法说明疑似否认系统支持time_series，与能力清单矛盾，请修订。')
    refs = parsed.get('evidence_ids', [])
    allowed = {r['id'] for r in evidence}
    if not isinstance(refs, list) or any(not isinstance(r, str) or r not in allowed for r in refs):
        issues.append('引用了未提供的证据编号。')
    elif evidence and not refs:
        issues.append('检索到了背景材料，但模型没有标明所用证据。')
    return {'status': 'needs_revision' if issues else 'candidate_only', 'issues': issues,
            'scope': '仅检查结构、字段、部分方法条件和引用编号；不保证自由文本正确或因果识别成立。'}

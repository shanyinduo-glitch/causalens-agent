"""Read-only source audit; writes new audit/metadata artifacts inside the project."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from causalens.causal.data_lineage import transformation_lineage
from causalens.ui.data_workbench import ROOT, MANIFEST, RISKS, LOGS, DIFFS, LEVELS, digest, transform_thesis


def main():
    folder = ROOT / 'data/private/gold_tips_thesis'
    source = folder / 'aligned/gold_tips_daily_levels.csv'
    target = folder / 'processed/gold_tips_daily_stationary.csv'
    levels, existing = pd.read_csv(source), pd.read_csv(target)
    calculated = transform_thesis(levels)
    checks = {}
    for col in calculated:
        same_shape = len(calculated) == len(existing) and col in existing
        checks[col] = bool(same_shape and (calculated[col].equals(existing[col]) if col == 'date' else np.allclose(calculated[col], existing[col], atol=1e-12, rtol=1e-10)))
    source_ref = 'E:/graduation_project/causal_analysis_scripts/02_descriptive_stats.py：转换与滞后构造'
    labels = {'y': '黄金下午定盘价，美元/盎司', 't': '10年期TIPS实际收益率，百分数口径', 'dollar': '名义广义美元指数', 'sp500': '标普500指数', 'fed_funds': '联邦基金利率，百分数口径', 'comex_gold': 'COMEX黄金库存，盎司', 'global_holdings_tonnes': '全球黄金ETF持仓，吨', 'vix': 'VIX指数', 'cpi_yoy': 'CPI同比，百分数', 'infl_exp5y': '5年期盈亏平衡通胀率，百分数口径', 'gprh': '地缘政治风险指数'}
    original = [{'field': k, 'text': v, 'source': '01_data_preprocessing.py：读取、重命名及单位注释'} for k, v in labels.items()]
    derived = [{'field': v, 'text': f'{labels[k]}的相邻观测对数差分，未乘100。', 'source': source_ref} for k, v in LOGS.items()]
    derived += [{'field': v, 'text': f'{labels[k]}的相邻观测差分，单位为百分点。', 'source': source_ref} for k, v in DIFFS.items()]
    derived += [{'field': k + '_level', 'text': labels[k] + '，保留原值。', 'source': source_ref} for k in LEVELS]
    derived += [{'field': 'time_trend', 'text': '删差分缺失前的观测序号，从1开始。', 'source': source_ref}, {'field': 'y_ret_lag1 t_diff_lag1', 'text': '对应转换变量的前一条有效输入观测值，不保证是前一自然日。', 'source': source_ref}]
    audit_time = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    lineage = transformation_lineage(levels, calculated)
    manifest = {'audit_date': audit_time, 'all_columns_match': all(checks.values()), 'column_checks': checks, 'source_rows': len(levels), 'output_rows': len(calculated), 'datasets': {}}
    for raw, kind, knowledge in [(source.read_bytes(), 'levels', original), (target.read_bytes(), 'stationary', derived), (calculated.to_csv(index=False).encode('utf-8'), 'stationary', derived)]:
        manifest['datasets'][digest(raw)] = {'kind': kind, 'name': '黄金TIPS论文案例：' + kind, 'risks': RISKS, 'knowledge': knowledge, 'verified_transform': all(checks.values()), 'data_status': 'legacy_reproduction_not_release_verified', 'lineage': {'parent_sha256': digest(source.read_bytes()) if kind == 'stationary' else None, 'recipe': 'thesis_v1' if kind == 'stationary' else 'original_cleaned_levels', 'release_dates_verified': False}}
    audit = {'audit_time': audit_time, 'source_sha256': digest(source.read_bytes()), 'existing_sha256': digest(target.read_bytes()), 'generated_sha256': digest(calculated.to_csv(index=False).encode('utf-8')), 'column_checks': checks, 'lineage': lineage, 'unresolved_risks': RISKS, 'corrected_dataset_created': False}
    audit_dir = ROOT / 'outputs/case_studies/gold_tips_thesis/audits'
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / ('audit_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with audit_path.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    manifest['audit_record'] = str(audit_path.relative_to(ROOT))
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'datasets'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

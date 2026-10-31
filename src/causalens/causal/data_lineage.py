"""Auditable date alignment; never infer release dates from observation periods."""
from __future__ import annotations

import numpy as np
import pandas as pd


def release_aligned_values(observation_dates, releases):
    """Use documented available_at timestamps, not monthly period labels.

    One row per publication; callers must resolve revisions/vintages upstream.
    Same-time availability is allowed, so daily dates must encode the intended cutoff.
    """
    required = {'period', 'available_at', 'value', 'source'}
    if not required.issubset(releases.columns):
        raise ValueError('必须提供统计期、实际可获得时间、数值及来源依据。')
    data = releases.copy()
    for column in ['period', 'available_at']:
        data[column] = pd.to_datetime(data[column], errors='coerce')
    times = pd.to_datetime(pd.Series(observation_dates), errors='coerce')
    if times.isna().any() or data[['period', 'available_at']].isna().any().any():
        raise ValueError('存在无效时间。')
    if data['available_at'].duplicated().any() or data['period'].duplicated().any():
        raise ValueError('存在重复发布时间或修订版本，请先明确实时版本策略。')
    if (data['available_at'] < data['period']).any():
        raise ValueError('可获得时间早于统计期，请核对证据。')
    if data['source'].isna().any() or data['source'].astype(str).str.strip().eq('').any():
        raise ValueError('每条发布时间都必须有来源。')
    values = pd.to_numeric(data['value'], errors='coerce')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('指标值必须为有限数值。')
    data['value'] = values
    left = pd.DataFrame({'observation_at': times, '_order': range(len(times))})
    result = pd.merge_asof(left.sort_values('observation_at'), data.sort_values('available_at'), left_on='observation_at', right_on='available_at', direction='backward')
    return result.sort_values('_order').drop(columns='_order').reset_index(drop=True)


def transformation_lineage(source, output):
    before = pd.to_datetime(source['date'], errors='raise')
    after = pd.to_datetime(output['date'], errors='raise')
    missing = before[~before.isin(after)]
    gaps = before.diff().dt.days
    return {
        'source_rows': len(source), 'output_rows': len(output),
        'removed_observations': [
            {'date': value.isoformat(), 'reason': '差分及一阶滞后导致开头缺失；不是异常值筛除'}
            for value in missing
        ],
        'calendar_gaps_over_one_day': [
            {'previous': before.iloc[i-1].isoformat(), 'current': before.iloc[i].isoformat(), 'days': int(gaps.iloc[i])}
            for i in range(1, len(before)) if gaps.iloc[i] > 1
        ],
        'gap_interpretation': '包括周末、假期或已删除观测；没有交易日历及清洗前数据，不能将所有间隔认定为缺失。',
        'upstream_deleted_rows': '未知：当前输入已完成清洗，无法从幸存样本恢复上游删除记录。',
    }

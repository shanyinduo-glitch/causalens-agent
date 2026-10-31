import pandas as pd
import pytest
from causalens.causal.data_lineage import release_aligned_values, transformation_lineage


def releases():
    return pd.DataFrame({'period': ['2020-01-01', '2020-02-01'], 'available_at': ['2020-02-13', '2020-03-11'], 'value': [1., 2.], 'source': ['release record A', 'release record B']})


def test_no_future_information_and_original_order():
    result = release_aligned_values(['2020-03-12', '2020-02-12', '2020-02-13'], releases())
    assert result.value.iloc[0] == 2
    assert pd.isna(result.value.iloc[1])
    assert result.value.iloc[2] == 1


@pytest.mark.parametrize('column,value', [('source', ''), ('available_at', '2019-01-01'), ('value', float('inf')), ('period', 'invalid')])
def test_bad_release_evidence_rejected(column, value):
    data = releases()
    data.loc[0, column] = value
    with pytest.raises(ValueError): release_aligned_values(['2020-03-01'], data)


def test_vintages_not_silently_resolved():
    data = releases()
    data.loc[1, 'period'] = data.loc[0, 'period']
    with pytest.raises(ValueError): release_aligned_values(['2020-03-01'], data)


def test_lineage_does_not_invent_upstream_deletions():
    before = pd.DataFrame({'date': ['2020-01-03', '2020-01-06', '2020-01-07']})
    report = transformation_lineage(before, before.iloc[2:])
    assert len(report['removed_observations']) == 2
    assert report['calendar_gaps_over_one_day'][0]['days'] == 3
    assert report['upstream_deleted_rows'].startswith('未知')

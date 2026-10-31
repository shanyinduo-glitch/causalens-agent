from causalens.ui.analysis_identity import analysis_identity


def test_key_order_not_a_change():
    assert analysis_identity(b'a', 'guided', 'q', {'a': 1, 'b': 2}) == analysis_identity(b'a', 'guided', 'q', {'b': 2, 'a': 1})


def test_input_changes_invalidate_result():
    baseline = analysis_identity(b'a', 'guided', 'q', {'lag': 1})['signature']
    for args in [(b'b', 'guided', 'q', {'lag': 1}), (b'a', 'free', 'q', {'lag': 1}), (b'a', 'guided', 'new', {'lag': 1}), (b'a', 'guided', 'q', {'lag': 2})]:
        assert analysis_identity(*args)['signature'] != baseline

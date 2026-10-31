import json
from causalens.ui.model_errors import describe_model_error


def test_no_secret_in_diagnostics():
    error = RuntimeError('api_key=secret-private-text')
    error.status_code = 401
    result = describe_model_error(error)
    assert 'secret' not in str(result)
    assert result['http_status'] == 401


def test_local_failure_not_quota():
    assert '本地处理' in describe_model_error(TypeError('private'))['reason']
    assert 'JSON' in describe_model_error(json.JSONDecodeError('bad', '', 0))['reason']

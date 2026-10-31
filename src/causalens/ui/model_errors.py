"""Safe diagnostics: never display provider messages, request bodies or secrets."""
import json


def describe_model_error(exc):
    from causalens.agent.qwen_client import _is_free_quota_exhausted
    name = type(exc).__name__
    status = getattr(exc, 'status_code', None)
    status = status if isinstance(status, int) else None
    if _is_free_quota_exhausted(exc):
        reason = '服务明确报告免费额度耗尽。请检查当前模型额度，保持用完即停保护。'
    elif 'BudgetExceeded' in name:
        reason = '请求预算已用完，未继续轮换。请检查首选模型免费额度。'
    elif 'Timeout' in name:
        reason = '模型请求超时；不代表额度耗尽。'
    elif 'Connection' in name:
        reason = '无法连接模型服务；请检查网络和服务地址。'
    elif status == 401:
        reason = '服务拒绝身份认证；请检查启动网页的环境中是否设置有效密钥。'
    elif status == 403:
        reason = '服务拒绝访问，但未识别为明确的免费额度耗尽错误；请检查权限和模型可用性。'
    elif status == 429:
        reason = '服务限流或配额限制；不能仅凭此状态认定免费额度耗尽。'
    elif status is not None:
        reason = '模型服务返回错误；可能涉及参数、模型可用性或服务状态，请结合状态码排查。'
    elif isinstance(exc, json.JSONDecodeError):
        reason = '已进入回复解析，但内容不是合法JSON；换模型不一定能解决。'
    elif isinstance(exc, (TypeError, AttributeError, KeyError, IndexError)):
        reason = '本地处理回复时发生程序异常；不能归因于免费额度。'
    else:
        reason = '配置、回复格式或本地处理异常，需要进一步定位。'
    return {'category': name, 'http_status': status, 'reason': reason}

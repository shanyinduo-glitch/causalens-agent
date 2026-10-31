"""Bind a rendered result/approval to the exact submitted input."""
import hashlib
import json


def analysis_identity(raw, mode, question, contract):
    data_hash = hashlib.sha256(raw).hexdigest()
    payload = {'data_sha256': data_hash, 'mode': mode, 'question': question.strip(), 'contract': contract}
    signature = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
    return {'signature': signature, 'data_sha256': data_hash, 'mode': mode, 'contract': contract}

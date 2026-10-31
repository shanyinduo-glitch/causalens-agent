import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from causalens.ui.research_workspace import context_question, merge_updates, approve_snapshot, interpret_result


def test_context_is_bounded_and_does_not_carry_approval():
    turns = [{'question': str(i), 'state': {'final_answer': 'answer', 'approved': True}} for i in range(5)]
    prompt = context_question('当前问题', '背景', turns, [])
    assert '"user_question": "0"' not in prompt
    assert '"user_question": "4"' in prompt
    assert '"approved"' not in prompt
    assert '历史批准不适用于新任务' in prompt


def test_updates_do_not_mutate_checked_snapshot():
    state = {'tool_results': [{'tool_name': 'check'}], 'approved': False}
    result = merge_updates(state, {'tool_results': [{'tool_name': 'estimate'}]})
    assert len(result['tool_results']) == 2
    assert len(state['tool_results']) == 1


def test_approval_rejects_changed_data_and_nonpending_state():
    with pytest.raises(ValueError):
        approve_snapshot({'workflow_status': 'awaiting_approval'}, b'new', 'old')
    with pytest.raises(ValueError):
        approve_snapshot({}, b'data', hashlib.sha256(b'data').hexdigest())


def test_approval_executes_frozen_state_without_replanning(monkeypatch):
    import causalens.agent.nodes as nodes
    execute = Mock(return_value={'workflow_status': 'reporting', 'tool_results': [{'success': True}]})
    report = Mock(return_value={'workflow_status': 'completed', 'final_answer': '结果'})
    monkeypatch.setattr(nodes, 'causal_execution_node', execute)
    monkeypatch.setattr(nodes, 'reporter_node', report)
    state = {'workflow_status': 'awaiting_approval', 'causal_question': {'treatment': 't'}, 'approved': False}
    result = approve_snapshot(state, b'data', hashlib.sha256(b'data').hexdigest())
    assert result['workflow_status'] == 'completed'
    assert execute.call_args.args[0]['approved'] is True
    assert execute.call_args.args[0]['causal_question'] == {'treatment': 't'}
    assert state['approved'] is False


def test_interpretation_calls_model_once_and_no_tools(monkeypatch):
    import causalens.agent.qwen_client as client
    call = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='仅作探索'))]))
    monkeypatch.setattr(client, 'create_chat_completion', call)
    assert interpret_result('说明限制', {'tool_results': []}, []) == '仅作探索'
    call.assert_called_once()


def test_workspace_initial_page():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path('streamlit_app.py').resolve()), default_timeout=30).run()
    assert not app.exception
    assert app.radio[0].value == '对话探索工作台'
    assert any('研究资料' in h.value for h in app.header)
    app.radio[0].set_value('研究设计与人工配置').run()
    assert not app.exception
    assert any('选择使用方式' == w.label for w in app.radio)


def test_workspace_consent_and_conversation():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path('tests/fixtures/research_workspace_app.py').resolve()), default_timeout=30).run()
    assert not app.exception
    assert next(b for b in app.button if b.label == '发送并开始探索').disabled
    app.checkbox(key='workspace_consent').check().run()
    assert not next(b for b in app.button if b.label == '发送并开始探索').disabled
    next(w for w in app.text_area if w.label == '你的问题').set_value('比较 group 两组的 value')
    next(b for b in app.button if b.label == '发送并开始探索').click().run()
    assert not app.exception
    assert any('工具结果已返回' in item.value for item in app.markdown)
    assert any('statistical_analysis' in item.value for item in app.markdown)


def test_exact_user_exploration_request_does_not_route_to_causal():
    from causalens.agent.nodes import intent_router_node, _infer_required_tool_names
    question = '请检查 t_diff、y_ret 的缺失情况和描述统计，再绘制 t_diff 与 y_ret 的散点图。只描述数据中的关联，不执行因果估计。如果字段或工具条件不满足，请说明原因，不要替换变量。'
    routed = intent_router_node({'question': question, 'research_context': '因果分析 treatment outcome DML 处理变量'})
    assert routed['task_type'] == 'mixed_analysis'
    tools = _infer_required_tool_names(routed['task_type'], question)
    assert 'profile_dataset' in tools and 'create_chart' in tools
    assert 'run_causal_estimators' not in tools

"""Two explicit planning paths; model drafts never approve execution."""
import json
import streamlit as st
from causalens.ui.analysis_identity import analysis_identity
from causalens.ui.data_workbench import bound_metadata
from causalens.ui.causal_guidance import suggest_causal_roles_with_model
from causalens.ui.planning_evidence import LABELS, capabilities, retrieve_case_evidence
from causalens.ui.method_knowledge import method_cards, comparison_rows
from causalens.ui.planning_handoff import text_items


def render_ai_planning(raw, question, frame):
    st.session_state.pop('active_planning_snapshot', None)
    st.markdown('#### AI 研究方案草案')
    st.caption('AI 提出方案 → 本地结构校验 → 回答澄清问题 → 人工确认 → 原有规则检查与统计计算。')
    route = st.radio('规划依据', ['新数据探索', '论文背景增强（RAG）'], horizontal=True)
    metadata = bound_metadata(raw) if route == '论文背景增强（RAG）' else None
    rag_blocked = route == '论文背景增强（RAG）' and metadata is None
    if rag_blocked:
        st.warning('当前数据没有匹配的案例知识库。不能套用黄金论文，请切换新数据探索。')
    st.caption('两条路径共用有来源的方法知识卡片。新数据探索不使用案例知识；论文路径另检索版本绑定的数据说明与论文摘述，不是全文语义检索。')
    base = analysis_identity(raw, route, question, {})['signature']
    history = st.session_state.get('guided_ai_plan', {})
    previous = history if history.get('base') == base else {}
    clarification = st.text_area('补充研究背景 / 回答澄清问题', key='ai_background_' + base,
                                 help='例如：每行是一个保留的日度观测，研究未来5个观测期；尚不能证明外生性。填写不会自动调用模型。')
    case_evidence = retrieve_case_evidence(metadata, question + ' ' + clarification)
    evidence = case_evidence + method_cards()
    risks = metadata.get('risks', []) if metadata else []
    identity = analysis_identity(raw, route, question, {'evidence': evidence, 'clarification': clarification,
                                                        'risks': risks, 'capabilities': capabilities()})['signature']
    with st.expander('查看本次可发送的依据与方法能力'):
        if case_evidence:
            for item in case_evidence:
                st.markdown(f"**[{item['id']}] {item['source']} · {item['section']}**")
                st.write(item['text'])
        elif metadata:
            st.info('未检索到相关片段；模型不能声称有论文依据。可补充关键词。')
        else:
            st.write('本路径不发送论文与案例知识。')
        for risk in risks:
            st.write('同时发送的已知数据风险：', risk)
        st.json(capabilities())
    with st.expander('因果方法知识库：目标、假设、诊断与来源'):
        st.caption('人工整理的8张方法卡，两个路径共用。不从数据相关性推断政策、工具变量或外生性。')
        for card in method_cards():
            st.markdown(f"**[{card['id']}] {card['name']}**")
            st.write(card['text'])
            st.markdown(f"来源：[{card['source_title']}]({card['url']})")
    consent = st.checkbox(
        '允许发送问题、字段概况、补充背景及上方证据；我已在百炼确认所用模型开启“免费额度用完即停”',
        key='ai_consent_' + identity,
    )
    st.caption('不发送数据行或 API Key；请勿在问题中粘贴敏感数据。应用无法查询剩余额度，每次点击最多一次请求，不自动重试。')
    button_label = '根据补充信息更新方案（1 次模型请求）' if previous else '让 AI 提出研究方案（1 次模型请求）'
    if st.button(button_label, disabled=not consent or not question.strip() or rag_blocked):
        with st.spinner('正在请求模型并校验草案…'):
            try:
                result = suggest_causal_roles_with_model(
                    question, frame, {'risks': metadata.get('risks', [])} if metadata else None,
                    evidence=evidence, clarification=clarification,
                    previous_plan=previous.get('value', {}).get('raw_draft'),
                )
            except Exception as exc:
                st.session_state.pop('guided_ai_plan', None)
                from causalens.ui.model_errors import describe_model_error
                st.session_state.ai_plan_error = describe_model_error(exc)
            else:
                st.session_state.pop('ai_plan_error', None)
                revisions = previous.get('revisions', []) + ([previous['value']] if previous else [])
                st.session_state.guided_ai_plan = dict(base=base, identity=identity, value=result,
                    route=route, evidence=evidence, clarification=clarification, revisions=revisions[-5:])
    if st.session_state.get('ai_plan_error'):
        error = st.session_state.ai_plan_error
        st.error(error['reason'])
        st.caption(f"最近一次失败：{error['category']}；HTTP状态：{error['http_status'] or '无'}。未展示原始异常内容以保护密钥及请求数据。")
    saved = st.session_state.get('guided_ai_plan', {})
    if saved.get('identity') != identity:
        if saved:
            st.info('输入、路径或依据已改变，旧草案不再用于候选填充。请重新确认发送并更新方案。')
            if previous:
                with st.expander('上一版待回答的问题（仅供补充背景）'):
                    for item in previous['value'].get('clarifying_questions', []):
                        st.write(item)
        return None
    value = saved['value']
    review = value.get('validation', {'status': 'needs_revision', 'issues': ['草案缺少校验记录。']})
    if review['status'] == 'needs_revision':
        st.warning('已收到模型响应，但草案需修正；不会自动填入变量候选。')
        for issue in review['issues']:
            st.write('校验问题：', issue)
    else:
        st.info('草案通过有限结构检查，可供人工审核；不是因果识别通过。')
    st.write('问题理解：', value['research_question'])
    st.write('方法候选：', LABELS.get(value['design_candidate'], LABELS['unknown']))
    st.write('AI 选择依据（尚需核实）：')
    for item in text_items(value['rationale']):
        st.write('• ' + item)
        if any(term in item for term in ('最适合', '完美对齐', '完全契合')):
            st.warning('该措辞强于现有证据：仅能视为候选理由，不是已经完成方法优选。')
    rows = comparison_rows(value.get('method_options', []))
    if rows:
        st.markdown('##### 多候选方法比较')
        st.table(rows)
        st.caption('实现状态由本地注册表给出，不采信模型的执行承诺。LP、连续DML和长期关系可能回答不同问题，不可直接比较显著性来选优。')
        st.info('请结合目标与待核实依据，在下方研究设计环节确认方向；未实现的方法只能作为研究计划，不能在本页执行。')
    st.table([{'角色': label, '字段候选': '、'.join(value.get(key, [])) or '待澄清'} for key, label in
              [('treatment', '处理变量'), ('outcome', '结果变量'), ('time', '时间字段'), ('adjustment', '控制变量')]])
    st.caption('引用编号：' + ('、'.join(str(x) for x in value.get('evidence_ids', [])) or '无') + '。编号有效不代表引用内容支持该结论，请对照上方原文片段。')
    with st.expander('其他方法的限制（AI 说明，非验证结论）'):
        for item in value['alternatives']:
            st.write(item)
    for item in value['clarifying_questions']:
        st.info('需要确认：' + item)
    for item in text_items(value['warning']):
        st.warning(item)
    st.caption('请在上方补充框回答问题，明确点击更新才会再次请求。下方研究设计仍由你确认；自由文本的正确性不能由结构检查保证。')
    with st.expander('查看 AI 调用、原始草案及修订记录'):
        st.json(saved)
        st.download_button('下载 AI 草案记录', json.dumps(saved, ensure_ascii=False, indent=2),
                           file_name='ai_research_plan.json', mime='application/json')
    st.session_state.active_planning_snapshot = saved
    return value if review['status'] == 'candidate_only' else None

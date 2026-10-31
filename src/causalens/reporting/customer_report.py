"""Readable deliverable separated from the complete technical audit."""
from causalens.reporting.plain_summary import summarize_time_series


def customer_report(state):
    summary = summarize_time_series(state)
    question = state.get('question') or '见本次确认的研究目标'
    if not summary:
        answer = state.get('final_answer')
        summary = (('## 研究助手解读\n\n' + str(answer) +
                    '\n\n以上为助手基于本轮记录的解读，数值以技术记录中的工具输出为准；统计差异不自动构成因果效应。')
                   if answer else
                   '本次未生成可用的客户摘要。请查看技术记录中的执行状态与检查原因；不能据此判断存在或不存在影响。')
    summary = summary.replace('原始系数见技术详情', '完整计算依据见单独的技术记录')
    return '# CausaLens 分析简报\n\n## 研究目标\n\n' + str(question) + '\n\n' + summary

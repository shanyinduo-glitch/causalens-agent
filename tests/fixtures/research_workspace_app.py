from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import streamlit as st
from causalens.ui.research_workspace import render_workspace

class Upload(BytesIO):
    name = 'test.csv'

raw = b'group,value\n0,1\n1,2\n'
upload = Upload(raw)

def run(*args, **kwargs):
    assert kwargs['approved'] is False
    assert args[1] == '比较 group 两组的 value'
    assert 'research_context' in kwargs
    return {'workflow_status': 'completed', 'final_answer': '工具结果已返回',
            'tool_results': [{'tool_name': 'statistical_analysis', 'success': True, 'result': {'mean': 1.5}}]}

with patch.object(st, 'file_uploader', return_value=upload), \
     patch('causalens.ui.research_workspace.run_causalens_graph', run):
    render_workspace(Path.cwd(), lambda _: Path('fixture.csv'),
                     lambda _: pd.read_csv(BytesIO(raw)), lambda _: [], lambda _: None)

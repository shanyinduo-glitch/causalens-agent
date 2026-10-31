from io import BytesIO
import streamlit as st
from causalens.ui.data_workbench import select_analysis_version, digest

source = BytesIO(b'date,t,y\n2020-01-01,1,2\n')
source.name = 'source.csv'
st.session_state['prepared_' + digest(source.getvalue())] = (b'date,t_diff,y_ret\n2020-01-02,0.1,0.02\n', {})
active = select_analysis_version(source)
st.session_state['selected_bytes'] = active.getvalue()
st.session_state['selected_name'] = active.name

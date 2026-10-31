import pandas as pd
import streamlit as st
from causalens.ui.ai_planning import render_ai_planning

question = st.text_input('研究问题', value='利率对黄金的影响')
render_ai_planning(b't,y\n1,2', question, pd.DataFrame({'t': [1, 2], 'y': [2, 3]}))

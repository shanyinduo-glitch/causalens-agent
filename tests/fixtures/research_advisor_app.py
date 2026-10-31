import pandas as pd
import streamlit as st
from causalens.ui.research_advisor import render_research_advisor

import numpy as np
rng = np.random.default_rng(42)
frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=100),
                      't': rng.normal(size=100), 'y': rng.normal(size=100)})
labels = {'横截面': 'cross_sectional', '局部投影': 'time_series'}
result = render_research_advisor(frame, 't', 'y', labels)
st.session_state['test_result'] = result

import pandas as pd
import streamlit as st
from causalens.ui.data_workbench import ROOT
from causalens.ui.control_selection import render_control_selection

path = ROOT / 'data/private/gold_tips_thesis/processed/gold_tips_daily_stationary.csv'
raw = path.read_bytes()
columns = list(pd.read_csv(path).columns)
st.session_state['control_result'] = render_control_selection(raw, columns, {'date', 't_diff', 'y_ret'}, 't_diff', 'y_ret', 'time_series')

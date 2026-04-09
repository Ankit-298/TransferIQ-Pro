import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="TransferIQ Pro - AI Valuation", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #060810; }
    .stMetric { background-color: #0c1120; padding: 15px; border-radius: 10px; border: 1px solid #1e2848; }
    div[data-testid="stSidebar"] { background-color: #080c18; border-right: 1px solid #1e2848; }
    .stButton>button { width: 100%; border-radius: 8px; background: linear-gradient(135deg, #4d87ff, #2d67e8); color: white; border: none; font-weight: bold; height: 3em; }
    h1, h2, h3 { color: #edf2ff; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_assets():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_USE_LEGACY_KERAS'] = '0'
    
    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model('transferiq_model.json')
    
    lstm_m = tf.keras.models.load_model('transferiq_lstm.keras')
    return xgb_m, lstm_m

xgb_model, lstm_model = load_assets()

# --- PREDICTION LOGIC ---
def predict_single(performance, injury, sentiment, age, contract_years=3, position="MID", name="Unknown"):
    xgb_features = np.array([[performance / 10, injury, (sentiment + 1) / 2, age / 40]])
    raw = xgb_model.predict(xgb_features)[0]
    base_value = abs(raw) * 65000000

    pos_map = {"FWD": 1.3, "MID": 1.1, "DEF": 0.9, "GK": 0.7}
    contract_multiplier = 1.0 + (contract_years - 2) * 0.15 
    value = base_value * pos_map.get(position, 1.0) * contract_multiplier

    seq = []
    for t in range(3, 0, -1):
        past = value / ((1.05) ** t)
        seq.append([past / 1e8, (age - t) / 40, performance / 10, injury])

    lstm_input = np.array([seq])
    lstm_val = max(abs(lstm_model.predict(lstm_input, verbose=0)[0][0]) * 1e8, 1000000)
    trend_mult = 1.15 if age <= 24 else (0.85 if age >= 30 else 1.02)
    forecast = [float(value), float(lstm_val * trend_mult), float(lstm_val * (trend_mult ** 2)), float(lstm_val * (trend_mult ** 3))]

    pct_change = ((forecast[-1] - value) / value) * 100 if value > 0 else 0
    risk_score = (injury * 0.6 + (age / 40) * 0.2 + (5 - contract_years) * 0.05)
    
    return {
        "name": name, "value": value, "forecast": forecast, "pct": pct_change,
        "risk": "Low 🟢" if risk_score < 0.3 else "Medium 🟡" if risk_score < 0.6 else "High 🔴",
        "perf": performance, "inj": injury, "sent": sentiment
    }

# --- UI ---
st.title("⚽ TransferIQ Pro")
st.markdown("### AI-Powered Football Valuation Engine")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("🎯 Attributes")
    name = st.text_input("Player Name", "Wonderkid")
    age = st.slider("Age", 16, 40, 22)
    perf = st.slider("Performance (0-10)", 0.0, 10.0, 8.0)
    inj = st.slider("Injury Risk", 0.0, 1.0, 0.1)
    sent = st.slider("Sentiment", -1.0, 1.0, 0.5)
    pos = st.selectbox("Position", ["FWD", "MID", "DEF", "GK"])
    
    if st.button("Predict Valuation"):
        res = predict_single(perf, inj, sent, age, 3, pos, name)
        st.session_state.result = res

if 'result' in st.session_state:
    res = st.session_state.result
    with col2:
        st.subheader("💎 AI Analysis")
        m1, m2, m3 = st.columns(3)
        m1.metric("Market Value", f"€{res['value']/1e6:.1f}M")
        m2.metric("3yr Projection", f"€{res['forecast'][-1]/1e6:.1f}M", f"{res['pct']:.1f}%")
        m3.metric("Risk", res['risk'])
        
        fig = px.line(x=['Current', 'Year 1', 'Year 2', 'Year 3'], y=[v/1e6 for v in res['forecast']], 
                     title="Price Trajectory (€M)", markers=True)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

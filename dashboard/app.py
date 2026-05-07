import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import sys
import os
import yfinance as yf

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="QUANTWHALE", layout="wide", page_icon="🐳")

# Custom CSS for Premium Design Aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Space+Grotesk:wght@400;700&display=swap');

/* Global Background Gradient & Typography */
.stApp {
    background: radial-gradient(circle at top right, #0a1128 0%, #000000 70%);
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Sidebar Glassmorphism */
[data-testid="stSidebar"] {
    background-color: rgba(10, 17, 40, 0.4) !important;
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(0, 255, 204, 0.15);
}

/* Metric Cards Customization */
div[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(0, 255, 204, 0.2);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 255, 204, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 15px rgba(0, 255, 204, 0.15);
}
[data-testid="stMetricValue"] {
    color: #00ffcc !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #a0aec0 !important;
    font-weight: 600;
}

/* Title Gradient */
.title-gradient {
    background: -webkit-linear-gradient(45deg, #00ffcc, #3182ce);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    text-align: center;
    padding-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #00ffcc;
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 30px;
    opacity: 0.9;
    text-shadow: 0px 0px 8px rgba(0, 255, 204, 0.4);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-gradient'>🐳 QUANTWHALE.ai</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>trade with technology</div>", unsafe_allow_html=True)

st.sidebar.header("Configuration")

from dotenv import load_dotenv, set_key
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_file = os.path.join(ROOT_DIR, ".env")
load_dotenv(env_file)
saved_assets = os.getenv("TRACKED_ASSETS", "AAPL,MSFT,BTC-USD")

tickers = st.sidebar.text_input("Assets (comma separated)", saved_assets)
ticker_list = [t.strip() for t in tickers.split(',')]

# Automatically sync preferred assets to the engine config file explicitly
set_key(env_file, "TRACKED_ASSETS", tickers)

st.sidebar.markdown("---")
st.sidebar.subheader("MetaTrader 5 Connector")
st.sidebar.markdown("Securely connect your MT5 account for live execution.")
mt5_login = st.sidebar.text_input("Login ID")
mt5_pass = st.sidebar.text_input("Password", type="password")
mt5_server = st.sidebar.text_input("Server", "MetaQuotes")
if st.sidebar.button("Connect MT5"):
    set_key(env_file, "MT5_LOGIN", mt5_login)
    set_key(env_file, "MT5_PASS", mt5_pass)
    set_key(env_file, "MT5_SERVER", mt5_server)
    st.sidebar.success("MT5 Credentials Cached! Live Trader will route orders natively.")

st.sidebar.markdown("---")
st.sidebar.text("RL Agent: PPO Active")
st.sidebar.text("Sentiment Analysis: FinBERT")
st.sidebar.text("Blockchain Logging: Enabled (Mock)")

# Live Data Fetcher
def load_live_data():
    portfolio_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "live_portfolio.csv")
    trades_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "live_trades.csv")
    holdings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "live_holdings.csv")
    metrics_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "live_metrics.csv")
    
    port_df = pd.DataFrame(columns=["Timestamp", "Portfolio Value", "Initial Balance"])
    trades_df = pd.DataFrame(columns=["Exit Time", "Action", "Ticker", "Amount", "Entry Price ($)", "Exit Price ($)", "Profit ($)", "Confidence (%)"])
    holdings_df = pd.DataFrame(columns=["Asset", "Weight"])
    metrics_df = pd.DataFrame(columns=["Metric", "Value"])
    
    if os.path.exists(portfolio_file):
        try:
            port_df = pd.read_csv(portfolio_file)
        except: pass
        
    if os.path.exists(trades_file):
        try:
            trades_df = pd.read_csv(trades_file)
        except: pass
        
    if os.path.exists(holdings_file):
        try:
            holdings_df = pd.read_csv(holdings_file)
        except: pass
        
    if os.path.exists(metrics_file):
        try:
            metrics_df = pd.read_csv(metrics_file)
        except: pass
        
    return port_df, trades_df, holdings_df, metrics_df

port_df, trades_df, holdings_df, metrics_df = load_live_data()

@st.cache_data(ttl=60)
def fetch_recent_prices(t_list):
    try:
        data = yf.download(t_list, period='3d', interval='1m', progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    return data['Close'].dropna(how='all')
            else:
                if 'Close' in data.columns:
                    df_out = data[['Close']].dropna(how='all')
                    df_out.columns = t_list
                    return df_out
    except Exception:
        pass
    return pd.DataFrame(columns=t_list)

df = fetch_recent_prices(ticker_list)

st.subheader("Market & Portfolio Performance")
tab1, tab2 = st.tabs(["Live Asset Prices", "Portfolio Cumulative Return"])

with tab1:
    fig_prices = go.Figure()
    for col in df.columns:
        fig_prices.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig_prices.update_layout(height=400, xaxis_title="Date", yaxis_title="Price ($)", template="plotly_dark")
    st.plotly_chart(fig_prices, use_container_width=True)

with tab2:
    if not port_df.empty and len(port_df) > 1:
        initial_balance = float(port_df['Initial Balance'].iloc[0])
        portfolio_return = port_df['Portfolio Value'].astype(float)
        
        # Calculate Percentage Change from Initial Balance strictly matching the user requirement
        pct_change = ((portfolio_return - initial_balance) / initial_balance) * 100
        
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=pd.to_datetime(port_df['Timestamp']), y=pct_change, 
            mode='lines', name='AI Strategy % Return', 
            line=dict(color='#00ffcc', width=3)
        ))
        

        
        fig_port.update_layout(
            height=400, 
            xaxis_title="Live Execution Time", 
            yaxis_title="Percentage Change (%)", 
            title="Portfolio % Change from Initial Balance",
            template="plotly_dark"
        )
        st.plotly_chart(fig_port, use_container_width=True)
    else:
        st.info("Waiting for Live Trader to start logging portfolio data... Use `python main.py live` inside terminal to launch autonomous loop.")

st.divider()

col1, col2, col3, col4 = st.columns(4)

# Dynamic Metric 1: FinBERT Sentiment
sentiment_val = metrics_df.loc[metrics_df['Metric']=='Sentiment', 'Value'].values if not metrics_df.empty else [0.0]
finbert_score = float(sentiment_val[0]) if len(sentiment_val) > 0 else 0.0

if finbert_score > 0:
    sentiment_label = "Bullish"
    s_color = "normal"
elif finbert_score < 0:
    sentiment_label = "-Bearish"
    s_color = "normal"
else:
    sentiment_label = "Neutral"
    s_color = "off"

# Dynamic Metric 2: PPO Expected Return
if not port_df.empty and len(port_df) > 1:
    initial = float(port_df['Initial Balance'].iloc[0])
    current = float(port_df['Portfolio Value'].iloc[-1])
    ppo_return = ((current - initial) / initial) * 100
    ppo_label = f"{ppo_return:+.2f}%"
else:
    ppo_label = "0.00%"

# Dynamic Metric 3: HFT Order Flow
total_trades = len(trades_df)

# Dynamic Metric 4: Strategy Accuracy (Win Rate on Squared Off Trades)
closed_trades = trades_df[trades_df['Action'] == 'SELL (Square Off)']
if len(closed_trades) > 0:
    win_rate = (len(closed_trades[closed_trades['Profit ($)'].astype(float) > 0]) / len(closed_trades)) * 100
    acc_label = f"{win_rate:.1f}%"
else:
    acc_label = "0.0%"

col1.metric("Live FinBERT Sentiment", f"{finbert_score:+.2f}", sentiment_label, delta_color=s_color)
col2.metric("Current Portfolio Return", ppo_label, "Live Accumulation")
col3.metric("Live Order Flow", f"{total_trades} ops", "Total Executed")
col4.metric("Strategy Win Rate", acc_label, f"across {len(closed_trades)} closures")

st.divider()

col_signals, col_pie, col_history = st.columns([1, 1, 2])

with col_signals:
    st.subheader("HFT Signal Weights")
    st.markdown("Relative contribution of each model to trade decisions.")
    
    w_ml = float(metrics_df.loc[metrics_df['Metric']=='Weight_ML', 'Value'].values[0]) if 'Weight_ML' in metrics_df['Metric'].values else 0.33
    w_dl = float(metrics_df.loc[metrics_df['Metric']=='Weight_DL', 'Value'].values[0]) if 'Weight_DL' in metrics_df['Metric'].values else 0.33
    w_sen = float(metrics_df.loc[metrics_df['Metric']=='Weight_Sentiment', 'Value'].values[0]) if 'Weight_Sentiment' in metrics_df['Metric'].values else 0.34
    
    weights_df = pd.DataFrame({
        "Engine": ["ML (XGBoost)", "DL (LSTM)", "Sentiment (FinBERT)"],
        "Weight": [w_ml, w_dl, w_sen]
    })
    fig_bar = go.Figure(data=[go.Bar(
        x=weights_df['Engine'], 
        y=weights_df['Weight'],
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
    )])
    fig_bar.update_layout(template="plotly_dark", height=300, yaxis_title="Signal Weight")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_pie:
    st.subheader("RL Portfolio Allocation")
    st.markdown("Current agent weights.")
    if not holdings_df.empty:
        fig_pie = go.Figure(data=[go.Pie(labels=holdings_df['Asset'], values=holdings_df['Weight'], hole=0.3)])
        fig_pie.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Awaiting live allocations from autonomous backend...")

with col_history:
    st.subheader("Live Trade Logs")
    st.markdown("Verified stream of execution.")
    if not trades_df.empty:
        st.dataframe(trades_df.tail(15), use_container_width=True)
    else:
        st.info("No trades executed yet. Awaiting RL algorithm thresholds in live broker integration.")

import time
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-Refresh Dashboard (10s)", value=True)
if auto_refresh:
    time.sleep(10)
    st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import urllib.request
import xml.etree.ElementTree as ET
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Safely handle NLTK download for the cloud environment
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# --- 1. SETUP PAGE DESIGN & LIVE HEARTBEAT ---
st.set_page_config(page_title="AlgoStax Premium Engine", layout="wide", page_icon="🤖")

# Auto-refresh every 30 seconds completely on its own
st_autorefresh(interval=30000, key="algo_heartbeat")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
        margin-bottom: 15px;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        margin: 4px;
    }
    .pill-bearish { background-color: #7f1d1d; color: #f87171; border: 1px solid #b91c1c; }
    .pill-bullish { background-color: #14532d; color: #4ade80; border: 1px solid #15803d; }
    .pill-neutral { background-color: #374151; color: #d1d5db; border: 1px solid #4b5563; }
    h1, h2, h3 { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)


# --- 2. NATIVE MATHEMATICAL INDICATORS ENGINE ---
def calculate_native_indicators(df):
    """
    Replaces pandas_ta entirely using vectorized pandas calculations.
    Maintains exact column name pairings expected by the Machine Learning Engine.
    """
    df = df.copy()
    
    # 20 & 50 period Exponential Moving Averages (EMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 14-period Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Safe division buffer
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD 12, 26)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema12 - ema26
    
    # 14-period Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
    low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATRr_14'] = true_range.ewm(alpha=1/14, adjust=False).mean()
    
    return df


@st.cache_data(ttl=15)
def fetch_live_market_and_indicators(ticker_symbol, interval_choice):
    period_mapping = {
        "1m": "5d",
        "15m": "30d",
        "30m": "30d",
        "1h": "60d"
    }
    chosen_period = period_mapping.get(interval_choice, "30d")

    try:
        df = yf.download(ticker_symbol, period=chosen_period, interval=interval_choice, progress=False)
        if df.empty: return None, None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # Run native math transformations
        df = calculate_native_indicators(df)
        df.dropna(inplace=True)

        chart_df = df.tail(100).copy()
        return df, chart_df
    except:
        return None, None


def calculate_fibonacci_levels(chart_df):
    swing_high = chart_df['High'].max()
    swing_low = chart_df['Low'].min()
    diff = swing_high - swing_low

    levels = {
        "0.0% (High)": swing_high,
        "23.6%": swing_high - (0.236 * diff),
        "38.2%": swing_high - (0.382 * diff),
        "50.0%": swing_high - (0.500 * diff),
        "61.8%": swing_high - (0.618 * diff),
        "78.6%": swing_high - (0.786 * diff),
        "100.0% (Low)": swing_low
    }
    return levels


def train_live_ai(df):
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df_clean = df.dropna().copy()
    X = df_clean[['EMA_20', 'EMA_50', 'RSI_14', 'MACD_12_26_9', 'ATRr_14']]
    y = df_clean['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)

    last_features = X.iloc[[-1]]
    probs = model.predict_proba(last_features)[0]
    last_close = df['Close'].iloc[-1]
    last_atr = df['ATRr_14'].iloc[-1]
    return probs, last_close, last_atr


def fetch_live_sentiment(search_target):
    analyzer = SentimentIntensityAnalyzer()
    query = search_target.replace(' ', '+') + '+stock'
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        xml_data = urllib.request.urlopen(req).read()
        root = ET.fromstring(xml_data)
    except:
        return 0, []

    total_score = 0
    articles = []
    items = root.findall('.//item')

    for item in items[:8]:
        headline = item.find('title').text
        score = analyzer.polarity_scores(headline)['compound']
        total_score += score
        articles.append((headline, score))

    return (total_score / len(articles) if articles else 0), articles


# --- 3. SIDEBAR CONFIGURATION (DUAL DROPDOWNS) ---
st.sidebar.header("🤖 AlgoStax Configurations")

TICKER_LIST = [
    "^NSEI", "^BSESN", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
    "INFY.NS", "ICICIBANK.NS", "TATAMOTORS.NS", "SBIN.NS",
    "AAPL", "TSLA", "NVDA", "BTC-USD"
]
TICKER = st.sidebar.selectbox("Select Ticker Symbol:", options=TICKER_LIST, index=0)

COMPANY_LIST = [
    "Nifty 50", "Sensex", "Reliance Industries", "Tata Consultancy Services",
    "HDFC Bank", "Infosys", "ICICI Bank", "Tata Motors",
    "State Bank of India", "Apple", "Tesla", "Nvidia", "Bitcoin"
]

default_company_index = 0
if TICKER in TICKER_LIST:
    default_company_index = TICKER_LIST.index(TICKER)
    if default_company_index >= len(COMPANY_LIST):
        default_company_index = 0

COMPANY_NAME = st.sidebar.selectbox("Select Company Search Name:", options=COMPANY_LIST, index=default_company_index)
INTERVAL = st.sidebar.selectbox("Select Chart Timeframe:", options=["1m", "15m", "30m", "1h"], index=1)

# --- 4. PROCESSING STREAM ---
df, chart_df = fetch_live_market_and_indicators(TICKER, INTERVAL)
sentiment_score, news_list = fetch_live_sentiment(COMPANY_NAME)

if df is not None and not df.empty:
    probs, last_close, last_atr = train_live_ai(df)
    fib_levels = calculate_fibonacci_levels(chart_df)

    if INTERVAL == "1h":
        time_labels = chart_df.index.strftime('%b %d, %H:%M')
    else:
        time_labels = chart_df.index.strftime('%H:%M (%d %b)')

    tech_points = probs[1] * 50
    sentiment_points = (sentiment_score + 1) * 25

    fused_up_prob = tech_points + sentiment_points
    fused_up_prob = max(0, min(100, fused_up_prob))

    tech_state = "BULLISH" if probs[1] > 0.55 else "BEARISH" if probs[1] < 0.45 else "NEUTRAL"
    sent_state = "BULLISH" if sentiment_score > 0.10 else "BEARISH" if sentiment_score < -0.10 else "NEUTRAL"

    if fused_up_prob > 60:
        fusion_decision = "BUY"
    elif fused_up_prob < 40:
        fusion_decision = "SELL"
    else:
        fusion_decision = "HOLD"

    # --- 5. AUTOMATED RISK MANAGEMENT CALCULATIONS ---
    if fusion_decision == "BUY":
        stop_loss = last_close - (1.5 * last_atr)
        take_profit = last_close + (3.0 * last_atr)
    elif fusion_decision == "SELL":
        stop_loss = last_close + (1.5 * last_atr)
        take_profit = last_close - (3.0 * last_atr)
    else:
        stop_loss = last_close - (1.5 * last_atr)
        take_profit = last_close + (3.0 * last_atr)

    # --- MAIN PREMIUM UI LAYOUT ---
    st.title("🤖 AlgoStax Trading Bot")

    pill_tech_html = f'<span class="status-pill pill-{tech_state.lower()}">Technical: {tech_state}</span>'
    pill_sent_html = f'<span class="status-pill pill-{sent_state.lower()}">News: {sent_state}</span>'
    pill_fuse_html = f'<span class="status-pill pill-{"bullish" if fusion_decision == "BUY" else "bearish" if fusion_decision == "SELL" else "neutral"}">Fusion: {fusion_decision}</span>'
    st.markdown(f"{pill_tech_html} {pill_sent_html} {pill_fuse_html}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(
            f"### Live Close ({INTERVAL})<br><span style='font-size:28px; font-weight:bold; color:#4ade80;'>₹{last_close:,.2f}</span>",
            unsafe_allow_html=True)

        gauge_color = "#22c55e" if fusion_decision == "BUY" else "#ef4444" if fusion_decision == "SELL" else "#9ca3af"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fused_up_prob,
            number={'suffix': "%", 'font': {'size': 36, 'color': '#ffffff'}},
            title={'text': f"{fusion_decision} CONFIDENCE", 'font': {'size': 16, 'color': '#9ca3af'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#30363d"},
                'bar': {'color': gauge_color},
                'bgcolor': "#161b22",
                'borderwidth': 2,
                'bordercolor': "#30363d",
                'steps': [
                    {'range': [0, 40], 'color': '#1f2937'},
                    {'range': [40, 60], 'color': '#111827'},
                    {'range': [60, 100], 'color': '#1f2937'}
                ],
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=240, margin=dict(t=10, b=10, l=10, r=10),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
            <div style="text-align: left; padding: 10px; background-color: #0e1117; border-radius: 8px; border: 1px solid #30363d;">
                <p style="margin: 4px 0; color: #4ade80;">🎯 <b>Take Profit:</b> ₹{take_profit:,.2f}</p>
                <p style="margin: 4px 0; color: #f87171;">🛑 <b>Stop Loss:</b> ₹{stop_loss:,.2f}</p>
                <p style="margin: 4px 0; color: #9ca3af; font-size: 12px;">📊 Current ATR Volatility: {last_atr:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        fig_chart = go.Figure(data=[go.Candlestick(
            x=time_labels,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name=f"{INTERVAL} Candles"
        )])

        colors = ["#ef4444", "#3b82f6", "#10b981", "#eab308", "#a855f7", "#6366f1", "#f43f5e"]
        for (label, val), color in zip(fib_levels.items(), colors):
            fig_chart.add_hline(y=val, line_dash="dot", line_color=color, line_width=1)
            fig_chart.add_annotation(
                x=1.01, y=val, xref="paper", yref="y",
                text=f"{label} (₹{val:,.1f})", showarrow=False,
                xanchor="left", yanchor="middle", font=dict(color=color, size=10)
            )

        fig_chart.add_hline(y=take_profit, line_dash="solid", line_color="#22c55e", line_width=2)
        fig_chart.add_annotation(
            x=0.01, y=take_profit, xref="paper", yref="y",
            text="🎯 TARGET (TP)", showarrow=False,
            xanchor="left", yanchor="bottom", font=dict(color="#22c55e", size=11, weight="bold")
        )

        fig_chart.add_hline(y=stop_loss, line_dash="solid", line_color="#ef4444", line_width=2)
        fig_chart.add_annotation(
            x=0.01, y=stop_loss, xref="paper", yref="y",
            text="🛑 STOP LOSS (SL)", showarrow=False,
            xanchor="left", yanchor="bottom", font=dict(color="#ef4444", size=11, weight="bold")
        )

        fig_chart.update_layout(
            title=f"Live Stream + Fibonacci + Risk Management ({TICKER})",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=390,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=20, l=10, r=160)
        )
        st.plotly_chart(fig_chart, use_container_width=True)

    st.markdown("### 📰 Neural Network Headline Analysis")
    for headline, score in news_list[:4]:
        color_class = "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
        score_symbol = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
        st.markdown(f"""
            <div style="background-color: #161b22; padding: 12px 18px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #30363d;">
                <span style="font-family: monospace;" class="status-pill pill-{color_class}">{score_symbol} {score:.2f}</span> 
                <span style="color: #e2e8f0; font-size: 15px; margin-left: 10px;">{headline}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: #30363d;'>", unsafe_allow_html=True)
    with st.expander("ℹ️ Beginner's Guide: How to understand this data", expanded=False):
        st.markdown("""
        ### 🎯 What are Take Profit and Stop Loss?
        Think of these as your automatic safety nets so you don't have to stare at the screen all day.
        * **Take Profit (Target):** This is your goal. If the stock hits this price, you automatically cash out and lock in your winnings.
        * **Stop Loss (Safety Brake):** This is your emergency exit. If the market suddenly crashes against you, the system knows to cut the cord immediately so a small mistake doesn't wipe out your account.

        ### 🕳️ Why is there "Empty Space" on the chart?
        Stock markets need sleep! The Indian Stock Market (NSE) is only open from 9:15 AM to 3:30 PM. The huge blank gaps you see on the timeline represent nights, weekends, and holidays. Time kept moving forward, but nobody was allowed to trade, creating a gap on the calendar.

        ### ⚖️ Why does the price here look slightly different than Google/Zerodha after 3:30 PM?
        If you check your phone right after the market closes, you will see the **LTP (Last Traded Price)**—which is just the final, single, random transaction of the day. 
        However, to prevent billionaires from manipulating the final second of the market, the exchange calculates the **Official Closing Price** by taking the mathematical average of *all* trades in the last 30 minutes. Your dashboard uses this official, highly accurate institutional average, not the chaotic final tick!
        """)

else:
    st.error(f"Could not download market data stream for '{TICKER}'.")

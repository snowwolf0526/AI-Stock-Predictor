import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import feedparser
from snownlp import SnowNLP
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
import json
import re
import sqlite3
import os

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI 股市預測系統", layout="wide")

# ==========================================
# 🗄️ 資料庫初始化 (System Architecture: SQLite)
# ==========================================
def init_db():
    """初始化 SQLite 資料庫，用於紀錄使用者的查詢與回測歷史"""
    conn = sqlite3.connect('quant_system.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS query_logs 
                 (timestamp TEXT, ticker TEXT, company TEXT, prediction TEXT, confidence REAL, strategy_roi REAL)''')
    conn.commit()
    conn.close()

def log_to_db(ticker, company, prediction, confidence, roi):
    """寫入查詢紀錄"""
    conn = sqlite3.connect('quant_system.db')
    c = conn.cursor()
    pred_str = "上漲" if prediction == 1 else "下跌"
    c.execute("INSERT INTO query_logs VALUES (?, ?, ?, ?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticker, company, pred_str, confidence, roi))
    conn.commit()
    conn.close()

init_db()

# ==========================================
# ⚡ 系統效能優化與智慧辨識區 (Caching)
# ==========================================
@st.cache_data(ttl=86400)
def get_company_name(clean_ticker):
    top_stocks = {
        '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電',
        '2382': '廣達', '2412': '中華電', '2881': '富邦金', '2882': '國泰金',
        '2603': '長榮', '2609': '陽明', '2615': '萬海', '3008': '大立光',
        '0050': '元大台灣50', '0056': '元大高股息', '00878': '國泰永續高股息'
    }
    if clean_ticker in top_stocks: return top_stocks[clean_ticker]
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        resp = genai.GenerativeModel('gemini-3.6-flash').generate_content(
            f"將台股代號 {clean_ticker} 轉換為公司簡稱。只回傳名稱，不要標點。"
        )
        name = resp.text.strip()
        if name and len(name) <= 15: return name
    except: pass
    return clean_ticker

@st.cache_data(ttl=3600)
def fetch_stock_and_macro_data(ticker):
    """抓取個股與總體經濟特徵 (大盤, 匯率, 費半)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    start_str, end_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    # 1. 抓取個股
    stock = yf.download(ticker, start=start_str, end=end_str, progress=False)
    if stock.empty: return pd.DataFrame()
    if isinstance(stock.columns, pd.MultiIndex): stock.columns = stock.columns.droplevel(1)
    stock.reset_index(inplace=True)
    df = stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 2. 抓取總體經濟指標 (TWII, SOX, USD/TWD)
    macro_symbols = {"^TWII": "TWII_Return", "^SOX": "SOX_Return", "USDTWD=X": "USDTWD_Return"}
    for sym, col_name in macro_symbols.items():
        macro = yf.download(sym, start=start_str, end=end_str, progress=False)
        if not macro.empty:
            if isinstance(macro.columns, pd.MultiIndex): macro.columns = macro.columns.droplevel(1)
            macro.reset_index(inplace=True)
            macro['Return'] = macro['Close'].pct_change()
            df = pd.merge(df, macro[['Date', 'Return']], on='Date', how='left')
            df.rename(columns={'Return': col_name}, inplace=True)
        else:
            df[col_name] = 0.0
            
    df.fillna(0, inplace=True)
    return df

@st.cache_data(ttl=3600)
def fetch_news_sentiment(keyword):
    url = f"https://news.google.com/rss/search?q={keyword}+when:3d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    feed = feedparser.parse(url)
    news_titles = [entry.title for entry in feed.entries[:5]]
    avg_sentiment, ai_reason = 0.5, "近期無相關財經新聞，模型以中立情緒計算。"
    
    if news_titles:
        try:
            import google.generativeai as genai
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            prompt = f"""你是一個台灣股市分析師。分析以下新聞標題情緒，只回傳JSON，格式：{{"score": 0.8, "reason": "利多..."}}
            新聞標題：{news_titles}"""
            raw_text = genai.GenerativeModel('gemini-3.6-flash').generate_content(prompt).text.strip()
            if raw_text.startswith("```"): raw_text = re.sub(r"^```(json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
            result = json.loads(raw_text)
            return float(result.get("score", 0.5)), result.get("reason", "中立")
        except: pass
    return avg_sentiment, ai_reason

def generate_ai_report(ticker, company_name, prediction, confidence, ai_reason, top_features, strategy_roi, market_roi):
    try:
        import google.generativeai as genai
        prompt = f"""你是頂級量化交易首席分析師。根據以下數據撰寫150字專業診斷報告。
        標的：{ticker} ({company_name})
        預測：{'上漲' if prediction == 1 else '下跌'} (信心：{confidence:.1f}%)
        消息：{ai_reason}
        關鍵特徵：{', '.join(top_features)}
        近半年報酬 (已扣手續費)：AI {strategy_roi:.2f}% vs 大盤 {market_roi:.2f}%
        給出投資與風險控管建議。"""
        return genai.GenerativeModel('gemini-3.6-flash').generate_content(prompt).text.strip()
    except: return "⚠️ AI 暫時離線。"

# ==========================================
# 🖥️ 主程式與 UI 顯示區
# ==========================================
st.title("📈 畢業專題：AI 股市預測系統 (業界實戰級完全體)")
st.markdown("本系統整合了 **總體經濟特徵 (費半/匯率)**、**多模型集成 (Ensemble)**、**真實摩擦成本回測** 與 **SQLite 查詢紀錄**。")

raw_ticker = st.text_input("🎯 輸入台股代號 (如: 2330, 2454, 0050)", value="2330", max_chars=8)
clean_ticker = raw_ticker.strip().upper()
pure_code = clean_ticker.split('.')[0] if clean_ticker.endswith(".TW") or clean_ticker.endswith(".TWO") else clean_ticker
ticker = f"{pure_code}.TW"
company_name = get_company_name(pure_code)
st.caption(f"📌 目標自動識別：**{pure_code} {company_name}**")

if st.button(f"🚀 啟動 {ticker} ({company_name}) 深度訓練與預測", type="primary"):
    with st.spinner('整合總體經濟數據與多模型 Ensemble 訓練中...'):
        try:
            df = fetch_stock_and_macro_data(ticker)
            if df.empty or len(df) < 300: st.error("❌ 歷史資料不足！"); st.stop()
                
            avg_sentiment, ai_reason = fetch_news_sentiment(company_name)
            df['Sentiment'] = avg_sentiment  
            
            # 技術指標與籌碼
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=10, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.obv(append=True)
            df.ta.mfi(length=14, append=True)
            
            df['Next_Close'] = df['Close'].shift(-1)
            df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
            
            # 加入總經特徵 (TWII, SOX, USDTWD)
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 
                        'TWII_Return', 'SOX_Return', 'USDTWD_Return',
                        'SMA_5', 'SMA_10', 'RSI_14', 'MACD_12_26_9', 'OBV', 'MFI_14']
                        
            train_df = df.dropna(subset=features + ['Target'])
            X_today = df.iloc[-1:][features]
            
            # 🧠 演算法進化：多模型集成 (Ensemble Learning)
            # 結合 XGBoost (梯度提升) 與 Random Forest (隨機森林)
            clf1 = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            clf2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            ensemble_model = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2)], voting='soft')
            
            ensemble_model.fit(train_df[features], train_df['Target'])
            prediction = ensemble_model.predict(X_today)[0]
            probability = ensemble_model.predict_proba(X_today)[0]
            
            st.success(f"✅ Ensemble 集成訓練完成！已融合技術面、籌碼面、新聞情緒與總經指標 (費半/匯率)。")
            
            # 顯示預測結果
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="預測方向", value="上漲 📈" if prediction == 1 else "下跌 📉", delta="模型看多" if prediction==1 else "-模型看空", delta_color="normal" if prediction==1 else "inverse")
            with col2:
                st.metric(label="聯合模型信心水準", value=f"{probability[prediction] * 100:.1f}%")

            # 📊 回測模組進化：加入真實摩擦成本 (交易手續費)
            backtest_days = 126
            strategy_roi, market_roi = 0, 0
            bt_test = pd.DataFrame()
            if len(train_df) > backtest_days * 1.5:
                bt_train, bt_test = train_df.iloc[:-backtest_days], train_df.iloc[-backtest_days:].copy()
                
                bt_model = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2)], voting='soft')
                bt_model.fit(bt_train[features], bt_train['Target'])
                
                bt_test['Prediction'] = bt_model.predict(bt_test[features])
                bt_test['Daily_Return'] = bt_test['Close'].pct_change().fillna(0)
                bt_test['Strategy_Return'] = bt_test['Prediction'].shift(1).fillna(0) * bt_test['Daily_Return']
                
                # 💸 計算摩擦成本 (每次換位產生 0.3% 耗損)
                bt_test['Signal_Change'] = bt_test['Prediction'].diff().fillna(0).abs()
                transaction_fee_rate = 0.003 
                bt_test['Strategy_Return'] = bt_test['Strategy_Return'] - (bt_test['Signal_Change'] * transaction_fee_rate)
                bt_test['Strategy_Return'] = bt_test['Strategy_Return'].clip(lower=-0.05) # 5% 停損
                
                bt_test['Cum_Market'] = (1 + bt_test['Daily_Return']).cumprod()
                bt_test['Cum_Strategy'] = (1 + bt_test['Strategy_Return']).cumprod()
                market_roi = (bt_test['Cum_Market'].iloc[-1] - 1) * 100
                strategy_roi = (bt_test['Cum_Strategy'].iloc[-1] - 1) * 100

            # 寫入 SQLite 資料庫
            log_to_db(ticker, company_name, prediction, probability[prediction], strategy_roi)

            # 提取 XGBoost 的特徵重要性 (VotingClassifier 無法直接給出)
            clf1.fit(train_df[features], train_df['Target'])
            importance_df = pd.DataFrame({'特徵': features, '重要性': clf1.feature_importances_}).sort_values(by='重要性', ascending=True)
            top_3 = importance_df['特徵'].iloc[-3:].tolist()

            # 🤖 AI 報告
            st.markdown("### 🤖 首席 AI 總體診斷報告")
            with st.spinner("撰寫報告中..."):
                st.info(generate_ai_report(ticker, company_name, prediction, probability[prediction] * 100, ai_reason, top_3, strategy_roi, market_roi))
                
            # 圖表
            st.markdown("---")
            fig = go.Figure(data=[go.Candlestick(x=df.tail(90)['Date'], open=df.tail(90)['Open'], high=df.tail(90)['High'], low=df.tail(90)['Low'], close=df.tail(90)['Close'], name='K線')])
            fig.update_layout(title=f'{company_name} ({ticker}) 近 90 日走勢', template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            fig_imp = go.Figure(go.Bar(x=importance_df['重要性'], y=importance_df['特徵'], orientation='h'))
            fig_imp.update_layout(title='XGBoost 核心特徵重要性分析', template='plotly_white', height=350)
            st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown("### 📊 AI 策略 vs 單純持有：近半年歷史回測 (含摩擦成本與停損)")
            if not bt_test.empty:
                col3, col4 = st.columns(2)
                with col3: st.metric(label="📈 AI 策略累積報酬 (已扣手續費)", value=f"{strategy_roi:.2f}%")
                with col4: st.metric(label="📉 單純買進持有 (Buy & Hold)", value=f"{market_roi:.2f}%")
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=bt_test['Date'], y=bt_test['Cum_Strategy'], line=dict(color='red', width=2.5), name='AI 策略'))
                fig_bt.add_trace(go.Scatter(x=bt_test['Date'], y=bt_test['Cum_Market'], line=dict(color='gray', width=1.5, dash='dash'), name='持有'))
                fig_bt.update_layout(template='plotly_white', height=400, hovermode='x unified')
                st.plotly_chart(fig_bt, use_container_width=True)
                
        except Exception as e: st.error(f"系統錯誤：{e}")

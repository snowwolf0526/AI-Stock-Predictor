import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
import feedparser
from snownlp import SnowNLP
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI 股市預測系統", layout="wide")

st.title("📈 畢業專題：AI 股市預測系統 (即時動態訓練版)")
st.markdown("本系統採用 **On-the-fly 即時訓練架構**，輸入任意台股代號後，系統將當場抓取歷史資料、建立專屬 XGBoost 模型並進行預測，完美解決不同股價量級的誤差問題。")
st.warning("⚠️ 免責聲明：本系統僅供學術專題展示使用，不構成任何投資建議。")

# === 1. 使用者輸入區 ===
st.markdown("### 🎯 請輸入預測標的")
col1, col2 = st.columns([1, 3])
with col1:
    raw_ticker = st.text_input("輸入台股代號 (如: 2330, 00679B)", value="2330", max_chars=6)
with col2:
    keyword = st.text_input("輸入公司名稱或關鍵字 (用於抓取新聞情緒)", value="台積電")
    
clean_ticker = raw_ticker.strip().upper()
if clean_ticker.endswith(".TW") or clean_ticker.endswith(".TWO"):
    ticker = clean_ticker
else:
    ticker = f"{clean_ticker}.TW"

# === 2. 核心運算區 (即時訓練與預測) ===
if st.button(f"🚀 啟動 {ticker} 即時訓練與預測", type="primary"):
    with st.spinner(f'正在抓取 {ticker} 過去兩年資料並訓練專屬模型中，請稍候約 5-8 秒...'):
        try:
            # [A] 抓取歷史股價 (2年)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            stock = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
            
            if stock.empty:
                st.error("❌ 找不到該股票資料，請確認代號是否正確。")
                st.stop()
                
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.droplevel(1)
            stock.reset_index(inplace=True)
            df = stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            if len(df) < 300:
                st.error(f"❌ {ticker} 上市時間過短，歷史資料不足 (需至少 300 天)，無法進行機器學習訓練！")
                st.stop()
                
            # [B] 🔥 抓取新聞情緒 (Gemini LLM 官方指定版 + SnowNLP 備用機制) 🔥
            url = f"https://news.google.com/rss/search?q={keyword}+when:3d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            feed = feedparser.parse(url)
            
            # 萃取前 5 則最新新聞標題
            news_titles = [entry.title for entry in feed.entries[:5]]
            avg_sentiment = 0.5  # 預設中立
            
            if news_titles:
                try:
                    # 優先嘗試使用 Gemini API 進行高階情緒分析
                    import google.generativeai as genai
                    api_key = st.secrets["GEMINI_API_KEY"]
                    genai.configure(api_key=api_key)
                    
                    # 聽從 Google 伺服器的建議，直接指定最新版模型
                    valid_model_name = 'gemini-3.6-flash'
                    model = genai.GenerativeModel(valid_model_name)
                    
                    prompt = f"你是一個專業的台灣股市分析師。請綜合分析以下新聞標題對該公司股價的情緒影響。請只回傳 0.0 到 1.0 之間的浮點數數字（0.0為極度看跌，1.0為極度看漲，0.5為中立），不要任何解釋。新聞標題：{news_titles}"
                    
                    response = model.generate_content(prompt)
                    avg_sentiment = float(response.text.strip())
                    
                    # 印出成功抓到的模型名稱
                    st.toast(f"✨ 成功使用 {valid_model_name} 進行新聞情緒分析！", icon="🧠")
                    
                except Exception as e:
                    # 萬一 API 沒設定好或失效，無縫切換回 SnowNLP 備用
                    st.toast(f"⚠️ Gemini 失敗，已自動切換回 SnowNLP。錯誤原因：{e}", icon="🔄")
                    sentiment_scores = []
                    for title in news_titles:
                        try:
                            sentiment_scores.append(SnowNLP(title).sentiments)
                        except:
                            pass
                    if sentiment_scores:
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            df['Sentiment'] = avg_sentiment  
            
            # [C] 計算技術指標
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=10, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            
            # [D] 定義預測目標 (Y) 與特徵 (X)
            df['Next_Close'] = df['Close'].shift(-1)
            df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'Sentiment', 'SMA_5', 'SMA_10', 'RSI_14', 
                        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
                        
            train_df = df.dropna(subset=features + ['Target'])
            latest_data = df.iloc[-1:]
            X_today = latest_data[features]
            
            # [E] 即時動態訓練 (On-the-fly)
            model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            model_xgb.fit(train_df[features], train_df['Target'])
            
            # [F] 進行預測
            prediction = model_xgb.predict(X_today)[0]
            probability = model_xgb.predict_proba(X_today)[0]
            
            st.success(f"✅ 專屬模型訓練完成！共使用 {len(train_df)} 筆歷史資料進行現場訓練。")
            
            # === 3. 畫面顯示區 ===
            st.markdown("### 🔮 明日趨勢預測結果")
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.metric(label="預測方向", value="上漲 📈", delta="模型看多")
                else:
                    st.metric(label="預測方向", value="下跌 📉", delta="-模型看空", delta_color="inverse")
            with col2:
                confidence = probability[prediction] * 100
                st.metric(label="模型信心水準", value=f"{confidence:.1f}%")
                
            # 畫 K 線圖
            plot_df = df.tail(90)
            fig = go.Figure(data=[go.Candlestick(x=plot_df['Date'],
                            open=plot_df['Open'], high=plot_df['High'],
                            low=plot_df['Low'], close=plot_df['Close'], name='K線')])
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA_5'], line=dict(color='orange', width=1.5), name='5日均線'))
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA_10'], line=dict(color='blue', width=1.5), name='10日均線'))
            fig.update_layout(title=f'{ticker} 近 90 日走勢與均線', yaxis_title='股價 (TWD)', xaxis_title='日期', template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 特徵重要性圖表
            st.markdown("### 🔍 模型決策關鍵 (特徵重要性)")
            importance_df = pd.DataFrame({'特徵': features, '重要性': model_xgb.feature_importances_}).sort_values(by='重要性', ascending=True)
            fig_imp = go.Figure(go.Bar(x=importance_df['重要性'], y=importance_df['特徵'], orientation='h', marker=dict(color='teal')))
            fig_imp.update_layout(title=f'{ticker} 專屬 XGBoost 特徵重要性分析', xaxis_title='重要性權重', yaxis_title='特徵名稱', template='plotly_white', height=400)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        except Exception as e:
            st.error(f"發生系統錯誤，可能為輸入代號無效或 API 連線異常。錯誤細節：{e}")

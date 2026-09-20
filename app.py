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
    # 放寬到 6 碼，以支援如 00679B 這種 ETF，或是 2881A 特別股
    raw_ticker = st.text_input("輸入台股代號 (如: 2330, 00679B)", value="2330", max_chars=6)
with col2:
    keyword = st.text_input("輸入公司名稱或關鍵字 (用於抓取新聞情緒)", value="台積電")
    
# 聰明防呆：去掉前後多餘空白，並自動轉大寫
clean_ticker = raw_ticker.strip().upper()

# 如果使用者自己手癢打了 .TW，我們就不重複加；否則才幫他加上 .TW
if clean_ticker.endswith(".TW") or clean_ticker.endswith(".TWO"):
    ticker = clean_ticker
else:
    ticker = f"{clean_ticker}.TW"

# === 2. 核心運算區 (即時訓練與預測) ===
if st.button(f"🚀 啟動 {ticker} 即時訓練與預測", type="primary"):
    with st.spinner(f'正在抓取 {ticker} 過去兩年資料並訓練專屬模型中，請稍候約 5 秒...'):
        try:
            # [A] 抓取歷史股價 (2年，用於現場訓練)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            stock = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
            
            # 防呆：找不到股票
            if stock.empty:
                st.error("❌ 找不到該股票資料，請確認代號是否正確。")
                st.stop()
                
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.droplevel(1)
            stock.reset_index(inplace=True)
            df = stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # 防呆：上市時間太短的股票不予訓練
            if len(df) < 300:
                st.error(f"❌ {ticker} 上市時間過短，歷史資料不足 (需至少 300 天)，無法進行機器學習訓練！")
                st.stop()
                
            # [B] 抓取新聞情緒 (近 3 天)
            url = f"https://news.google.com/rss/search?q={keyword}+when:3d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            feed = feedparser.parse(url)
            sentiment_scores = []
            for entry in feed.entries:
                try:
                    sentiment_scores.append(SnowNLP(entry.title).sentiments)
                except:
                    pass
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
            df['Sentiment'] = avg_sentiment  
            
            # [C] 計算技術指標
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=10, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            
            # [D] 定義預測目標 (Y) 與特徵 (X)
            # 今天的 Target 是明天的漲跌，所以把 Close 往上移一格
            df['Next_Close'] = df['Close'].shift(-1)
            df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'Sentiment', 'SMA_5', 'SMA_10', 'RSI_14', 
                        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
                        
            # 準備訓練資料 (剔除 NaN，特別是最後一天沒有 Next_Close)
            train_df = df.dropna(subset=features + ['Target'])
            
            # 準備今日特徵 (也就是 df 的最後一筆，用來推論明天)
            latest_data = df.iloc[-1:]
            X_today = latest_data[features]
            
            # [E] 🔥 核心魔法：即時動態訓練 (On-the-fly) 🔥
            model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            model.fit(train_df[features], train_df['Target'])
            
            # [F] 進行預測
            prediction = model.predict(X_today)[0]
            probability = model.predict_proba(X_today)[0]
            
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
                
            # 畫 K 線圖 (只取近 90 天畫圖比較好看)
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
            importance_df = pd.DataFrame({'特徵': features, '重要性': model.feature_importances_}).sort_values(by='重要性', ascending=True)
            fig_imp = go.Figure(go.Bar(x=importance_df['重要性'], y=importance_df['特徵'], orientation='h', marker=dict(color='teal')))
            fig_imp.update_layout(title=f'{ticker} 專屬 XGBoost 特徵重要性分析', xaxis_title='重要性權重', yaxis_title='特徵名稱', template='plotly_white', height=400)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        except Exception as e:
            st.error(f"發生系統錯誤，可能為輸入代號無效或 API 連線異常。錯誤細節：{e}")
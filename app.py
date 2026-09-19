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
import json
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI 股市預測系統", layout="wide")

# ==========================================
# ⚡ 系統效能優化區：導入快取機制 (Caching)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_stock_and_market_data(ticker):
    """同時抓取個股與大盤歷史資料，並快取 1 小時"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # 1. 抓取個股資料
    stock = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
    if stock.empty:
        return pd.DataFrame()
        
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.droplevel(1)
    stock.reset_index(inplace=True)
    df = stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 2. 抓取台灣加權指數 (^TWII) 作為總體經濟特徵
    market = yf.download("^TWII", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
    if not market.empty:
        if isinstance(market.columns, pd.MultiIndex):
            market.columns = market.columns.droplevel(1)
        market.reset_index(inplace=True)
        market = market[['Date', 'Close']].rename(columns={'Close': 'TWII_Close'})
        market['TWII_Return'] = market['TWII_Close'].pct_change()
        
        # 3. 將大盤漲跌幅合併進個股資料表
        df = pd.merge(df, market[['Date', 'TWII_Return']], on='Date', how='left')
        df['TWII_Return'].fillna(0, inplace=True)
    else:
        df['TWII_Return'] = 0.0
        
    return df

@st.cache_data(ttl=3600)
def fetch_news_sentiment(keyword):
    """抓取新聞情緒並快取 1 小時，包含 JSON 結構化解析"""
    url = f"https://news.google.com/rss/search?q={keyword}+when:3d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    feed = feedparser.parse(url)
    news_titles = [entry.title for entry in feed.entries[:5]]
    
    avg_sentiment = 0.5
    ai_reason = "近期無相關財經新聞，模型以中立情緒計算。"
    
    if news_titles:
        try:
            import google.generativeai as genai
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            
            valid_model_name = 'gemini-3.6-flash'
            model = genai.GenerativeModel(valid_model_name)
            
            prompt = f"""你是一個專業的台灣股市分析師。請綜合分析以下新聞標題對該公司股價的情緒影響。
            請務必只回傳一個標準的 JSON 格式字串，不要包含任何其他解釋文字或 Markdown 標籤。
            格式範例：{{"score": 0.8, "reason": "因為營收創新高且外資調升評等，市場情緒樂觀。"}}
            score 必須是 0.0 到 1.0 的浮點數（0.0為極度看跌，1.0為極度看漲，0.5為中立）。
            新聞標題：{news_titles}"""
            
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
                
            result = json.loads(raw_text)
            avg_sentiment = float(result.get("score", 0.5))
            ai_reason = result.get("reason", "AI 判定為中立或無特別理由。")
            
        except Exception as e:
            ai_reason = f"⚠️ API 呼叫失敗，已切換回 SnowNLP 備用模組計算。錯誤細節：{e}"
            sentiment_scores = []
            for title in news_titles:
                try:
                    sentiment_scores.append(SnowNLP(title).sentiments)
                except:
                    pass
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
    return avg_sentiment, ai_reason

# ==========================================
# 🖥️ 主程式與 UI 顯示區
# ==========================================
st.title("📈 畢業專題：AI 股市預測系統 (即時動態訓練版)")
st.markdown("本系統採用 **On-the-fly 即時訓練架構**，結合 LLM 情緒分析、大盤趨勢特徵，並導入 **風險控管停損機制** 的量化回測模組。")
st.warning("⚠️ 免責聲明：本系統僅供學術專題展示使用，不構成任何投資建議。")

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

if st.button(f"🚀 啟動 {ticker} 即時訓練與預測", type="primary"):
    with st.spinner(f'正在進行運算中，若為首次查詢需時 5-8 秒，再次查詢將啟動極速快取...'):
        try:
            # 呼叫快取函數抓資料 (包含個股與大盤 TWII)
            df = fetch_stock_and_market_data(ticker)
            
            if df.empty or len(df) < 300:
                st.error("❌ 找不到該股票資料或上市時間過短 (需至少 300 天)，無法進行機器學習訓練！")
                st.stop()
                
            # 呼叫快取函數抓新聞情緒與 AI 解析
            avg_sentiment, ai_reason = fetch_news_sentiment(keyword)
            df['Sentiment'] = avg_sentiment  
            
            # 計算技術指標
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=10, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            
            # 定義預測目標 (Y)
            df['Next_Close'] = df['Close'].shift(-1)
            df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
            
            # 🔥 將「大盤漲跌幅 (TWII_Return)」正式加入特徵矩陣中
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'Sentiment', 'TWII_Return', 'SMA_5', 'SMA_10', 'RSI_14', 
                        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
                        
            train_df = df.dropna(subset=features + ['Target'])
            latest_data = df.iloc[-1:]
            X_today = latest_data[features]
            
            # 即時動態訓練 (On-the-fly)
            model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            model_xgb.fit(train_df[features], train_df['Target'])
            
            # 進行預測
            prediction = model_xgb.predict(X_today)[0]
            probability = model_xgb.predict_proba(X_today)[0]
            
            st.success(f"✅ 專屬模型訓練完成！共使用 {len(train_df)} 筆歷史資料進行現場訓練，並已納入台灣加權指數(大盤)特徵。")
            
            # --- 畫面顯示區 ---
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
            
            # 🤖 新增 AI 財經新聞觀點區塊
            st.markdown("### 🤖 AI 財經新聞綜合觀點")
            st.info(f"**Gemini 洞察：** {ai_reason}")
                
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

            # --- 歷史回測模組 (含停損機制) ---
            st.markdown("---")
            st.markdown("### 📊 AI 策略 vs 單純持有：近半年歷史回測 (含 5% 停損機制)")
            
            backtest_days = 126
            if len(train_df) > backtest_days * 1.5:
                bt_train = train_df.iloc[:-backtest_days]
                bt_test = train_df.iloc[-backtest_days:].copy()
                
                bt_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
                bt_model.fit(bt_train[features], bt_train['Target'])
                
                bt_test['Prediction'] = bt_model.predict(bt_test[features])
                bt_test['Daily_Return'] = bt_test['Close'].pct_change()
                bt_test['Daily_Return'].fillna(0, inplace=True)
                
                # 計算 AI 策略原始報酬
                bt_test['Strategy_Return'] = bt_test['Prediction'].shift(1).fillna(0) * bt_test['Daily_Return']
                
                # 🔥 實作 5% 動態停損機制 (Risk Management)
                # 假設盤中跌幅過大，強迫在 -5% 停損出場，避免單日暴跌重創資產
                stop_loss_threshold = -0.05
                bt_test['Strategy_Return'] = bt_test['Strategy_Return'].clip(lower=stop_loss_threshold)
                
                bt_test['Cum_Market'] = (1 + bt_test['Daily_Return']).cumprod()
                bt_test['Cum_Strategy'] = (1 + bt_test['Strategy_Return']).cumprod()
                
                market_roi = (bt_test['Cum_Market'].iloc[-1] - 1) * 100
                strategy_roi = (bt_test['Cum_Strategy'].iloc[-1] - 1) * 100
                
                col3, col4 = st.columns(2)
                with col3:
                    st.metric(label="📈 AI 策略累積報酬 (近半年)", value=f"{strategy_roi:.2f}%", 
                              delta=f"勝過單純持有 {strategy_roi - market_roi:.2f}%" if strategy_roi > market_roi else f"落後單純持有 {strategy_roi - market_roi:.2f}%")
                with col4:
                    st.metric(label="📉 單純買進持有 (Buy & Hold)", value=f"{market_roi:.2f}%")
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=bt_test['Date'], y=bt_test['Cum_Strategy'], 
                                            line=dict(color='red', width=2.5), name='AI 交易策略 (含停損)'))
                fig_bt.add_trace(go.Scatter(x=bt_test['Date'], y=bt_test['Cum_Market'], 
                                            line=dict(color='gray', width=1.5, dash='dash'), name='單純買進持有'))
                
                fig_bt.update_layout(title=f'{ticker} 近半年 AI 策略與單純持有之績效對決',
                                     yaxis_title='累積資產倍數 (1.0 = 本金)', xaxis_title='日期',
                                     template='plotly_white', height=450, hovermode='x unified')
                st.plotly_chart(fig_bt, use_container_width=True)
                
                st.caption("ℹ️ **回測與停損邏輯說明**：系統保留最近半年數據作為盲測，採用時間平移 (Shift) 排除未來函數。並實作**單日 5% 停損機制**，當策略持倉且單日跌幅超過 5% 時，模擬盤中強制出場，強化資產下檔保護。")
            else:
                st.info("歷史資料不足以進行嚴謹的半年期回測。")
                
        except Exception as e:
            st.error(f"發生系統錯誤，可能為輸入代號無效或 API 連線異常。錯誤細節：{e}")

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
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
# ⚡ 系統效能優化與智慧辨識區 (Caching)
# ==========================================
@st.cache_data(ttl=86400)
def get_company_name(clean_ticker):
    """自動將台股代號轉譯為中文公司/ETF名稱，避免使用者手動輸入"""
    # 常見權值股與熱門 ETF 快速比對表
    top_stocks = {
        '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電',
        '2382': '廣達', '2412': '中華電', '2881': '富邦金', '2882': '國泰金',
        '2603': '長榮', '2609': '陽明', '2615': '萬海', '3008': '大立光',
        '0050': '元大台灣50', '0056': '元大高股息', '00878': '國泰永續高股息',
        '00929': '復華台灣科技優息', '00919': '群益台灣精選高息', '00679B': '元大美債20年'
    }
    if clean_ticker in top_stocks:
        return top_stocks[clean_ticker]
    
    # 其他標的由 Gemini 於背景自動推論名稱
    try:
        import google.generativeai as genai
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.6-flash')
        prompt = f"請將台股代號 {clean_ticker} 轉換為繁體中文的公司簡稱或ETF名稱（例如 2330 回傳 台積電，0050 回傳 元大台灣50）。請只回傳名稱本身，不要包含代號、任何標點或多餘說明。"
        resp = model.generate_content(prompt)
        name = resp.text.strip()
        if name and len(name) <= 15:
            return name
    except Exception:
        pass
        
    return clean_ticker

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
            請務必只回傳一個標準的 JSON 格式字串，不要包含任何其他解釋文字。
            格式範例：{{"score": 0.8, "reason": "因為營收創新高，市場情緒樂觀。"}}
            score 必須是 0.0 到 1.0 的浮點數。
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

def generate_ai_report(ticker, company_name, prediction, confidence, ai_reason, top_features, strategy_roi, market_roi):
    """呼叫 Gemini 進行綜合數據的投顧報告生成"""
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel('gemini-3.6-flash')
        prompt = f"""你是一位頂級的量化交易首席分析師。請根據以下 AI 模型運算結果，為投資人撰寫一篇 150-200 字的專業綜合診斷報告。
        語氣要客觀、專業、具備說服力。直接給出分析結論。
        
        【系統數據分析結果】
        1. 分析標的：{ticker} ({company_name})
        2. 模型預測明日走勢：{'上漲' if prediction == 1 else '下跌'} (模型信心水準：{confidence:.1f}%)
        3. 消息面解讀：{ai_reason}
        4. 決策最關鍵特徵 (技術/大盤/籌碼)：{', '.join(top_features)}
        5. 近半年量化回測報酬率：AI 策略 {strategy_roi:.2f}% vs 單純持有 {market_roi:.2f}%
        
        請統整以上數據，給出一段清晰的投資與風險控管建議。"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "⚠️ 首席分析師 AI 暫時離線，無法生成總結報告。"

# ==========================================
# 🖥️ 主程式與 UI 顯示區
# ==========================================
st.title("📈 畢業專題：AI 股市預測系統 (AutoML 終極版)")
st.markdown("本系統整合了 **On-the-fly AutoML 超參數調優**、**Gemini 首席分析師**、**籌碼動能特徵 (OBV/MFI)** 與 **風險控管回測模組**。")
st.warning("⚠️ 免責聲明：本系統僅供學術專題展示使用，不構成任何投資建議。")

st.markdown("### 🎯 請輸入預測標的")

# 簡化輸入區：僅需輸入股票代號
raw_ticker = st.text_input("輸入台股代號 (如: 2330, 2454, 0050, 00679B)", value="2330", max_chars=8)
clean_ticker = raw_ticker.strip().upper()

# 自動補齊 .TW 後綴
if clean_ticker.endswith(".TW") or clean_ticker.endswith(".TWO"):
    ticker = clean_ticker
    pure_code = clean_ticker.split('.')[0]
else:
    pure_code = clean_ticker
    ticker = f"{clean_ticker}.TW"

# 背景自動對應公司名稱
company_name = get_company_name(pure_code)
st.caption(f"📌 目標公司/ETF 自動識別：**{pure_code} {company_name}**")

if st.button(f"🚀 啟動 {ticker} ({company_name}) 深度訓練與預測", type="primary"):
    with st.spinner(f'系統正在自動分析 {company_name} 最新消息與歷史數據，請稍候約 10 秒...'):
        try:
            # 1. 抓取資料
            df = fetch_stock_and_market_data(ticker)
            if df.empty or len(df) < 300:
                st.error(f"❌ 找不到 {ticker} 歷史資料或上市時間過短 (需至少 300 天)，無法進行訓練！")
                st.stop()
                
            # 2. 自動使用辨識出的公司名稱抓取新聞情緒
            avg_sentiment, ai_reason = fetch_news_sentiment(company_name)
            df['Sentiment'] = avg_sentiment  
            
            # 3. 進階特徵工程 (技術指標 + 籌碼動能 OBV, MFI)
            df.ta.sma(length=5, append=True)
            df.ta.sma(length=10, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.obv(append=True)
            df.ta.mfi(length=14, append=True)
            
            df['Next_Close'] = df['Close'].shift(-1)
            df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'TWII_Return', 
                        'SMA_5', 'SMA_10', 'RSI_14', 'MACD_12_26_9', 'OBV', 'MFI_14']
                        
            train_df = df.dropna(subset=features + ['Target'])
            latest_data = df.iloc[-1:]
            X_today = latest_data[features]
            
            # 4. AutoML 動態超參數調優
            param_grid = {
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 150]
            }
            base_model = XGBClassifier(random_state=42)
            rs = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=5, cv=3, random_state=42)
            rs.fit(train_df[features], train_df['Target'])
            model_xgb = rs.best_estimator_
            best_params = rs.best_params_
            
            prediction = model_xgb.predict(X_today)[0]
            probability = model_xgb.predict_proba(X_today)[0]
            
            st.success(f"✅ AutoML 訓練完成！系統已為 {company_name} ({ticker}) 調校出最佳參數：{best_params}。")
            
            # 5. 畫面顯示區
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
            
            # 歷史回測模組 (先算出來提供給 AI 診斷報告使用)
            backtest_days = 126
            strategy_roi, market_roi = 0, 0
            bt_test = pd.DataFrame()
            if len(train_df) > backtest_days * 1.5:
                bt_train = train_df.iloc[:-backtest_days]
                bt_test = train_df.iloc[-backtest_days:].copy()
                
                bt_model = XGBClassifier(**best_params, random_state=42)
                bt_model.fit(bt_train[features], bt_train['Target'])
                
                bt_test['Prediction'] = bt_model.predict(bt_test[features])
                bt_test['Daily_Return'] = bt_test['Close'].pct_change()
                bt_test['Daily_Return'].fillna(0, inplace=True)
                
                bt_test['Strategy_Return'] = bt_test['Prediction'].shift(1).fillna(0) * bt_test['Daily_Return']
                bt_test['Strategy_Return'] = bt_test['Strategy_Return'].clip(lower=-0.05) # 5% 停損
                
                bt_test['Cum_Market'] = (1 + bt_test['Daily_Return']).cumprod()
                bt_test['Cum_Strategy'] = (1 + bt_test['Strategy_Return']).cumprod()
                
                market_roi = (bt_test['Cum_Market'].iloc[-1] - 1) * 100
                strategy_roi = (bt_test['Cum_Strategy'].iloc[-1] - 1) * 100

            # 特徵重要性分析
            importance_df = pd.DataFrame({'特徵': features, '重要性': model_xgb.feature_importances_}).sort_values(by='重要性', ascending=True)
            top_3_features = importance_df['特徵'].iloc[-3:].tolist()

            # 6. 首席 AI 投顧報告區
            st.markdown("### 🤖 首席 AI 總體診斷報告")
            with st.spinner("AI 正在綜整技術面、籌碼動能與最新新聞..."):
                final_report = generate_ai_report(ticker, company_name, prediction, confidence, ai_reason, top_3_features, strategy_roi, market_roi)
                st.info(final_report)
                
            # 畫 K 線圖與特徵重要性圖表
            st.markdown("---")
            plot_df = df.tail(90)
            fig = go.Figure(data=[go.Candlestick(x=plot_df['Date'],
                            open=plot_df['Open'], high=plot_df['High'],
                            low=plot_df['Low'], close=plot_df['Close'], name='K線')])
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA_5'], line=dict(color='orange', width=1.5), name='5日均線'))
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA_10'], line=dict(color='blue', width=1.5), name='10日均線'))
            fig.update_layout(title=f'{company_name} ({ticker}) 近 90 日走勢與均線', yaxis_title='股價 (TWD)', xaxis_title='日期', template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            fig_imp = go.Figure(go.Bar(x=importance_df['重要性'], y=importance_df['特徵'], orientation='h', marker=dict(color='teal')))
            fig_imp.update_layout(title=f'{company_name} ({ticker}) 專屬 XGBoost 特徵重要性分析', xaxis_title='重要性權重', yaxis_title='特徵名稱', template='plotly_white', height=400)
            st.plotly_chart(fig_imp, use_container_width=True)

            # 顯示回測結果
            st.markdown("---")
            st.markdown("### 📊 AI 策略 vs 單純持有：近半年歷史回測 (含 5% 停損機制)")
            if not bt_test.empty:
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
                
                fig_bt.update_layout(title=f'{company_name} ({ticker}) 近半年 AI 策略與單純持有之績效對決',
                                     yaxis_title='累積資產倍數 (1.0 = 本金)', xaxis_title='日期',
                                     template='plotly_white', height=450, hovermode='x unified')
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.info("歷史資料不足以進行嚴謹的半年期回測。")
                
        except Exception as e:
            st.error(f"發生系統錯誤，可能為輸入代號無效或 API 連線異常。錯誤細節：{e}")

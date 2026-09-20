import yfinance as yf
import pandas as pd
import feedparser
from snownlp import SnowNLP
from datetime import datetime, timedelta
import json
import re
import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

def get_company_name(clean_ticker: str) -> str:
    """自動將台股代號轉譯為中文公司/ETF名稱"""
    top_stocks = {
        '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電',
        '2382': '廣達', '2412': '中華電', '2881': '富邦金', '2882': '國泰金',
        '2603': '長榮', '2609': '陽明', '2615': '萬海', '3008': '大立光',
        '0050': '元大台灣50', '0056': '元大高股息', '00878': '國泰永續高股息'
    }
    if clean_ticker in top_stocks:
        return top_stocks[clean_ticker]
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        resp = genai.GenerativeModel('gemini-3.6-flash').generate_content(
            f"將台股代號 {clean_ticker} 轉換為公司簡稱。只回傳名稱，不要標點。"
        )
        name = resp.text.strip()
        if name and len(name) <= 15: 
            return name
    except Exception:
        pass
    return clean_ticker

def fetch_stock_and_macro_data(ticker: str) -> pd.DataFrame:
    """抓取個股與總體經濟特徵"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    start_str, end_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    stock = yf.download(ticker, start=start_str, end=end_str, progress=False)
    if stock.empty: 
        return pd.DataFrame()
        
    if isinstance(stock.columns, pd.MultiIndex): 
        stock.columns = stock.columns.droplevel(1)
    stock.reset_index(inplace=True)
    df = stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    macro_symbols = {"^TWII": "TWII_Return", "^SOX": "SOX_Return", "USDTWD=X": "USDTWD_Return"}
    for sym, col_name in macro_symbols.items():
        macro = yf.download(sym, start=start_str, end=end_str, progress=False)
        if not macro.empty:
            if isinstance(macro.columns, pd.MultiIndex): 
                macro.columns = macro.columns.droplevel(1)
            macro.reset_index(inplace=True)
            macro['Return'] = macro['Close'].pct_change()
            df = pd.merge(df, macro[['Date', 'Return']], on='Date', how='left')
            df.rename(columns={'Return': col_name}, inplace=True)
        else:
            df[col_name] = 0.0
            
    df.fillna(0, inplace=True)
    return df

def fetch_news_sentiment(keyword: str) -> tuple:
    """抓取新聞情緒並使用 LLM 分析"""
    url = f"https://news.google.com/rss/search?q={keyword}+when:3d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    feed = feedparser.parse(url)
    news_titles = [entry.title for entry in feed.entries[:5]]
    avg_sentiment, ai_reason = 0.5, "近期無相關財經新聞，模型以中立情緒計算。"
    
    if news_titles:
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            prompt = f"""你是一個台灣股市分析師。分析以下新聞標題情緒，只回傳JSON，格式：{{"score": 0.8, "reason": "利多..."}}
            新聞標題：{news_titles}"""
            raw_text = genai.GenerativeModel('gemini-3.6-flash').generate_content(prompt).text.strip()
            if raw_text.startswith("```"): 
                raw_text = re.sub(r"^```(json)?|```$", "", raw_text, flags=re.MULTILINE).strip()
            result = json.loads(raw_text)
            return float(result.get("score", 0.5)), result.get("reason", "中立")
        except Exception as e:
            ai_reason = f"API 呼叫失敗，已切換備用模組。錯誤：{e}"
            sentiment_scores = [SnowNLP(t).sentiments for t in news_titles if True]
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    return avg_sentiment, ai_reason
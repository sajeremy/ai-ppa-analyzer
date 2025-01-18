import streamlit as st
import requests
import google.generativeai as palm
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure API keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
palm.configure(api_key=os.getenv('PALM_API_KEY'))

# Streamlit page config
st.set_page_config(page_title="AI Counterparty Risk Dashboard", layout="wide")

def fetch_financial_data(symbol):
    """Fetch financial data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Calculate financial metrics
        market_cap = info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap = f"{market_cap:,}"
            
        data = {
            'Company Name': info.get('longName', symbol),
            'Debt-to-Equity Ratio': round(info.get('debtToEquity', 0) / 100, 3),
            'Market Cap': market_cap,
            'Current Price': info.get('currentPrice', 'N/A'),
            'P/E Ratio': round(info.get('forwardPE', 0), 4),
            'Cash Flow': info.get('freeCashflow', 'N/A')
        }
        return data
    except Exception as e:
        st.error(f"Error fetching financial data: {e}")
        return None

def fetch_news_articles(symbol, page_size=10):
    """Fetch news articles using NewsAPI"""
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&pageSize={page_size}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def analyze_sentiment(articles):
    """Analyze sentiment of articles using Google Gemini"""
    try:
        model = palm.GenerativeModel('gemini-2.0-flash-exp')
        
        # Prepare texts for analysis
        texts = [f"{article['title']}. {article['description']}" for article in articles]
        
        # Create sentiment analysis prompt
        prompt = "Analyze the sentiment of the following news. Respond with ONLY ONE WORD (Positive/Neutral/Negative) for each:\n\n"
        for text in texts:
            prompt += f"- {text}\n"
        
        response = model.generate_content(prompt)
        if response:
            sentiments = [s.strip() for s in response.text.split('\n') if s.strip()]
            return sentiments[:len(texts)]
        return ['Neutral'] * len(texts)
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return ['Neutral'] * len(texts)

def calculate_risk_scores(financial_data, sentiments):
    """Calculate various risk scores"""
    # Financial Risk Score (0-100)
    try:
        de_ratio = financial_data['Debt-to-Equity Ratio']
        financial_risk = min(100, max(0, de_ratio * 10))  # Scale debt-to-equity ratio
    except:
        financial_risk = 50  # Default value if calculation fails
    
    # Sentiment Risk Score (0-100)
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for sentiment in sentiments:
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    total_sentiments = len(sentiments) or 1
    negative_percentage = (sentiment_counts['Negative'] / total_sentiments) * 100
    sentiment_risk = negative_percentage
    
    # Legal Risk Score (placeholder - would need actual legal data)
    legal_risk = 0
    
    # Overall Risk Score (weighted average)
    overall_risk = (financial_risk * 0.4) + (sentiment_risk * 0.4) + (legal_risk * 0.2)
    
    return {
        'Overall Risk Score': round(overall_risk, 1),
        'Financial Risk Score': round(financial_risk, 1),
        'Sentiment Risk Score': round(sentiment_risk, 1),
        'Legal Risk Score': round(legal_risk, 1)
    }

def main():
    # Title and description
    st.title("🔍 AI Counterparty Risk Dashboard")
    
    # Input for stock symbol
    symbol = st.text_input("Enter Company Stock Symbol:", "FSLR").upper()
    
    if st.button("Fetch Data"):
        # Fetch and display data
        with st.spinner("Fetching data..."):
            # Get financial data
            financial_data = fetch_financial_data(symbol)
            if not financial_data:
                st.error("Unable to fetch financial data")
                return
            
            # Get news and sentiment
            articles = fetch_news_articles(symbol)
            sentiments = analyze_sentiment(articles)
            
            # Calculate risk scores
            risk_scores = calculate_risk_scores(financial_data, sentiments)
            
            # Display Risk Overview
            st.header("📊 FSLR Risk Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⚠️ Overall Risk Score", f"{risk_scores['Overall Risk Score']}")
            with col2:
                st.metric("💰 Financial Risk Score", f"{risk_scores['Financial Risk Score']}")
            with col3:
                st.metric("📰 Sentiment Risk Score", f"{risk_scores['Sentiment Risk Score']}")
            with col4:
                st.metric("⚖️ Legal Risk Score", f"{risk_scores['Legal Risk Score']}")
            
            # Display Financial Data
            st.header("📈 Financial Data")
            df = pd.DataFrame([financial_data])
            st.dataframe(df)
            
            # Display News & Sentiment Analysis
            st.header("📰 News & Sentiment Analysis")
            for article, sentiment in zip(articles, sentiments):
                with st.expander(article['title']):
                    st.write(article['description'])
                    st.write(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
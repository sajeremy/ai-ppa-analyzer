import streamlit as st
import requests
import google.generativeai as palm
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure API keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
palm.configure(api_key=os.getenv('PALM_API_KEY'))

# Streamlit page config
st.set_page_config(
    page_title="AI Counterparty Risk Dashboard",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 15px;
        background-color: #e6f4ea;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 15px;
        background-color: #fff8e6;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 15px;
        background-color: #fce8e8;
    }
    .news-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def fetch_news_articles(symbol):
    """Fetch news articles using NewsAPI"""
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&pageSize=10&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def analyze_sentiment(articles):
    """Analyze sentiment of articles using Google Gemini"""
    try:
        model = palm.GenerativeModel('gemini-pro')
        texts = [f"{article['title']}. {article['description']}" for article in articles]
        
        prompt = "Analyze the sentiment of the following news. For each text, respond with exactly ONE WORD from these options: Positive, Neutral, or Negative. Provide each response on a new line without any prefixes or additional text.\n\n"
        for i, text in enumerate(texts, 1):
            prompt += f"Text {i}: {text}\n"
        
        response = model.generate_content(prompt)
        if response:
            # Clean up sentiments - remove any prefixes and ensure valid values
            valid_sentiments = {'Positive', 'Neutral', 'Negative'}
            sentiments = []
            for s in response.text.split('\n'):
                # Clean and validate each sentiment
                cleaned = s.strip().strip('- ').strip()
                if cleaned in valid_sentiments:
                    sentiments.append(cleaned)
                else:
                    sentiments.append('Neutral')  # Default to Neutral for invalid responses
            return sentiments[:len(texts)]
        return ['Neutral'] * len(texts)
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return ['Neutral'] * len(texts)

def create_sentiment_gauge(score):
    """Create a gauge chart for sentiment score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#e8f4f8"},
                {'range': [33, 66], 'color': "#bfdde9"},
                {'range': [66, 100], 'color': "#95c6da"}
            ],
        },
        title={'text': "Overall Sentiment Score"}
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def create_sentiment_distribution(sentiments):
    """Create a pie chart for sentiment distribution"""
    sentiment_counts = pd.Series(sentiments).value_counts()
    colors = ['#28a745', '#ffc107', '#dc3545']
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.3,
        marker_colors=colors
    )])
    fig.update_layout(
        title="Sentiment Distribution",
        height=250,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def display_news_card(article, sentiment):
    """Display a formatted news card"""
    sentiment_class = f"sentiment-{sentiment.lower()}"
    
    st.markdown(f"""
        <div class="news-card">
            <h4>{article['title']}</h4>
            <p>{article['description']}</p>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="{sentiment_class}">{sentiment}</span>
                <span style="color: #666; font-size: 0.9em;">
                    {datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d').strftime('%B %d, %Y')}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def calculate_sentiment_score(sentiments):
    """Calculate overall sentiment score"""
    weights = {'Positive': 100, 'Neutral': 50, 'Negative': 0}
    scores = [weights[s] for s in sentiments]
    return sum(scores) / len(scores) if scores else 50

def fetch_financial_data(symbol):
    """Fetch financial data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
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

def main():
    st.title("🔍 AI Counterparty Risk Dashboard")
    
    # Input for stock symbol with fetch button
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter Company Stock Symbol:", "FSLR").upper()
    with col2:
        fetch_button = st.button("🔄 Fetch Data", use_container_width=True)
    
    if fetch_button:
        with st.spinner("Analyzing data..."):
            # Get news and sentiment
            articles = fetch_news_articles(symbol)
            sentiments = analyze_sentiment(articles)
            
            # Calculate overall sentiment score
            sentiment_score = calculate_sentiment_score(sentiments)
            
            # Display sentiment overview
            st.markdown("### 📊 Sentiment Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_sentiment_gauge(sentiment_score), use_container_width=True)
            with col2:
                st.plotly_chart(create_sentiment_distribution(sentiments), use_container_width=True)
            
            # Display news with sentiments
            st.markdown("### 📰 Latest News & Sentiment Analysis")
            
            # Display all news
            for article, sentiment in zip(articles, sentiments):
                display_news_card(article, sentiment)
            
            # Financial metrics
            st.markdown("### 💹 Financial Metrics")
            financial_data = fetch_financial_data(symbol)
            if financial_data:
                metrics_df = pd.DataFrame([financial_data])
                st.dataframe(
                    metrics_df.style.background_gradient(cmap='Blues'),
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
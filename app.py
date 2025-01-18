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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with dark theme
st.markdown("""
    <style>
    /* Dark theme colors */
    :root {
        --background-color: #1a1a1a;
        --text-color: #ffffff;
        --card-background: #2d2d2d;
        --accent-color: #4d9fff;
    }
    
    /* Main container */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Custom sentiment badges */
    .sentiment-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 1.2em;
        font-weight: 600;
        text-align: center;
        min-width: 120px;
        margin-right: 10px;
    }
    
    .sentiment-positive {
        background-color: rgba(40, 167, 69, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    
    .sentiment-neutral {
        background-color: rgba(255, 193, 7, 0.2);
        color: #ffd700;
        border: 1px solid #ffd700;
    }
    
    .sentiment-negative {
        background-color: rgba(220, 53, 69, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    
    /* News card styling */
    .news-card {
        background-color: var(--card-background);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .news-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .news-title {
        font-size: 1.1em;
        font-weight: 500;
        color: var(--text-color);
        margin-bottom: 10px;
    }
    
    .news-date {
        color: #888;
        font-size: 0.9em;
    }
    
    /* Input container */
    .input-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 30px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5em;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--accent-color);
    }
    
    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .bg {
        background-color: var(--card-background) !important;
    }
    
    /* Add these new styles for the input and button container */
    .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    .stButton > button {
        width: 100%;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        padding: 0.5rem 1rem;
    }
    
    /* Ensure the text input takes full width */
    .stTextInput > div > div > input {
        width: 100%;
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
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#4d9fff"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(220, 53, 69, 0.3)"},
                {'range': [33, 66], 'color': "rgba(255, 193, 7, 0.3)"},
                {'range': [66, 100], 'color': "rgba(40, 167, 69, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        },
        title={'text': "Overall Sentiment Score", 'font': {'color': 'white', 'size': 24}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig


def create_sentiment_distribution(sentiments):
    """Create a pie chart for sentiment distribution"""
    sentiment_counts = pd.Series(sentiments).value_counts()
    colors = ['#2ecc71', '#ffd700', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.4,
        marker_colors=colors,
        textinfo='percent',
        textfont_size=14,
        textfont_color='white'
    )])
    
    fig.update_layout(
        title={
            'text': "Sentiment Distribution",
            'font': {'size': 24, 'color': 'white'},
            'y': 0.95
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=True,
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)'
        )
    )
    return fig


def display_news_card(article, sentiment):
    """Display a formatted news card with prominent sentiment"""
    sentiment_class = f"sentiment-{sentiment.lower()}"
    
    st.markdown(f"""
        <div class="news-card">
            <div class="news-header">
                <span class="sentiment-badge {sentiment_class}">{sentiment}</span>
                <span class="news-date">
                    {datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d').strftime('%B %d, %Y')}
                </span>
            </div>
            <div class="news-title">{article['title']}</div>
            <div style="color: #888;">{article['description']}</div>
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
    st.markdown('<h1 style="color: white;">🔍 AI Counterparty Risk Dashboard</h1>', unsafe_allow_html=True)
    
    # Center the input field
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        symbol = st.text_input("", "TSLA", placeholder="Enter Company Stock Symbol")
        # Place button directly below input, centered
        fetch_button = st.button("🔄 Analyze", use_container_width=True)
    
    if fetch_button:
        with st.spinner("Analyzing market sentiment..."):
            # Get news and analyze sentiment
            articles = fetch_news_articles(symbol)
            sentiments = analyze_sentiment(articles)
            sentiment_score = calculate_sentiment_score(sentiments)
            
            # Sentiment Overview Section
            st.markdown('<div class="section-header">📊 Market Sentiment Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_sentiment_gauge(sentiment_score), use_container_width=True)
            with col2:
                st.plotly_chart(create_sentiment_distribution(sentiments), use_container_width=True)
            
            # News Section
            st.markdown('<div class="section-header">📰 Latest News Impact</div>', unsafe_allow_html=True)
            for article, sentiment in zip(articles, sentiments):
                display_news_card(article, sentiment)

if __name__ == "__main__":
    main()
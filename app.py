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
import numpy as np
from scipy.stats import norm

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
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: fit-content;
    }
    
    .news-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .news-title {
        font-size: 0.9em;
        font-weight: 500;
        color: var(--text-color);
        margin: 10px 0;
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
    
    /* Style for the details/summary elements */
    details {
        margin-top: 10px;
    }
    
    summary {
        color: var(--accent-color);
        cursor: pointer;
        font-size: 0.8em;
    }
    
    summary:hover {
        color: #6aafff;
    }
    
    /* Make sentiment badges more compact */
    .sentiment-badge {
        padding: 4px 12px;
        font-size: 0.9em;
        min-width: 90px;
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
        model = palm.GenerativeModel('gemini-2.0-flash-exp')
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
            <details>
                <summary>Read more</summary>
                <div style="color: #888; margin-top: 10px;">{article['description']}</div>
            </details>
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

def calculate_stock_volatility(symbol, period='1y'):
    """Calculate stock volatility score (0-100)"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        returns = np.log(hist['Close']/hist['Close'].shift(1))
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        # Convert to 0-100 score (higher volatility = lower score)
        score = 100 * (1 - norm.cdf(volatility, loc=0.3, scale=0.2))
        return max(0, min(100, score))
    except:
        return 50  # Default score if calculation fails

def calculate_interest_coverage(info):
    """Calculate interest coverage ratio score (0-100)"""
    try:
        ebit = info.get('ebit', 0)
        interest_expense = info.get('interestExpense', 0)
        if interest_expense == 0:
            return 100
        ratio = abs(ebit / interest_expense)
        # Convert to 0-100 score (higher is better)
        score = min(100, ratio * 10)  # Scale ratio to 0-100
        return max(0, score)
    except:
        return 50

def calculate_liquidity_ratio(info):
    """Calculate liquidity ratio score (0-100)"""
    try:
        current_ratio = info.get('currentRatio', 0)
        # Convert to 0-100 score (optimal range around 1.5-3.0)
        if current_ratio < 1:
            score = current_ratio * 50
        elif current_ratio < 2:
            score = 75 + (current_ratio - 1) * 25
        else:
            score = 100 - max(0, (current_ratio - 2) * 10)
        return max(0, min(100, score))
    except:
        return 50

def calculate_de_ratio_score(de_ratio):
    """Calculate D/E ratio score (0-100)"""
    try:
        # Convert to 0-100 score (lower is better)
        if de_ratio <= 0:
            return 100
        score = 100 * (1 - norm.cdf(de_ratio, loc=2, scale=1))
        return max(0, min(100, score))
    except:
        return 50

def get_credit_rating_score(info):
    """Calculate credit rating score based on actual rating"""
    try:
        # Get financial health metrics as indicators
        total_debt = info.get('totalDebt', 0)
        total_assets = info.get('totalAssets', 1)  # Default to 1 to avoid division by zero
        
        # Ensure we have valid data
        if total_debt == 0 and total_assets == 1:
            return 50  # Return neutral score if no data
            
        debt_ratio = total_debt / total_assets
        
        # Calculate score based on financial metrics
        base_score = 100 * (1 - min(debt_ratio, 1))
        
        # Get credit rating from business summary if available
        rating = info.get('longBusinessSummary', '').lower()
        
        # Adjust score based on credit rating mentions
        if any(term in rating for term in ['aaa', 'aa', 'a+']):
            score = min(100, base_score + 20)
        elif any(term in rating for term in ['bbb', 'bb']):
            score = min(100, base_score + 10)
        elif any(term in rating for term in ['b', 'ccc']):
            score = max(0, base_score - 10)
        elif any(term in rating for term in ['cc', 'c', 'd']):
            score = max(0, base_score - 20)
        else:
            score = base_score
            
        return max(0, min(100, score))
    except Exception as e:
        st.warning(f"Credit rating calculation limited: {e}")
        return 50  # Return neutral score on error

def get_macro_risk_score(info):
    """Calculate macro risk score using market indicators"""
    try:
        # Get relevant market indicators
        beta = info.get('beta', 1)
        sector = info.get('sector', '').lower()
        
        # Get market cap
        market_cap = info.get('marketCap', 0)
        
        # Calculate base score using beta (inverse relationship)
        beta_score = 100 * (1 - min(abs(beta - 1), 1))
        
        # Adjust for market cap (larger companies typically have lower macro risk)
        size_score = 0
        if market_cap > 200e9:  # Large cap
            size_score = 100
        elif market_cap > 10e9:  # Mid cap
            size_score = 75
        elif market_cap > 2e9:  # Small cap
            size_score = 50
        else:  # Micro cap
            size_score = 25
            
        # Sector risk adjustment
        sector_adjustment = 0
        defensive_sectors = ['consumer staples', 'utilities', 'healthcare']
        cyclical_sectors = ['technology', 'consumer discretionary', 'real estate']
        volatile_sectors = ['energy', 'materials', 'cryptocurrencies']
        
        if any(s in sector for s in defensive_sectors):
            sector_adjustment = 10
        elif any(s in sector for s in cyclical_sectors):
            sector_adjustment = 0
        elif any(s in sector for s in volatile_sectors):
            sector_adjustment = -10
            
        # Calculate final score
        final_score = (beta_score * 0.4 + size_score * 0.4) + sector_adjustment
        return max(0, min(100, final_score))
    except:
        return 50

def calculate_financial_risk_score(symbol):
    """Calculate overall financial risk score"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Calculate individual components
        volatility_score = calculate_stock_volatility(symbol)
        de_ratio = info.get('debtToEquity', 0) / 100  # Convert from percentage
        de_score = calculate_de_ratio_score(de_ratio)
        interest_coverage_score = calculate_interest_coverage(info)
        liquidity_score = calculate_liquidity_ratio(info)
        credit_rating_score = get_credit_rating_score(info)
        macro_risk_score = get_macro_risk_score(info)
        
        # Calculate weighted score
        weights = {
            'Stock Volatility': 0.10,
            'Debt-to-Equity': 0.10,
            'Interest Coverage': 0.10,
            'Credit Rating': 0.10,
            'Liquidity': 0.05,
            'Macro Risk': 0.05
        }
        
        scores = {
            'Stock Volatility': volatility_score,
            'Debt-to-Equity': de_score,
            'Interest Coverage': interest_coverage_score,
            'Credit Rating': credit_rating_score,
            'Liquidity': liquidity_score,
            'Macro Risk': macro_risk_score
        }
        
        total_score = sum(score * weights[metric] for metric, score in scores.items())
        
        return total_score, scores, weights
        
    except Exception as e:
        st.error(f"Error calculating financial risk: {e}")
        return 50, {}, {}

def create_risk_gauge(score, title="Risk Score"):
    """Create a gauge chart for risk score"""
    # Color logic - low score (low risk) is green
    color = "rgba(40, 167, 69, 0.8)" if score <= 33 else \
            "rgba(255, 193, 7, 0.8)" if score <= 66 else \
            "rgba(220, 53, 69, 0.8)"
            
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#4d9fff"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': "rgba(40, 167, 69, 0.8)"},  # Green for low risk (0-33)
                {'range': [33, 66], 'color': "rgba(255, 193, 7, 0.8)"},  # Yellow for moderate risk (33-66)
                {'range': [66, 100], 'color': "rgba(220, 53, 69, 0.8)"}  # Red for high risk (66-100)
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        },
        title={'text': title, 'font': {'color': 'white', 'size': 24}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig

def create_component_chart(scores, weights):
    """Create a horizontal bar chart for risk components"""
    components = list(scores.keys())
    values = list(scores.values())
    
    # Create color scale based on scores
    colors = ['rgba(220, 53, 69, 0.8)' if v < 33 else
              'rgba(255, 193, 7, 0.8)' if v < 66 else
              'rgba(40, 167, 69, 0.8)' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=components,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title={'text': 'Risk Components', 'font': {'color': 'white', 'size': 24}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(
            title='Score',
            range=[0, 100],
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title='Component',
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        )
    )
    return fig

def main():
    st.markdown('<h1 style="color: white;">🔍 AI Counterparty Risk Dashboard</h1>', unsafe_allow_html=True)
    
    # Center the input field with proper label
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        symbol = st.text_input(
            label="Stock Symbol",
            value="FSLR",
            placeholder="Enter Company Stock Symbol",
            label_visibility="collapsed"
        )
        fetch_button = st.button("🔄 Analyze", use_container_width=True)
    
    if fetch_button:
        with st.spinner("Analyzing market sentiment..."):
            # Get news and analyze sentiment
            articles = fetch_news_articles(symbol)
            articles = sorted(articles, 
                            key=lambda x: datetime.strptime(x['publishedAt'][:19], '%Y-%m-%dT%H:%M:%S'),
                            reverse=True)
            sentiments = analyze_sentiment(articles)
            sentiment_score = calculate_sentiment_score(sentiments)
            
            # Financial Risk Section
            st.markdown('<div class="section-header">💰 Financial Risk Analysis</div>', unsafe_allow_html=True)
            risk_score, component_scores, weights = calculate_financial_risk_score(symbol)
            
            # Add error handling for empty scores
            if not component_scores:
                st.error("Unable to fetch financial data. Please check the stock symbol and try again.")
                return
                
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_risk_gauge(risk_score, "Financial Risk Score"), use_container_width=True)
            with col2:
                st.plotly_chart(create_component_chart(component_scores, weights), use_container_width=True)
            
            # Market Sentiment Section
            st.markdown('<div class="section-header">📊 Market Sentiment Analysis</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_sentiment_gauge(sentiment_score), use_container_width=True)
            with col2:
                st.plotly_chart(create_sentiment_distribution(sentiments), use_container_width=True)
            
            # News Section
            st.markdown('<div class="section-header">📰 Latest News Impact</div>', unsafe_allow_html=True)
            for i in range(0, len(articles), 2):
                col1, col2 = st.columns(2)
                with col1:
                    if i < len(articles):
                        display_news_card(articles[i], sentiments[i])
                with col2:
                    if i + 1 < len(articles):
                        display_news_card(articles[i + 1], sentiments[i + 1])

if __name__ == "__main__":
    main()
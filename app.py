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
from pdf_parser import analyze_document_risks  # Update this import
import json

# Load environment variables
load_dotenv()

# Configure API keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
palm.configure(api_key=os.getenv('GOOGLE_API_KEY'))

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
    """Calculate stock volatility risk score using beta (0-100, where higher = more risk)"""
    try:
        stock = yf.Ticker(symbol)
        beta = stock.info.get('beta', 1.0)
        
        # Convert beta to 0-100 risk score
        # Beta interpretation:
        # beta < 0.5: very low volatility
        # 0.5-0.8: low volatility
        # 0.8-1.2: market-like volatility
        # 1.2-2.0: high volatility
        # > 2.0: very high volatility
        
        if beta <= 0:  # Negative beta (rare) - treat as moderate risk
            score = 50
        elif beta <= 0.5:  # Very low volatility
            score = beta * 20  # Maps 0-0.5 beta to 0-10 score
        elif beta <= 0.8:  # Low volatility
            score = 10 + (beta - 0.5) * (30/0.3)  # Maps 0.5-0.8 beta to 10-40 score
        elif beta <= 1.2:  # Market-like volatility
            score = 40 + (beta - 0.8) * (20/0.4)  # Maps 0.8-1.2 beta to 40-60 score
        elif beta <= 2.0:  # High volatility
            score = 60 + (beta - 1.2) * (30/0.8)  # Maps 1.2-2.0 beta to 60-90 score
        else:  # Very high volatility
            score = 90 + min((beta - 2.0) * 5, 10)  # Maps beta > 2.0 to 90-100 score
            
        return max(0, min(100, score))
    except:
        return 50  # Return moderate risk score if calculation fails

def calculate_interest_coverage(info):
    """Calculate interest coverage ratio risk score (0-100, where higher = more risk)"""
    try:
        operating_cash = info.get('operatingCashflow', 0)
        total_debt = info.get('totalDebt', 1)  # Use 1 to avoid division by zero
        
        # Calculate debt service capability using Operating Cash Flow to Total Debt ratio
        coverage_ratio = (operating_cash / total_debt) if total_debt else 0
        
        # Convert to 0-100 risk score (lower coverage = higher risk)
        if coverage_ratio <= 0:  # Negative or no operating cash flow
            return 100
        elif coverage_ratio >= 0.3:  # Can cover 30% of debt with one year's cash flow
            return 0
        else:
            # Linear scaling between 0 and 0.3
            score = 100 - (coverage_ratio / 0.3) * 100
            
        return max(0, min(100, score))
    except:
        return 50  # Return moderate risk score if calculation fails

def calculate_liquidity_ratio(info):
    """Calculate liquidity ratio risk score (0-100, where higher = more risk)"""
    try:
        current_ratio = info.get('currentRatio', 0)
        # Convert to 0-100 risk score (lower liquidity = higher risk)
        if current_ratio < 1:
            score = 100 - (current_ratio * 50)
        elif current_ratio < 2:
            score = 50 - (current_ratio - 1) * 25
        else:
            score = max(0, (current_ratio - 2) * 10)
        return max(0, min(100, score))
    except:
        return 50

def calculate_de_ratio_score(de_ratio):
    """Calculate D/E ratio risk score (0-100, where higher = more risk)"""
    try:
        # Convert to 0-100 risk score (higher D/E = higher risk)
        if de_ratio <= 0:
            return 0
        score = 100 * norm.cdf(de_ratio, loc=2, scale=1)
        return max(0, min(100, score))
    except:
        return 50

def get_credit_rating_score(info):
    """Calculate credit rating risk score (0-100, where higher = more risk)"""
    try:
        # Get key metrics
        market_cap = info.get('marketCap', 0)
        total_debt = info.get('totalDebt', 0)
        ebitda = info.get('ebitda', 1)  # Use 1 to avoid division by zero
        current_ratio = info.get('currentRatio', 1)
        
        # Calculate key ratios
        debt_to_ebitda = total_debt / ebitda if ebitda else 5  # Higher is worse
        size_factor = 1 - min(market_cap / 1e11, 1)  # Smaller companies are riskier
        
        # Base score calculation (0-100, higher = more risk)
        # 1. Debt to EBITDA component (0-40 points)
        if debt_to_ebitda <= 2:
            debt_score = 0
        elif debt_to_ebitda <= 4:
            debt_score = 20
        else:
            debt_score = min(40, debt_to_ebitda * 10)
            
        # 2. Size component (0-30 points)
        size_score = size_factor * 30
        
        # 3. Current ratio component (0-30 points)
        if current_ratio >= 2:
            liquidity_score = 0
        elif current_ratio >= 1:
            liquidity_score = 15
        else:
            liquidity_score = 30
            
        # Calculate final score
        final_score = debt_score + size_score + liquidity_score
        
        return max(0, min(100, final_score))
        
    except Exception as e:
        st.warning(f"Credit rating calculation limited: {e}")
        return 50

def get_macro_risk_score(info):
    """Calculate macro risk score (0-100, where higher = more risk)"""
    try:
        beta = info.get('beta', 1)
        sector = info.get('sector', '').lower()
        market_cap = info.get('marketCap', 0)
        
        # Higher beta deviation from 1 = higher risk
        beta_score = 100 * min(abs(beta - 1), 1)
        
        # Smaller companies = higher risk
        size_score = 100
        if market_cap > 200e9:  # Large cap
            size_score = 25
        elif market_cap > 10e9:  # Mid cap
            size_score = 50
        elif market_cap > 2e9:  # Small cap
            size_score = 75
            
        # Sector risk adjustment
        sector_adjustment = 0
        defensive_sectors = ['consumer staples', 'utilities', 'healthcare']
        cyclical_sectors = ['technology', 'consumer discretionary', 'real estate']
        volatile_sectors = ['energy', 'materials', 'cryptocurrencies']
        
        if any(s in sector for s in defensive_sectors):
            sector_adjustment = -10
        elif any(s in sector for s in cyclical_sectors):
            sector_adjustment = 0
        elif any(s in sector for s in volatile_sectors):
            sector_adjustment = 10
            
        # Calculate final score
        final_score = (beta_score * 0.4 + size_score * 0.4) + sector_adjustment
        return max(0, min(100, final_score))
    except:
        return 50

def calculate_financial_risk_score(symbol):
    """Calculate overall financial risk score where higher scores indicate higher risk"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # All scores now directly represent risk (0=low risk, 100=high risk)
        scores = {
            'Stock Volatility': calculate_stock_volatility(symbol),
            'Debt-to-Equity': calculate_de_ratio_score(info.get('debtToEquity', 0) / 100),
            'Interest Coverage': calculate_interest_coverage(info),
            'Credit Rating': get_credit_rating_score(info),
            'Liquidity': calculate_liquidity_ratio(info),
            'Macro Risk': get_macro_risk_score(info)
        }
        
        # Adjust weights to give more impact to high-risk indicators
        weights = {
            'Stock Volatility': 0.25,    # Increased from 0.10
            'Debt-to-Equity': 0.20,      # Increased from 0.10
            'Interest Coverage': 0.20,    # Increased from 0.10
            'Credit Rating': 0.15,       # Increased from 0.10
            'Liquidity': 0.10,           # Increased from 0.05
            'Macro Risk': 0.10           # Increased from 0.05
        }
        
        # Calculate weighted score with amplification for high risks
        total_score = 0
        for metric, score in scores.items():
            # Amplify high scores (risks) more than low scores
            adjusted_score = score
            if score > 66:  # High risk
                adjusted_score = score * 1.2  # Amplify high risks by 20%
            elif score < 33:  # Low risk
                adjusted_score = score * 0.8  # Reduce low risks by 20%
            
            total_score += adjusted_score * weights[metric]
        
        # Ensure the final score stays within 0-100
        final_score = min(100, max(0, total_score))
        
        return final_score, scores, weights
        
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
        mode="gauge",  # Removed '+number' to disable default number
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
    
    # Force number position to be centered with larger font
    fig.add_annotation(
        x=0.5,
        y=0.25,  # Moved down from 0.5 to 0.25
        text=f"{score:.0f}",
        showarrow=False,
        font=dict(size=80, color="white"),  # Increased font size from 50 to 80
        xanchor='center',
        yanchor='middle'
    )
    
    return fig

def create_component_chart(scores, weights):
    """Create a horizontal bar chart for risk components"""
    components = list(scores.keys())
    values = list(scores.values())
    
    # Create color scale based on scores - higher risk should be red
    colors = ['rgba(40, 167, 69, 0.8)' if v < 33 else  # Green for low risk (0-33)
              'rgba(255, 193, 7, 0.8)' if v < 66 else  # Yellow for medium risk (33-66)
              'rgba(220, 53, 69, 0.8)' for v in values]  # Red for high risk (66-100)
    
    fig = go.Figure(go.Bar(
        x=values,
        y=components,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(color='white'),
    ))
    
    fig.update_layout(
        title={'text': 'Risk Components', 'font': {'color': 'white', 'size': 24}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(
            title='Risk Score',
            range=[-5, 105],
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title='Component',
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        ),
        bargap=0.3
    )
    return fig

def calculate_contractual_risk_score(risks):
    """Calculate Contractual Risk Score (CRS) using weighted risk components"""
    try:
        model = palm.GenerativeModel('gemini-2.0-flash-exp')
        
        # Define weights for each risk category
        weights = {
            'Payment Risk': 0.15,      # w1 = 15% (Payment Structure)
            'Credit Support': 0.10,    # w2 = 10% (Credit Support)
            'Default Risk': 0.10,      # w3 = 10% (Default & Termination)
            'Performance Risk': 0.05,   # w4 = 5% (Performance Guarantees)
            'Regulatory Risk': 0.05,    # w5 = 5% (Regulatory Risk)
            'Market Exposure': 0.05     # w6 = 5% (Market Exposure)
        }
        
        # Modify prompt to be more explicit about scoring
        prompt = """Analyze each risk and categorize it into one of these categories with a risk score (0-100):

Scoring Guidelines:
- Low Risk (0-30): Minor issues that are easily mitigated
- Medium Risk (31-70): Significant concerns requiring attention
- High Risk (71-100): Critical issues that could severely impact the contract

Categories:
- Payment Risk: Risks related to payment structures, terms, and flexibility
- Credit Support: Risks related to credit guarantees and financial backing
- Default Risk: Risks related to default scenarios and legal protections
- Performance Risk: Risks related to operational performance and delivery guarantees
- Regulatory Risk: Risks related to regulatory compliance and protection
- Market Exposure: Risks related to pricing structures and market volatility

For each risk below, respond with exactly ONE WORD for the category and ONE NUMBER for the score, each on a new line without any prefixes or additional text.

Here are the risks to analyze:

"""
        # Rest of the prompt building remains the same...
        for i, risk in enumerate(risks, 1):
            prompt += f"""
Risk {i}:
{risk['description']}
Severity: {risk['severity']}
{risk['relevant_text']}

"""
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        
        # Initialize categorized risks
        categorized_risks = {
            "Payment Risk": {"risks": [], "score": 0},
            "Credit Support": {"risks": [], "score": 0},
            "Default Risk": {"risks": [], "score": 0},
            "Performance Risk": {"risks": [], "score": 0},
            "Regulatory Risk": {"risks": [], "score": 0},
            "Market Exposure": {"risks": [], "score": 0}
        }
        
        if response:
            lines = [line.strip() for line in response.text.split('\n') if line.strip()]
            
            # Process responses and calculate average scores for each category
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    try:
                        category = lines[i].strip()
                        score = int(lines[i + 1])
                        
                        # Map severity to score ranges to validate LLM output
                        severity_ranges = {
                            "LOW": (0, 30),
                            "MEDIUM": (31, 70),
                            "HIGH": (71, 100)
                        }
                        
                        # Convert single word category to full category name
                        category_map = {
                            "Payment": "Payment Risk",
                            "Credit": "Credit Support",
                            "Default": "Default Risk",
                            "Performance": "Performance Risk",
                            "Regulatory": "Regulatory Risk",
                            "Market": "Market Exposure"
                        }
                        
                        full_category = category_map.get(category, category)
                        
                        if full_category in categorized_risks:
                            idx = i // 2
                            if idx < len(risks):
                                # Adjust score based on original severity if needed
                                original_severity = risks[idx]['severity']
                                severity_range = severity_ranges.get(original_severity, (0, 100))
                                adjusted_score = min(max(score, severity_range[0]), severity_range[1])
                                
                                categorized_risks[full_category]["risks"].append({
                                    "original_risk": risks[idx],
                                    "score": adjusted_score
                                })
                    except ValueError as e:
                        st.warning(f"Error parsing response lines {i}/{i+1}: {e}")
                        continue
        
        # Calculate average scores for each category
        for category in categorized_risks:
            risks_in_category = categorized_risks[category]["risks"]
            if risks_in_category:
                avg_score = sum(r["score"] for r in risks_in_category) / len(risks_in_category)
                categorized_risks[category]["score"] = avg_score
        
        # Calculate CRS using the weighted formula
        crs = 0
        for category, data in categorized_risks.items():
            if data["risks"]:  # Only include categories that have risks
                crs += data["score"] * weights[category]
        
        # Scale CRS to 0-100
        crs = min(100, crs * 2)  # Multiply by 2 since weights sum to 0.5
        
        return crs, categorized_risks
        
    except Exception as e:
        st.error(f"Error in risk categorization: {e}")
        return None, None

def main():
    st.markdown('<h1 style="color: white;">🔍 AI PPA Financial Risk Dashboard</h1>', unsafe_allow_html=True)
    
    # Stock Analysis Section First
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        symbol = st.text_input(
            label="Stock Symbol",
            value="FSLR",
            placeholder="Enter Company Stock Symbol",
            label_visibility="collapsed"
        )
        fetch_button = st.button("🔄 Analyze", use_container_width=True)

    tab1, tab2 = st.tabs([
        "Market-Based Financial Risk (MFRS)",
        "PPA Contract Analysis (CRS)"
    ])

    with tab2:
        st.markdown('<h2 class="upload-header">📄 Upload PPA Document for Contractual Risk Analysis (CRS)</h2>', 
                unsafe_allow_html=True)
    
        # File uploader with custom text
        uploaded_file = st.file_uploader(
            "Upload PPA PDF",
            type=['pdf'],
            help="Limit 200MB per file • PDF",
            label_visibility="visible"
        )
        
        if uploaded_file is not None:
            # Display file details
            st.write(f"{uploaded_file.name} • {round(uploaded_file.size/1e6, 1)}MB")
            
            # Add process button
            if st.button("Process Document", type="primary"):
                with st.spinner("Analyzing PPA document..."):
                    # Add your PPA document analysis logic here
                    st.success("Document analysis complete!")

    
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
            st.markdown('<div class="section-header">💰 Market-Based Financial Risk Score (MFRS)</div>', unsafe_allow_html=True)
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
            
            # Contract Risk Analysis Section
            st.markdown('<div class="section-header">📄 Contract Risk Analysis</div>', unsafe_allow_html=True)
            try:
                # Read the risk_report.json file
                with open('risk_report.json', 'r') as f:
                    risk_report = json.load(f)
                
                # Extract financial risks from the report
                financial_risks = risk_report['risks_by_category']['financial']['risks']
                
                # Calculate contractual risk score
                crs, categorized_risks = calculate_contractual_risk_score(financial_risks)
                
                if crs is not None and categorized_risks:
                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Overall risk gauge
                        st.plotly_chart(
                            create_risk_gauge(
                                crs,
                                "Contractual Risk Score (CRS)"
                            ),
                            use_container_width=True
                        )
                    
                    with col2:
                        # Component breakdown
                        scores = {category: data['score'] for category, data in categorized_risks.items()}
                        st.plotly_chart(
                            create_component_chart(scores, weights),
                            use_container_width=True
                        )
            
            except FileNotFoundError:
                st.warning("No contract risk report found. Please analyze a document first.")
            except Exception as e:
                st.error(f"Error analyzing contract risks: {e}")
            
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
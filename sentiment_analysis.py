import requests
import google.generativeai as palm
from absl import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging and API Keys
logging.use_absl_handler()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
palm.configure(api_key=os.getenv('PALM_API_KEY'))

# Add error checking for environment variables
if not NEWS_API_KEY or not os.getenv('PALM_API_KEY'):
    raise ValueError("Missing required API keys in .env file")

# Step 1: Fetch News Articles Using NewsAPI
def fetch_articles(query, page_size=10):
    """
    Fetches news articles related to the query using NewsAPI.
    Args:
        query (str): The search term.
        page_size (int): Number of articles to fetch.
    Returns:
        list: A list of article titles and descriptions.
    """
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"NewsAPI Error: {response.status_code} - {response.text}")
    
    articles = response.json().get("articles", [])
    return [(article["title"], article["description"]) for article in articles]

# Step 2: Perform Sentiment Analysis Using Google Gemini
def analyze_sentiment(text):
    """
    Analyzes sentiment using Google Gemini (Generative AI).
    Args:
        text (str): The input text for sentiment analysis.
    Returns:
        str: Sentiment label (Positive, Neutral, Negative).
    """
    # Configure the model
    model = palm.GenerativeModel('gemini-2.0-flash-exp')
    
    # Send the text to Gemini for sentiment analysis
    response = model.generate_content(
        f"Analyze the sentiment of the following text and classify it as Positive, Neutral, or Negative:\n\n{text}\n\nSentiment:"
    )
    
    # Extract the result from the response
    if response:
        return response.text.strip()
    else:
        return "Error: Unable to generate a response"

def analyze_sentiment_batch(texts):
    """
    Analyzes sentiment for multiple texts at once using Google Gemini.
    Args:
        texts (list): List of texts to analyze.
    Returns:
        list: List of sentiment labels.
    """
    model = palm.GenerativeModel('gemini-pro')
    
    # Create a single prompt for all texts
    combined_prompt = "Analyze the sentiment of each of the following texts. For each text, respond with ONLY ONE WORD: either Positive, Neutral, or Negative. Provide the answers in order, one per line:\n\n"
    for i, text in enumerate(texts, 1):
        combined_prompt += f"{i}. {text}\n"
    
    response = model.generate_content(combined_prompt)
    if not response:
        return ["Error"] * len(texts)
    
    # Parse the response into individual sentiments
    sentiments = [line.strip() for line in response.text.split('\n') if line.strip()]
    return sentiments[:len(texts)]  # Ensure we return exactly the right number of sentiments

# Step 3: Main Function
def main():
    # Initialize logging
    logging.set_verbosity(logging.INFO)
    
    query = "First Solar"
    print("Fetching articles...")
    try:
        articles = fetch_articles(query, page_size=10)
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return
    
    print("Analyzing sentiment...")
    combined_texts = [f"{title}. {desc}" if desc else title for title, desc in articles]
    sentiments = analyze_sentiment_batch(combined_texts)
    
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for sentiment in sentiments:
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
        else:
            print(f"Unknown sentiment: {sentiment}")

    print("\nSentiment Summary:")
    total_articles = sum(sentiment_counts.values())
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_articles) * 100
        print(f"{sentiment}: {count} articles ({percentage:.1f}%)")

    # Cleanup Palm client at the end
    try:
        palm.clear()
    except:
        pass

if __name__ == "__main__":
    main()
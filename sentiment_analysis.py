import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']


def fetch_news(ticker):
    api_key = "d9fc775c1a854d028b3c8a4a443d4a6e"  # Replace with your actual API key
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("articles", [])
        return [article['title'] for article in articles]
    else:
        print("Failed to fetch news")
        return []

def get_stock_sentiment(ticker):
    headlines = fetch_news(ticker)
    if not headlines:
        return 0  # Default to neutral if no news available

    scores = [get_sentiment_score(headline) for headline in headlines]
    avg_sentiment = sum(scores) / len(scores)
    return avg_sentiment

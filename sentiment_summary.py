import feedparser
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

labels = ["Positive", "Negative", "Neutral"]

def fetch_news(query, num_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    news_items = feed.entries[:num_articles]

    articles = []

    for item in news_items:
        title = item.title
        link = item.link
        published = item.published
        content = fetch_article_content(link)

        articles.append({
            "title": title,
            "link": link,
            "published": published,
            "content": content
        })

    return articles

def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraph = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraph])
        return content.strip()
    
    except:
        return "Content not retrieved"
    
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    polarity = scores["compound"]

    if polarity > 0.05:
        sentiment = "Positive"

    elif polarity < -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return polarity, sentiment

def summarize_sentiments(articles):
    summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    for article in articles:
        _, sentiment = analyze_sentiment(article["title"])
        summary[sentiment] += 1

    total = len(articles)
    print("\n--- Market Sentiment Summary ---")
    print(f"Total articles analyzed: {total}")
    for sentiment, count in summary.items():
        percent = (count / total) * 100
        print(f"{sentiment}: {count} ({percent:.2f}%)")

if __name__ == "__main__":
    queries = [
        "gold market",
        "gold price",
        "gold news",
        "gold trends",
        "gold forecast",
    ]

    num_per_query = 10
    articles = []
    for query in queries:
        response = fetch_news(query, num_per_query)
        articles.extend(response)

    for idx, article in enumerate(articles, 1):
        polarity, sentiment = analyze_sentiment(article['title'])
        print(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})\n")

    summarize_sentiments(articles=articles)

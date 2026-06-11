import urllib.request
import xml.etree.ElementTree as ET
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the NLP vocabulary dictionary (only runs the first time)
nltk.download('vader_lexicon', quiet=True)


def analyze_live_news(ticker_symbol, company_name):
    print(f"📡 Fetching live news for {company_name} ({ticker_symbol}) via Google News...\n")

    # Initialize the NLP Brain
    analyzer = SentimentIntensityAnalyzer()

    # Format the query for Google News (e.g., "Reliance+Industries+stock")
    query = company_name.replace(' ', '+') + '+stock'
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        # We use a "User-Agent" to tell Google we are a standard web browser, preventing blocks
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
    except Exception as e:
        print(f"Failed to fetch news: {e}")
        return

    total_score = 0
    article_count = 0

    print("📰 --- LATEST HEADLINES & AI SCORES --- 📰")

    # Find the top 10 articles in the XML feed
    for item in root.findall('.//item')[:10]:
        headline = item.find('title').text

        # The AI reads the headline and generates a "Compound" score between -1.0 and 1.0
        sentiment = analyzer.polarity_scores(headline)
        score = sentiment['compound']

        total_score += score
        article_count += 1

        # Color-code the output based on the score
        if score > 0.1:
            emoji = "🟢 BULLISH"
        elif score < -0.1:
            emoji = "🔴 BEARISH"
        else:
            emoji = "⚪ NEUTRAL"

        print(f"{emoji} [{score:.2f}]: {headline}")

    # Calculate the final average sentiment for the day
    if article_count > 0:
        avg_score = total_score / article_count
        print("\n=============================================")
        print(f"🧠 OVERALL MARKET SENTIMENT SCORE: {avg_score:.2f} / 1.00")
        if avg_score > 0.2:
            print("Action: AI sees positive news. Boost 'BUY' probability.")
        elif avg_score < -0.2:
            print("Action: AI sees negative news. Boost 'SELL' probability.")
        else:
            print("Action: News is mixed/neutral. Rely strictly on Technical Chart.")
        print("=============================================")
    else:
        print("No articles found.")


# --- CONFIGURATION ---
TICKER = "RELIANCE.NS"
COMPANY_NAME = "Reliance Industries"

# Run the NLP Engine
analyze_live_news(TICKER, COMPANY_NAME)
# Install required packages
!pip install kagglehub pandas matplotlib seaborn wordcloud nltk

# === 1. Import libraries ===
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# === 2. Download dataset using kagglehub ===
path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
print("Dataset downloaded to:", path)

# === 3. Load the CSV file ===
csv_path = os.path.join(path, "Tweets.csv")
df = pd.read_csv(csv_path)

# Display the first few rows
print("Sample data:")
print(df[['airline_sentiment', 'text']].head())

# === 4. Sentiment Analysis with VADER ===
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Optional: rename columns for consistency
df = df.rename(columns={"airline_sentiment": "label", "text": "text"})

# Apply VADER sentiment scoring
def analyze_sentiment(df):
    sentiments = df['text'].apply(sia.polarity_scores).tolist()
    sentiment_df = pd.DataFrame(sentiments)
    df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
    df['vader_sentiment'] = df['compound'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
    )
    return df

df = analyze_sentiment(df)

# === 5. Visualize VADER Sentiment Distribution ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='vader_sentiment', palette='Set2')
plt.title("VADER Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.show()

# === 6. Word Clouds for Each Sentiment ===
def plot_wordcloud(df, sentiment_label):
    text = " ".join(df[df['vader_sentiment'] == sentiment_label]['text'].dropna())
    if text.strip():
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud — {sentiment_label.capitalize()} Tweets')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No tweets found for sentiment: {sentiment_label}")

for sentiment in ['positive', 'negative', 'neutral']:
    plot_wordcloud(df, sentiment)

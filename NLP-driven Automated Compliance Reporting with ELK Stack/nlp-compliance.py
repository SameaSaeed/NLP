import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Sample log file
with open("logs/system.log", "r") as f:
    logs = f.readlines()

# Clean and structure logs
data = []
for line in logs:
    timestamp = re.search(r"\[(.*?)\]", line).group(1)
    level = re.search(r"\] (\w+):", line).group(1)
    message = re.search(r": (.*)", line).group(1)
    data.append((timestamp, level, message))

df = pd.DataFrame(data, columns=["timestamp", "level", "message"])

# NLP sentiment analysis
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["message"].apply(lambda x: sia.polarity_scores(x)["compound"])

# Flag risky logs
df["compliance_risk"] = df["sentiment"] < -0.2
print(df[["timestamp", "level", "message", "compliance_risk"]])

df.to_csv("compliance_report.csv", index=False)
print("\nâ Compliance report generated: compliance_report.csv")
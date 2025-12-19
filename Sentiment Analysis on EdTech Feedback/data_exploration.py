import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/edtech_feedback.csv')

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATASET INFO ===")
print(df.info())

print("\n=== SENTIMENT DISTRIBUTION ===")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)
print(f"Positive feedback: {sentiment_counts[1]} ({sentiment_counts[1]/len(df)*100:.1f}%)")
print(f"Negative feedback: {sentiment_counts[0]} ({sentiment_counts[0]/len(df)*100:.1f}%)")

# Visualize sentiment distribution
plt.figure(figsize=(8, 6))
sentiment_labels = ['Negative', 'Positive']
colors = ['#ff6b6b', '#4ecdc4']

plt.subplot(1, 2, 1)
plt.pie(sentiment_counts.values, labels=sentiment_labels, autopct='%1.1f%%', colors=colors)
plt.title('Sentiment Distribution')

plt.subplot(1, 2, 2)
plt.bar(sentiment_labels, sentiment_counts.values, color=colors)
plt.title('Sentiment Counts')
plt.ylabel('Number of Feedback')

plt.tight_layout()
plt.savefig('results/sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== SAMPLE FEEDBACK BY SENTIMENT ===")
print("\nPositive Feedback Examples:")
positive_samples = df[df['sentiment'] == 1]['feedback'].head(3)
for i, feedback in enumerate(positive_samples, 1):
    print(f"{i}. {feedback}")

print("\nNegative Feedback Examples:")
negative_samples = df[df['sentiment'] == 0]['feedback'].head(3)
for i, feedback in enumerate(negative_samples, 1):
    print(f"{i}. {feedback}")

# Text length analysis
df['text_length'] = df['feedback'].str.len()
print(f"\n=== TEXT LENGTH STATISTICS ===")
print(f"Average text length: {df['text_length'].mean():.1f} characters")
print(f"Minimum text length: {df['text_length'].min()} characters")
print(f"Maximum text length: {df['text_length'].max()} characters")

# Text length by sentiment
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
df[df['sentiment'] == 1]['text_length'].hist(bins=15, alpha=0.7, label='Positive', color='#4ecdc4')
df[df['sentiment'] == 0]['text_length'].hist(bins=15, alpha=0.7, label='Negative', color='#ff6b6b')
plt.xlabel('Text Length (characters)')
plt.ylabel('Frequency')
plt.title('Text Length Distribution by Sentiment')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='sentiment', y='text_length')
plt.xlabel('Sentiment (0=Negative, 1=Positive)')
plt.ylabel('Text Length (characters)')
plt.title('Text Length by Sentiment')

plt.tight_layout()
plt.savefig('results/text_length_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
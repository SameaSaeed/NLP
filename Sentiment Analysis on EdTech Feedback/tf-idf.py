import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load preprocessed training data
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

print("=== TF-IDF VECTORIZATION ===")

# Initialize TF-IDF Vectorizer
# max_features: limit vocabulary size to most frequent words
# min_df: ignore terms that appear in less than min_df documents
# max_df: ignore terms that appear in more than max_df fraction of documents
# ngram_range: consider both single words and bigrams

tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,      # Use top 1000 most frequent words
    min_df=2,               # Word must appear in at least 2 documents
    max_df=0.8,             # Ignore words that appear in more than 80% of documents
    ngram_range=(1, 2),     # Use both unigrams and bigrams
    stop_words='english'    # Remove common English stop words
)

# Fit the vectorizer on training data and transform both training and test data
print("Fitting TF-IDF vectorizer on training data...")
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['feedback'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['feedback'])

print(f"Training data shape after TF-IDF: {X_train_tfidf.shape}")
print(f"Testing data shape after TF-IDF: {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# Display some feature names (words/phrases)
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"\nSample features (first 20): {feature_names[:20]}")
print(f"Sample features (last 20): {feature_names[-20:]}")

# Analyze most important features
# Get the mean TF-IDF score for each feature across all documents
mean_tfidf_scores = np.array(X_train_tfidf.mean(axis=0)).flatten()
feature_importance = list(zip(feature_names, mean_tfidf_scores))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print(f"\n=== TOP 20 MOST IMPORTANT FEATURES ===")
for i, (feature, score) in enumerate(feature_importance[:20], 1):
    print(f"{i:2d}. {feature:<20} (TF-IDF: {score:.4f})")

# Analyze features by sentiment
print(f"\n=== FEATURE ANALYSIS BY SENTIMENT ===")

# Separate positive and negative samples
positive_mask = train_df['sentiment'] == 1
negative_mask = train_df['sentiment'] == 0

X_train_positive = X_train_tfidf[positive_mask]
X_train_negative = X_train_tfidf[negative_mask]

# Calculate mean TF-IDF scores for each sentiment
positive_mean = np.array(X_train_positive.mean(axis=0)).flatten()
negative_mean = np.array(X_train_negative.mean(axis=0)).flatten()

# Find features that are more characteristic of each sentiment
positive_features = []
negative_features = []

for i, feature in enumerate(feature_names):
    pos_score = positive_mean[i]
    neg_score = negative_mean[i]
    
    if pos_score > neg_score and pos_score > 0.01:
        positive_features.append((feature, pos_score, pos_score - neg_score))
    elif neg_score > pos_score and neg_score > 0.01:
        negative_features.append((feature, neg_score, neg_score - pos_score))

# Sort by difference in scores
positive_features.sort(key=lambda x: x[2], reverse=True)
negative_features.sort(key=lambda x: x[2], reverse=True)

print(f"\nTop 10 features associated with POSITIVE sentiment:")
for i, (feature, score, diff) in enumerate(positive_features[:10], 1):
    print(f"{i:2d}. {feature:<20} (Score: {score:.4f}, Diff: +{diff:.4f})")

print(f"\nTop 10 features associated with NEGATIVE sentiment:")
for i, (feature, score, diff) in enumerate(negative_features[:10], 1):
    print(f"{i:2d}. {feature:<20} (Score: {score:.4f}, Diff: +{diff:.4f})")

# Save the vectorizer and transformed data for later use
import joblib
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(X_train_tfidf, 'models/X_train_tfidf.pkl')
joblib.dump(X_test_tfidf, 'models/X_test_tfidf.pkl')

print(f"\nTF-IDF vectorizer and transformed data saved to models/ directory")
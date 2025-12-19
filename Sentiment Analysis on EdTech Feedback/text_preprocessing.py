import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

def preprocess_text(text):
    """
    Preprocess text data for sentiment analysis
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (if any)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Load the dataset
df = pd.read_csv('data/edtech_feedback.csv')

print("=== TEXT PREPROCESSING ===")
print("Original text example:")
print(df['feedback'].iloc[0])

# Apply preprocessing
df['processed_feedback'] = df['feedback'].apply(preprocess_text)

print("\nProcessed text example:")
print(df['processed_feedback'].iloc[0])

# Split the data into training and testing sets
X = df['processed_feedback']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n=== DATA SPLIT ===")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print(f"Training positive samples: {sum(y_train)}")
print(f"Training negative samples: {len(y_train) - sum(y_train)}")
print(f"Testing positive samples: {sum(y_test)}")
print(f"Testing negative samples: {len(y_test) - sum(y_test)}")

# Save preprocessed data
train_df = pd.DataFrame({
    'feedback': X_train,
    'sentiment': y_train
})
test_df = pd.DataFrame({
    'feedback': X_test,
    'sentiment': y_test
})

train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

print("\nPreprocessed data saved to:")
print("- data/train_data.csv")
print("- data/test_data.csv")
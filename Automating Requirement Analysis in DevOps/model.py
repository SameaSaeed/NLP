# python -m spacy download en_core_web_sm

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample user stories
data = {
    'user_story': [
        'As a user, I want to be able to log in with Google so that I can access the system quickly.',
        'As an admin, I need to manage users so that I can assign appropriate permissions.',
        'As a user, I want to reset my password in case I forget it.',
        'As a developer, I need to implement a caching system for faster data retrieval.',
        'As a user, I want to search for products so that I can find the items I need.'
    ],
    'category': ['Authentication', 'User Management', 'Authentication', 'Performance', 'Search']
}

# Create a DataFrame
df = pd.DataFrame(data)
print(df)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess the user stories
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]  # Tokenize
    pos_tags = [token.pos_ for token in doc]  # POS tagging
    return tokens, pos_tags

# Apply preprocessing to each user story
df[['tokens', 'pos_tags']] = df['user_story'].apply(lambda x: pd.Series(preprocess_text(x)))
print(df[['user_story', 'tokens', 'pos_tags']])


# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the 'user_story' column
X = vectorizer.fit_transform(df['user_story'])

print(X.shape)  # Check the shape of the feature matrix

# Split the data into training and testing sets
y = df['category']  # Target labels (categories)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
plt.title('Confusion Matrix for User Story Categorization')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
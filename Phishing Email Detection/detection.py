import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Simulated email dataset
data = {
    'email': [
        'Dear customer, you have won a $1000 gift card. Click here to claim it!',
        'Your invoice for this month is attached. Please review it and make the payment.',
        'Urgent: Your bank account has been compromised. Click here to secure it immediately.',
        'We are reaching out to confirm your appointment for tomorrow.',
        'Congratulations! You have a package waiting for pickup. Click here to track.',
        'Reminder: Your subscription to XYZ service will renew tomorrow.'
    ],
    'label': ['phishing', 'legitimate', 'phishing', 'legitimate', 'phishing', 'legitimate']  # phishing or legitimate
}

# Create a DataFrame
df = pd.DataFrame(data)

print(df)

# Convert emails to lowercase and clean text (remove punctuation, etc.)
def clean_text(text):
    text = text.lower()
    return text

df['cleaned_email'] = df['email'].apply(clean_text)

# Prepare feature (X) and target (y)
X = df['cleaned_email']
y = df['label']

# Apply TF-IDF vectorizer to transform email text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Shape of the resulting feature matrix
print(X_vectorized.shape)  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

print(f"Training Set: {X_train.shape}, Test Set: {X_test.shape}")

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
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['legitimate', 'phishing'], yticklabels=['legitimate', 'phishing'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
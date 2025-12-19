import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=== LOGISTIC REGRESSION CLASSIFIER ===")

# Load preprocessed data
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')

# Load TF-IDF transformed data
X_train_tfidf = joblib.load('models/X_train_tfidf.pkl')
X_test_tfidf = joblib.load('models/X_test_tfidf.pkl')

y_train = train_df['sentiment']
y_test = test_df['sentiment']

print(f"Training data shape: {X_train_tfidf.shape}")
print(f"Testing data shape: {X_test_tfidf.shape}")

# Initialize Logistic Regression classifier
# C: regularization strength (smaller values = stronger regularization)
# max_iter: maximum number of iterations for convergence
# random_state: for reproducible results

logistic_classifier = LogisticRegression(
    C=1.0,                    # Regularization strength
    max_iter=1000,           # Maximum iterations
    random_state=42,         # For reproducible results
    solver='liblinear'       # Solver suitable for small datasets
)

print("\nTraining Logistic Regression classifier...")
logistic_classifier.fit(X_train_tfidf, y_train)

# Make predictions on both training and test sets
y_train_pred = logistic_classifier.predict(X_train_tfidf)
y_test_pred = logistic_classifier.predict(X_test_tfidf)

# Get prediction probabilities for test set
y_test_proba = logistic_classifier.predict_proba(X_test_tfidf)

print("Training completed successfully!")

# Calculate performance metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Testing Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"Testing Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"Testing F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")

# Detailed classification report
print(f"\n=== DETAILED CLASSIFICATION REPORT ===")
class_names = ['Negative', 'Positive']
print(classification_report(y_test, y_test_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n=== CONFUSION MATRIX ===")
print("Actual vs Predicted:")
print(f"                Predicted")
print(f"Actual    Negative  Positive")
print(f"Negative     {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"Positive     {cm[1,0]:3d}      {cm[1,1]:3d}")

# Visualize Confusion Matrix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

# Plot prediction probabilities distribution
plt.subplot(1, 2, 2)
positive_probs = y_test_proba[:, 1]  # Probabilities for positive class
plt.hist(positive_probs[y_test == 0], bins=20, alpha=0.7, label='Negative Actual', color='red')
plt.hist(positive_probs[y_test == 1], bins=20, alpha=0.7, label='Positive Actual', color='green')
plt.xlabel('Predicted Probability of Positive Sentiment')
plt.ylabel('Frequency')
plt.title('Prediction Probability Distribution')
plt.legend()
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')

plt.tight_layout()
plt.savefig('results/model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the trained model
joblib.dump(logistic_classifier, 'models/logistic_classifier.pkl')
print(f"\nTrained model saved to models/logistic_classifier.pkl")

# Analyze feature importance (coefficients)
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
feature_names = tfidf_vectorizer.get_feature_names_out()
coefficients = logistic_classifier.coef_[0]

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

print(f"\n=== TOP 15 MOST IMPORTANT FEATURES ===")
print("(Positive coefficients indicate positive sentiment, negative indicate negative sentiment)")
for i, row in feature_importance_df.head(15).iterrows():
    sentiment_indicator = "→ POSITIVE" if row['coefficient'] > 0 else "→ NEGATIVE"
    print(f"{feature_importance_df.index.get_loc(i)+1:2d}. {row['feature']:<20} "
          f"(Coef: {row['coefficient']:+.4f}) {sentiment_indicator}")

# Save feature importance
feature_importance_df.to_csv('results/feature_importance.csv', index=False)
print(f"\nFeature importance saved to results/feature_importance.csv")
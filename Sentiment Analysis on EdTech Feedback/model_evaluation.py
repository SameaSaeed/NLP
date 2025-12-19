import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

print("=== COMPREHENSIVE MODEL EVALUATION ===")

# Load test data and model
test_df = pd.read_csv('data/test_data.csv')
X_test_tfidf = joblib.load('models/X_test_tfidf.pkl')
logistic_classifier = joblib.load('models/logistic_classifier.pkl')

y_test = test_df['sentiment']
y_test_pred = logistic_classifier.predict(X_test_tfidf)
y_test_proba = logistic_classifier.predict_proba(X_test_tfidf)

# Calculate all metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Calculate metrics for each class
precision_per_class = precision_score(y_test, y_test_pred, average=None)
recall_per_class = recall_score(y_test, y_test_pred, average=None)
f1_per_class = f1_score(y_test, y_test_pred, average=None)

print(f"=== OVERALL PERFORMANCE METRICS ===")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print(f"\n=== PER-CLASS PERFORMANCE ===")
class_names = ['Negative', 'Positive']
for i, class_name in enumerate(class_names):
    print(f"{class_name} Class:")
    print(f"  Precision: {precision_per_class[i]:.4f} ({precision_per_class[i]*100:.2f}%)")
    print(f"  Recall:    {recall_per_class[i]:.4f} ({recall_per_class[i]*100:.2f}%)")
    print(f"  F1-Score:  {f1_per_class[i]:.4f} ({f1_per_class[i]*100:.2f}%)")

# Confusion Matrix Analysis
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n=== CONFUSION MATRIX BREAKDOWN ===")
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")

# Calculate additional metrics
specificity = tn / (tn + fp)  # True Negative Rate
sensitivity = tp / (tp + fn)  # Same as recall, True Positive Rate

print(f"\n=== ADDITIONAL METRICS ===")
print(f"Sensitivity (True Positive Rate):  {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"Specificity (True Negative Rate):  {specificity:.4f} ({specificity*100:.2f}%)")
print(f"False Positive Rate: {fp/(fp+tn):.4f} ({fp/(fp+tn)*100:.2f}%)")
print(f"False Negative Rate: {fn/(fn+tp):.4f} ({fn/(fn+tp)*100:.2f}%)")

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
roc_auc = auc(fpr, tpr)

print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix')
axes[0,0].set_xlabel('Predicted Label')
axes[0,0].set_ylabel('Actual Label')

# 2. ROC Curve
axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[0,1].set_xlim([0.0, 1.0])
axes[0,1].set_ylim([0.0, 1.05])
axes[0,1].set_xlabel('False Positive Rate')
axes[0,1].set_ylabel('True Positive Rate')
axes[0,1].set_title('ROC Curve')
axes[0,1].legend(loc="lower right")
axes[0,1].grid(True)

# 3. Performance Metrics Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = axes[1,0].bar(metrics_names, metrics_values, color=colors)
axes[1,0].set_ylim([0, 1])
axes[1,0].set_ylabel('Score')
axes[1,0].set_title('Performance Metrics')
axes[1,0].grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

# 4. Prediction Probability Distribution
positive_probs = y_test_proba[:, 1]
axes[1,1].hist(positive_probs[y_test == 0], bins=15, alpha=0.7, 
               label='Actual Negative', color='red', density=True)
axes[1,1].hist(positive_probs[y_test == 1], bins=15, alpha=0.7, 
               label='Actual Positive', color='green', density=True)
axes[1,1].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
axes[1,1].set_xlabel('Predicted Probability of Positive Sentiment')
axes[1,1].set_ylabel('Density')
axes[1,1].set_title('Prediction's Probability Distribution')
axes[1,1].legend()
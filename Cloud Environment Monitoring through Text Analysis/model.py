# pip install spacy pandas matplotlib
# python -m spacy download en_core_web_sm

import spacy
import pandas as pd
import matplotlib.pyplot as plt

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample log data (replace this with your actual log data)
df_logs = pd.DataFrame({
    'log_message': [
        "ERROR: Connection to server failed in California.",
        "WARNING: Disk space running low at Amazon.",
        "User logged in from Germany.",
        "ERROR: Authentication failed at Microsoft.",
        "WARNING: API response time exceeded threshold.",
        "INFO: Scheduled backup completed successfully."
    ]
})

# Function to tokenize logs and remove stopwords/punctuation
def process_log(log_message):
    doc = nlp(log_message)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Apply token processing
df_logs['processed_log'] = df_logs['log_message'].apply(process_log)

# Function to extract entities based on log level context
def extract_entities(log_message):
    doc = nlp(log_message)
    entities = {"errors": [], "warnings": []}
    
    if 'error' in log_message.lower():
        entities["errors"].extend([ent.text for ent in doc.ents])
    elif 'warning' in log_message.lower():
        entities["warnings"].extend([ent.text for ent in doc.ents])
    
    return entities

# Apply entity extraction
df_logs['log_entities'] = df_logs['log_message'].apply(extract_entities)

# Count occurrences of error/warning keywords
df_logs['error_count'] = df_logs['log_message'].str.lower().str.count('error')
df_logs['warning_count'] = df_logs['log_message'].str.lower().str.count('warning')

# Simple anomaly detection rule (threshold = 1)
threshold = 1
df_logs['anomaly'] = (df_logs['error_count'] > threshold) | (df_logs['warning_count'] > threshold)

# Optional: Extract log level
def get_log_level(log):
    log = log.lower()
    if 'error' in log:
        return 'ERROR'
    elif 'warning' in log:
        return 'WARNING'
    else:
        return 'INFO'

df_logs['log_level'] = df_logs['log_message'].apply(get_log_level)

# Print the full processed DataFrame
print(df_logs)

# Plot the anomalies
plt.figure(figsize=(8, 6))
ax = df_logs['anomaly'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Anomaly Detection in Cloud Logs')
plt.xlabel('Anomaly Status')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['Normal', 'Anomalous'], rotation=0)

# Add count labels on bars
for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.show()

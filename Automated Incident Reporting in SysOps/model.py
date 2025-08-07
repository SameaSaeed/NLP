# python -m spacy download en_core_web_sm

import pandas as pd
import spacy
import matplotlib.pyplot as plt

# Simulated incident logs
incident_logs = [
    "2025-07-01 08:23:45 - ERROR - System crash on server 'web-server-01' due to memory overflow.",
    "2025-07-01 09:15:32 - WARNING - CPU usage exceeds 85% on server 'database-server-03'.",
    "2025-07-01 10:03:11 - INFO - Successful login attempt by user 'admin' on server 'auth-server-01'.",
    "2025-07-01 11:42:59 - ERROR - Security breach detected in 'api-server-04', unauthorized access attempt.",
    "2025-07-01 12:12:15 - INFO - Backup completed successfully on 'backup-server-01'.",
    "2025-07-01 13:25:33 - ERROR - Database connection failure on 'db-server-02'."
]

# Create a DataFrame
df_incidents = pd.DataFrame(incident_logs, columns=['log_message'])

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract key information from incident logs
def extract_incident_info(log_message):
    doc = nlp(log_message)

    # Extract incident type (ERROR, WARNING, INFO)
    incident_type = None
    if 'ERROR' in log_message:
        incident_type = 'ERROR'
    elif 'WARNING' in log_message:
        incident_type = 'WARNING'
    elif 'INFO' in log_message:
        incident_type = 'INFO'

    # Extract server/system name (e.g., 'web-server-01')
    server = None
    for ent in doc.ents:
        if ent.label_ == "ORG":  # Assuming the server name is considered as an organization
            server = ent.text

    # Extract description of the issue
    issue_description = None
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            issue_description = ent.text

    # Extract date and time
    date_time = log_message.split(' - ')[0]

    return date_time, incident_type, server, issue_description

# Apply extraction to each incident log
df_incidents[['date_time', 'incident_type', 'server', 'issue_description']] = df_incidents['log_message'].apply(
    lambda x: pd.Series(extract_incident_info(x))
)

def categorize_incident(issue_description):
    if 'security' in issue_description.lower() or 'unauthorized' in issue_description.lower():
        return 'Security'
    elif 'cpu' in issue_description.lower() or 'memory' in issue_description.lower():
        return 'Performance'
    elif 'database' in issue_description.lower():
        return 'System'
    else:
        return 'General'

# Categorize each incident based on the issue description
df_incidents['incident_category'] = df_incidents['issue_description'].apply(categorize_incident)

print(df_incidents)

# Plot the incident categories
incident_category_counts = df_incidents['incident_category'].value_counts()

plt.figure(figsize=(8, 6))
incident_category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Incident Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re

# Step 1: Read and parse the Python-style list from threat_report.txt
with open('threat_report.txt', 'r') as file:
    content = file.read()

# Extract the list from the file content using regex and ast.literal_eval
match = re.search(r'threat_reports\s*=\s*(\[[\s\S]*?\])', content)
if not match:
    raise ValueError("Could not find threat_reports list in the file.")

threat_reports = ast.literal_eval(match.group(1))

# Step 2: Function to tokenize the text using OpenNLP
def tokenize_text(text):
    with open("temp.txt", "w") as file:
        file.write(text)

    result = subprocess.run(
        ['java', '-cp', '/opt/apache-opennlp-1.9.3/lib/*', 'opennlp.tools.cmdline.TokenizerTool',
         'en-token.bin', '<', 'temp.txt'],
        capture_output=True, text=True, shell=True
    )
    tokens = result.stdout.strip().split()
    return tokens

# Step 3: Function to get POS tags and Named Entities
def pos_tag_and_ner(text):
    with open("temp.txt", "w") as file:
        file.write(text)

    pos_result = subprocess.run(
        ['java', '-cp', '/opt/apache-opennlp-1.9.3/lib/*', 'opennlp.tools.cmdline.POSTaggerTool',
         'en-pos-maxent.bin', '<', 'temp.txt'],
        capture_output=True, text=True, shell=True
    )
    pos_tags = pos_result.stdout.strip().splitlines()

    ner_result = subprocess.run(
        ['java', '-cp', '/opt/apache-opennlp-1.9.3/lib/*', 'opennlp.tools.cmdline.NameFinderTool',
         'en-ner-person.bin', '<', 'temp.txt'],
        capture_output=True, text=True, shell=True
    )
    named_entities = ner_result.stdout.strip().splitlines()

    return pos_tags, named_entities

# Step 4: Function to categorize threat
def categorize_report(text):
    text_lower = text.lower()
    if "malware" in text_lower:
        return "Malware"
    elif "phishing" in text_lower:
        return "Phishing"
    elif "ransomware" in text_lower:
        return "Ransomware"
    elif "vulnerability" in text_lower:
        return "Vulnerability"
    else:
        return "General Threat"

# --- Run Analysis on All Reports ---
all_categories = []

for i, report in enumerate(threat_reports):
    print(f"\nReport {i+1}: {report}")
    
    tokens = tokenize_text(report)
    print("Tokens:", tokens)
    
    pos_tags, named_entities = pos_tag_and_ner(report)
    print("POS Tags:", pos_tags)
    print("Named Entities:", named_entities)
    
    category = categorize_report(report)
    print("Category:", category)
    all_categories.append(category)

# Step 5: Plot threat category distribution
category_counts = pd.Series(all_categories).value_counts()

category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Threat Report Categories Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

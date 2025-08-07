# python -m spacy download en_core_web_sm

from collections import Counter
import re
import spacy
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Read documentation content from Markdown file
with open("documentation.md", "r", encoding="utf-8") as file:
    documentation = file.read()

required_sections = ["Introduction", "Setup", "Best Practices", "Troubleshooting"]

def check_completeness(documentation, required_sections):
    missing_sections = [section for section in required_sections if section.lower() not in documentation.lower()]
    return missing_sections

missing_sections = check_completeness(documentation, required_sections)
print("Missing Sections:", missing_sections)

def check_consistency(documentation):
    words = re.findall(r'\b\w+\b', documentation.lower())
    word_counts = Counter(words)
    repeated_words = {word: count for word, count in word_counts.items() if count > 1}
    return repeated_words

repeated_words = check_consistency(documentation)
print("Repeated Words:", repeated_words)

def check_clarity(documentation):
    doc = nlp(documentation)
    long_sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 20]
    entities = [ent.text for ent in doc.ents]
    return long_sentences, entities

long_sentences, entities = check_clarity(documentation)
print("Long Sentences:", long_sentences)
print("Entities:", entities)

# Bar chart for section completeness
section_status = [0 if section in missing_sections else 1 for section in required_sections]
plt.figure(figsize=(8, 4))
plt.bar(required_sections, section_status, color='lightblue')
plt.title('Documentation Completeness Check')
plt.xlabel('Sections')
plt.ylabel('Present (1) / Missing (0)')
plt.ylim(0, 1.5)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Bar chart for repeated words
if repeated_words:
    plt.figure(figsize=(10, 5))
    plt.bar(repeated_words.keys(), repeated_words.values(), color='lightcoral')
    plt.title('Repeated Words in Documentation')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No repeated words found.")

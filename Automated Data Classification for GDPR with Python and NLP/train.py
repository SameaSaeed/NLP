import pandas as pd
import spacy
import matplotlib.pyplot as plt

df = pd.read_csv("personal_data.csv")
print(df.head())

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Function to extract entities from text
def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.text] = ent.label_
    return entities

# Apply the function to the columns in the dataframe
personal_data = {}

for index, row in df.iterrows():
    name_entities = extract_entities(row["Name"])
    email_entities = extract_entities(row["Email"])
    phone_entities = extract_entities(row["Phone"])
    address_entities = extract_entities(row["Address"])

    personal_data[index] = {
        "Name_Entities": name_entities,
        "Email_Entities": email_entities,
        "Phone_Entities": phone_entities,
        "Address_Entities": address_entities
    }

# Display the extracted entities
for key, value in personal_data.items():
    print(f"Row {key}:")
    for field, entities in value.items():
        print(f"  {field}: {entities}")
        
personal_data_tags = []

for index, row in df.iterrows():
    data_tags = []
    name_entities = extract_entities(row["Name"])
    email_entities = extract_entities(row["Email"])
    phone_entities = extract_entities(row["Phone"])
    address_entities = extract_entities(row["Address"])
    
    # Tagging logic for GDPR compliance
    if "PERSON" in [ent for ent in name_entities.values()]:
        data_tags.append("Name")
    if "EMAIL" in [ent for ent in email_entities.values()]:
        data_tags.append("Email")
    if "PHONE" in [ent for ent in phone_entities.values()]:
        data_tags.append("Phone")
    if "ADDRESS" in [ent for ent in address_entities.values()]:
        data_tags.append("Address")
    
    personal_data_tags.append(data_tags)

# Add tags to dataframe
df["GDPR_Tags"] = personal_data_tags

compliance_report = []

for index, row in df.iterrows():
    report = {
        "Name": row["Name"],
        "Email": row["Email"],
        "Phone": row["Phone"],
        "Address": row["Address"],
        "GDPR_Tags": ", ".join(row["GDPR_Tags"]),
    }
    compliance_report.append(report)

# Save the compliance report to a CSV
compliance_df = pd.DataFrame(compliance_report)
compliance_df.to_csv("compliance_report.csv", index=False)

print("Compliance report generated: compliance_report.csv")

# Create a count of GDPR Tags
tag_counts = df["GDPR_Tags"].explode().value_counts()

# Plot the distribution of tags
tag_counts.plot(kind='bar', color='skyblue')
plt.title("GDPR Data Distribution")
plt.xlabel("Data Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()
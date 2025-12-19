# Install required Python libraries pip3 install spacy pandas matplotlib seaborn

# Download spaCy's English language model python3 -m spacy download en_core_web_sm

import spacy
import pandas as pd
import re
from spacy import displacy
from spacy.matcher import Matcher
import matplotlib.pyplot as plt
import seaborn as sns

# Sample cybersecurity log data
cybersecurity_logs = [
    "Failed login attempt from IP address 192.168.1.100 to Microsoft Exchange server at 2023-10-15 14:30:22",
    "Malware detected: Trojan.Win32.Agent found in file system by Symantec Endpoint Protection",
    "Suspicious network activity from 10.0.0.45 attempting to connect to known botnet C&C server 203.0.113.5",
    "DDoS attack detected from multiple IP addresses including 172.16.0.1, 172.16.0.2, and 172.16.0.3 targeting Apache web server",
    "Phishing email detected from sender@malicious-domain.com targeting employees at Acme Corporation",
    "Ransomware activity detected: WannaCry variant attempting to encrypt files on Windows workstation 192.168.10.50",
    "Firewall blocked connection attempt from 198.51.100.10 to internal database server at Oracle Corporation",
    "Intrusion detection system alert: SQL injection attempt detected on web application hosted by Amazon Web Services",
    "Unauthorized access attempt to Active Directory server from IP 203.0.113.100 at Google headquarters",
    "Vulnerability scan detected from Nessus scanner at IP address 10.10.10.5 targeting Cisco router firmware"
]

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

def extract_entities_from_logs(logs):
    """
    Extract named entities from cybersecurity logs using spaCy NER
    """
    all_entities = []
    
    for i, log in enumerate(logs):
        # Process the log with spaCy
        doc = nlp(log)
        
        # Extract entities
        log_entities = []
        for ent in doc.ents:
            log_entities.append({
                'log_id': i,
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        all_entities.extend(log_entities)
    
    return all_entities

# Extract entities from our sample logs
entities = extract_entities_from_logs(cybersecurity_logs)

def categorize_security_entities(entities):
    """
    Categorize entities relevant to cybersecurity
    """
    ip_addresses = []
    organizations = []
    products = []
    dates = []
    other_entities = []
    
    for entity in entities:
        if entity['label'] in ['CARDINAL', 'QUANTITY']:
            # Check if it might be an IP address using regex
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            if re.match(ip_pattern, entity['text']):
                ip_addresses.append(entity)
            else:
                other_entities.append(entity)
        elif entity['label'] in ['ORG']:
            organizations.append(entity)
        elif entity['label'] in ['PRODUCT']:
            products.append(entity)
        elif entity['label'] in ['DATE', 'TIME']:
            dates.append(entity)
        else:
            other_entities.append(entity)
    
    return {
        'ip_addresses': ip_addresses,
        'organizations': organizations,
        'products': products,
        'dates': dates,
        'other': other_entities
    }

# Categorize our extracted entities
categorized_entities = categorize_security_entities(entities)

print("Entity Categories:")
for category, entity_list in categorized_entities.items():
    print(f"{category.upper()}: {len(entity_list)} entities")
    if entity_list:
        print(f"  Examples: {[e['text'] for e in entity_list[:3]]}")

def extract_ip_addresses(logs):
    """
    Extract IP addresses using regex patterns
    """
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    ip_addresses = []
    
    for i, log in enumerate(logs):
        matches = re.finditer(ip_pattern, log)
        for match in matches:
            ip_addresses.append({
                'log_id': i,
                'ip_address': match.group(),
                'position': match.span(),
                'log_text': log
            })
    
    return ip_addresses

# Extract IP addresses
ip_addresses = extract_ip_addresses(cybersecurity_logs)

print(f"IP Addresses found: {len(ip_addresses)}")
print("\nExtracted IP Addresses:")
for ip in ip_addresses:
    print(f"Log {ip['log_id']}: {ip['ip_address']}")

def categorize_ip_addresses(ip_list):
    """
    Categorize IP addresses by type (private, public, etc.)
    """
    private_ranges = [
        (r'^10\.', 'Class A Private'),
        (r'^172\.(1[6-9]|2[0-9]|3[0-1])\.', 'Class B Private'),
        (r'^192\.168\.', 'Class C Private'),
        (r'^127\.', 'Loopback'),
        (r'^169\.254\.', 'Link-local')
    ]
    
    categorized = {'private': [], 'public': [], 'special': []}
    
    for ip_info in ip_list:
        ip = ip_info['ip_address']
        is_private = False
        
        for pattern, category in private_ranges:
            if re.match(pattern, ip):
                categorized['private'].append({**ip_info, 'category': category})
                is_private = True
                break
        
        if not is_private:
            categorized['public'].append({**ip_info, 'category': 'Public'})
    
    return categorized

# Categorize IP addresses
ip_categories = categorize_ip_addresses(ip_addresses)

print("\nIP Address Categories:")
for category, ips in ip_categories.items():
    print(f"{category.upper()}: {len(ips)} addresses")
    for ip in ips:
        print(f"  {ip['ip_address']} ({ip.get('category', 'Unknown')})")

# Visualize the distribution of entity types
def visualize_entities_in_logs(logs, output_file="entities_visualization.html"):
    """
    Create HTML visualization of entities in logs
    """
    html_content = "<html><head><title>Cybersecurity Log NER Visualization</title></head><body>"
    html_content += "<h1>Named Entity Recognition in Cybersecurity Logs</h1>"
    
    for i, log in enumerate(logs):
        doc = nlp(log)
        html_content += f"<h3>Log {i+1}:</h3>"
        
        # Generate spaCy visualization
        html = displacy.render(doc, style="ent", page=False)
        html_content += html
        html_content += "<br><br>"
    
    html_content += "</body></html>"
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visualization saved to {output_file}")

# Create visualization
visualize_entities_in_logs(cybersecurity_logs[:5])  # Visualize first 5 logs

# Plot entity distribution
def create_entity_statistics(entities):
    """
    Create statistical visualizations of extracted entities
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(entities)
    
    if not df.empty:
        # Count entities by label
        entity_counts = df['label'].value_counts()
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Bar plot of entity types
        plt.subplot(1, 2, 1)
        entity_counts.plot(kind='bar')
        plt.title('Entity Types Distribution')
        plt.xlabel('Entity Label')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Pie chart of top entity types
        plt.subplot(1, 2, 2)
        top_entities = entity_counts.head(5)
        plt.pie(top_entities.values, labels=top_entities.index, autopct='%1.1f%%')
        plt.title('Top 5 Entity Types')
        
        plt.tight_layout()
        plt.savefig('entity_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Entity statistics visualization saved as 'entity_statistics.png'")
    else:
        print("No entities found for visualization")

# Create statistics visualization
create_entity_statistics(entities)

Customize with Spacy Matcher
def create_threat_patterns():
    """
    Create custom patterns for cybersecurity threats using spaCy Matcher
    """
    matcher = Matcher(nlp.vocab)
    
    # Define patterns for different threat types
    malware_patterns = [
        [{"LOWER": "malware"}],
        [{"LOWER": "trojan"}],
        [{"LOWER": "virus"}],
        [{"LOWER": "ransomware"}],
        [{"LOWER": "wannacry"}],
        [{"LOWER": "botnet"}]
    ]
    
    attack_patterns = [
        [{"LOWER": "ddos"}, {"LOWER": "attack"}],
        [{"LOWER": "phishing"}],
        [{"LOWER": "sql"}, {"LOWER": "injection"}],
        [{"LOWER": "intrusion"}],
        [{"LOWER": "vulnerability"}, {"LOWER": "scan"}]
    ]
    
    network_patterns = [
        [{"LOWER": "firewall"}],
        [{"LOWER": "router"}],
        [{"LOWER": "server"}],
        [{"LOWER": "endpoint"}],
        [{"LOWER": "workstation"}]
    ]
    
    # Add patterns to matcher
    matcher.add("MALWARE", malware_patterns)
    matcher.add("ATTACK", attack_patterns)
    matcher.add("NETWORK_COMPONENT", network_patterns)
    
    return matcher

# Create custom matcher
threat_matcher = create_threat_patterns()
print("Custom threat patterns created successfully!")

def extract_custom_threats(logs, matcher):
    """
    Extract custom threat patterns from logs
    """
    custom_entities = []
    
    for i, log in enumerate(logs):
        doc = nlp(log)
        matches = matcher(doc)
        
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            span = doc[start:end]
            custom_entities.append({
                'log_id': i,
                'text': span.text,
                'label': label,
                'start': span.start_char,
                'end': span.end_char,
                'log_text': log
            })
    
    return custom_entities

# Extract custom threat entities
custom_threats = extract_custom_threats(cybersecurity_logs, threat_matcher)

print(f"Custom threat entities found: {len(custom_threats)}")
print("\nCustom Threat Entities:")
for threat in custom_threats:
    print(f"Log {threat['log_id']}: '{threat['text']}' -> {threat['label']}")

def combine_ner_results(standard_entities, custom_entities):
    """
    Combine standard spaCy NER results with custom pattern matches
    """
    # Create comprehensive entity analysis
    all_entities = standard_entities + custom_entities
    
    # Create summary statistics
    entity_summary = {}
    for entity in all_entities:
        label = entity['label']
        if label not in entity_summary:
            entity_summary[label] = []
        entity_summary[label].append(entity['text'])
    
    return all_entities, entity_summary

# Combine results
combined_entities, entity_summary = combine_ner_results(entities, custom_threats)

print(f"Total combined entities: {len(combined_entities)}")
print("\nEntity Summary by Type:")
for label, texts in entity_summary.items():
    unique_texts = list(set(texts))
    print(f"{label}: {len(texts)} total, {len(unique_texts)} unique examples")

def generate_security_report(logs, entities, ip_addresses, custom_threats):
    """
    Generate a comprehensive security intelligence report
    """
    report = []
    report.append("="*60)
    report.append("CYBERSECURITY LOG ANALYSIS REPORT")
    report.append("="*60)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Logs Analyzed: {len(logs)}")
    report.append("")
    
    # IP Address Analysis
    report.append("IP ADDRESS ANALYSIS:")
    report.append("-" * 30)
    ip_categories = categorize_ip_addresses(ip_addresses)
    for category, ips in ip_categories.items():
        report.append(f"{category.upper()} IP Addresses: {len(ips)}")
        for ip in ips[:5]:  # Show first 5
            report.append(f"  - {ip['ip_address']}")
    report.append("")
    
    # Threat Analysis
    report.append("THREAT ANALYSIS:")
    report.append("-" * 30)
    threat_counts = {}
    for threat in custom_threats:
        label = threat['label']
        threat_counts[label] = threat_counts.get(label, 0) + 1
    
    for threat_type, count in threat_counts.items():
        report.append(f"{threat_type}: {count} instances")
    report.append("")
    
    # Organization Analysis
    report.append("ORGANIZATION ANALYSIS:")
    report.append("-" * 30)
    orgs = [e for e in entities if e['label'] == 'ORG']
    org_names = list(set([org['text'] for org in orgs]))
    for org in org_names:
        report.append(f"  - {org}")
    report.append("")
    
    # Recommendations
    report.append("SECURITY RECOMMENDATIONS:")
    report.append("-" * 30)
    if len(ip_addresses) > 5:
        report.append("- High number of IP addresses detected - consider network segmentation")
    if any(threat['label'] == 'MALWARE' for threat in custom_threats):
        report.append("- Malware detected - update antivirus signatures and scan systems")
    if any(threat['label'] == 'ATTACK' for threat in custom_threats):
        report.append("- Attack patterns detected - review firewall rules and access controls")
    
    return "\n".join(report)

# Generate and display report
security_report = generate_security_report(cybersecurity_logs, entities, ip_addresses, custom_threats)
print(security_report)

# Save report to file
with open('security_analysis_report.txt', 'w') as f:
    f.write(security_report)

print("\nReport saved to 'security_analysis_report.txt'")

def export_results_to_csv(entities, ip_addresses, custom_threats):
    """
    Export analysis results to CSV files for further processing
    """
    # Export standard entities
    if entities:
        entities_df = pd.DataFrame(entities)
        entities_df.to_csv('standard_entities.csv', index=False)
        print("Standard entities exported to 'standard_entities.csv'")
    
    # Export IP addresses
    if ip_addresses:
        ip_df = pd.DataFrame(ip_addresses)
        ip_df.to_csv('ip_addresses.csv', index=False)
        print("IP addresses exported to 'ip_addresses.csv'")
    
    # Export custom threats
    if custom_threats:
        threats_df = pd.DataFrame(custom_threats)
        threats_df.to_csv('custom_threats.csv', index=False)
        print("Custom threats exported to 'custom_threats.csv'")

# Export results
export_results_to_csv(entities, ip_addresses, custom_threats)
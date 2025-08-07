Install dependencies:
sudo apt update
sudo apt install openjdk-11-jdk wget apt-transport-https -y

Install ELK Stack (Offline-friendly Setup)

Elasticsearch:
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.17-amd64.deb
sudo dpkg -i elasticsearch-7.17.17-amd64.deb
sudo systemctl enable elasticsearch
sudo systemctl start elasticsearch

Logstash:
wget https://artifacts.elastic.co/downloads/logstash/logstash-7.17.17-amd64.deb
sudo dpkg -i logstash-7.17.17-amd64.deb

Kibana:
wget https://artifacts.elastic.co/downloads/kibana/kibana-7.17.17-amd64.deb
sudo dpkg -i kibana-7.17.17-amd64.deb
sudo systemctl enable kibana
sudo systemctl start kibana
Access Kibana at: http://localhost:5601

Steps

Set Up Log Ingestion with Logstash

a. Create a sample log file:
mkdir -p ~/elk-nlp-lab/logs
echo "[2025-07-09] ERROR: User login failed due to invalid credentials." >> ~/elk-nlp-lab/logs/system.log
echo "[2025-07-09] WARNING: Unusual login location detected." >> ~/elk-nlp-lab/logs/system.log

b. Create a Logstash config:

c. Run Logstash:
sudo /usr/share/logstash/bin/logstash -f ~/elk-nlp-lab/logstash.conf

Visualize Logs in Kibana

a. Go to Kibana Management, Stack Management Index Patterns
b. Create index pattern: compliance-logs*
c. Use the Discover tab to explore logs
d. Add visualizations (e.g., a pie chart of log levels)

3. NLP Analysis of Logs
python3 nlp_compliance.py
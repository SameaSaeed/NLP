import re
import matplotlib.pyplot as plt

# Sample system logs (simulated content)
logs = """
2025-07-01 00:05:23, Error, User 'admin' failed to login from IP 192.168.1.1
2025-07-01 00:10:45, Info, User 'john' successfully logged in from IP 192.168.1.2
2025-07-01 00:15:32, Warning, CPU usage exceeded 85% on server 'web-server-01'
2025-07-01 00:20:12, Info, User 'admin' successfully logged in from IP 192.168.1.3
2025-07-01 00:25:47, Error, User 'admin' failed to login from IP 192.168.1.4
"""

# Define regex pattern to extract timestamp, log level, user, and IP address
pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), (\w+),.*User '(\w+)' .*from IP (\d+\.\d+\.\d+\.\d+)"

# Use re.findall to extract data based on the pattern
log_data = re.findall(pattern, logs)

# Convert extracted data into a DataFrame-like structure (list of tuples)
log_entries = [(timestamp, log_level, user, ip) for timestamp, log_level, user, ip in log_data]
print(log_entries)

# Extract log levels and their count
log_levels = [log[1] for log in log_entries]

# Plot the distribution of log levels
plt.figure(figsize=(8, 6))
plt.hist(log_levels, bins=len(set(log_levels)), color='skyblue', edgecolor='black')
plt.title('Log Level Distribution')
plt.xlabel('Log Level')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Extract users and their associated IP addresses
user_ip_mapping = [(log[2], log[3]) for log in log_entries]

# Count occurrences of each user-IP pair
from collections import Counter
user_ip_count = Counter(user_ip_mapping)

# Prepare data for plotting
users, ip_counts = zip(*user_ip_count.items())
user_ip_pairs = [f"{user} - {ip}" for user, ip in users]

# Plot the distribution of users and their corresponding IPs
plt.figure(figsize=(10, 6))
plt.barh(user_ip_pairs, ip_counts, color='lightgreen')
plt.title('User Activity by IP')
plt.xlabel('Number of Events')
plt.ylabel('User - IP Pair')
plt.show()

# Example: Detecting "failed login" events
failed_logins = [log for log in log_entries if 'failed to login' in log[2]]

# Count and display failed login attempts
print(f"Failed login attempts: {len(failed_logins)}")
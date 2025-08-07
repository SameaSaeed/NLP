import pandas as pd

# Simulating cloud logs
logs = [
    "Error: Pod 'web-server' crashed due to memory exhaustion",
    "Warning: CPU usage for 'database' exceeds 85% threshold",
    "Info: Pod 'load-balancer' started successfully",
    "Error: Pod 'api-server' failed to start due to missing configuration",
    "Info: 'worker' pod is running at 90% efficiency",
    "Warning: 'database' pod memory usage increased by 20%",
    "Error: 'auth-service' failed with a 500 error",
    "Info: 'auth-service' restarted successfully after failure"
]

# Convert logs to DataFrame
df_logs = pd.DataFrame(logs, columns=['log_message'])
print(df_logs)
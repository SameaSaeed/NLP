# python -m textblob.download_corpora

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Simulated user feedback data
feedback_data = {
    'user_feedback': [
        'The service is great, very happy with the performance!',
        'I am disappointed, the system crashes frequently.',
        'Good support, but the response time could be improved.',
        'Terrible experience, I will not use this again.',
        'Love the new update, it makes things easier!',
        'It was okay, nothing special but not bad either.',
        'Worst experience ever, everything is slow and unresponsive.',
        'Fantastic, everything works as expected, great job!',
        'The system is slow, but it gets the job done eventually.',
        'Happy with the service, but I wish there were more features.'
    ]
}

# Create a DataFrame
df_feedback = pd.DataFrame(feedback_data)

# Function to get sentiment polarity and subjectivity
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Apply sentiment analysis to each feedback
df_feedback['polarity'], df_feedback['subjectivity'] = zip(*df_feedback['user_feedback'].apply(get_sentiment))

# Print the DataFrame with sentiment values
print(df_feedback)

# Plot the polarity values
plt.figure(figsize=(8, 6))
plt.barh(df_feedback['user_feedback'], df_feedback['polarity'], color='skyblue', edgecolor='black')
plt.xlabel('Polarity')
plt.title('Sentiment Polarity of User Feedback')
plt.show()

# Plot polarity vs. subjectivity
plt.figure(figsize=(8, 6))
plt.scatter(df_feedback['polarity'], df_feedback['subjectivity'], color='purple', alpha=0.6)
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.title('Polarity vs. Subjectivity of User Feedback')
plt.show()
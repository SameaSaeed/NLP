# pip install transformers torch datasets scikit-learn matplotlib

import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt

# Simulated dataset of malware descriptions and categories
data = {
    'malware_description': [
        'This Trojan horse infects the victimâs system by disguising itself as a legitimate file.',
        'Ransomware encrypts files on the victimâs computer and demands payment for the decryption key.',
        'Spyware monitors the victimâs activity and sends sensitive data back to the attacker.',
        'Adware is software that automatically displays advertisements on the victimâs screen.',
        'This Trojan horse spreads through phishing emails, disguised as a legitimate software update.',
        'Ransomware locks access to the system and demands Bitcoin payment for decryption.'
    ],
    'category': ['Trojan', 'Ransomware', 'Spyware', 'Adware', 'Trojan', 'Ransomware']
}

# Create DataFrame
df = pd.DataFrame(data)
print(df)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples['malware_description'], padding="max_length", truncation=True)

# Prepare the dataset for BERT
from datasets import Dataset

# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column('category', 'labels')

# Split the dataset into training and testing sets
train_dataset, test_dataset = tokenized_datasets.train_test_split(test_size=0.2).values()

# Load BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate after every epoch
)

# Define the Trainer
trainer = Trainer(
    model=model,                         # the pre-trained model
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Plot training and validation loss
training_loss = trainer.state.log_history
epochs = [x['epoch'] for x in training_loss]

train_loss = [x['loss'] for x in training_loss if 'loss' in x]
eval_loss = [x['eval_loss'] for x in training_loss if 'eval_loss' in x]

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, eval_loss, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
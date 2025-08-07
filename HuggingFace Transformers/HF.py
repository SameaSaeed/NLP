from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load the IMDb dataset
dataset = load_dataset("imdb")
print(dataset)

# Load the pre-trained tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Apply the preprocessing function
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# Load the pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=8,   # Batch size per device
    per_device_eval_batch_size=8,    # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps
    weight_decay=0.01,               # Strength of weight decay
    logging_dir="./logs",            # Directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # The model to train
    args=training_args,                  # Training arguments
    train_dataset=tokenized_datasets["train"],   # Training dataset
    eval_dataset=tokenized_datasets["test"],     # Evaluation dataset
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model and tokenizer
model.save_pretrained("fine_tuned_bert_imdb")
tokenizer.save_pretrained("fine_tuned_bert_imdb")
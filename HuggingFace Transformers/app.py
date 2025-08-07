from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("fine_tuned_bert_imdb")
tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert_imdb")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the request
    data = request.get_json()
    text = data['text']

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Predict using the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()

    # Return the prediction (0 for negative, 1 for positive)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
# pip install nltk

import nltk
nltk.download('punkt')  # For tokenization
nltk.download('averaged_perceptron_tagger')  # For POS tagging

# Sample text
text = "Hello! This is an example sentence. Natural Language Processing is fun."

# Sentence Tokenization
sentences = nltk.sent_tokenize(text)
print("Sentences:", sentences)

# Word Tokenization
words = nltk.word_tokenize(text)
print("Words:", words)

# Part-of-speech tagging
tagged_words = nltk.pos_tag(words)
print("POS Tagged Words:", tagged_words)
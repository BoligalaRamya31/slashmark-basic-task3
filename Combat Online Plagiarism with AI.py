import numpy as np
 import pandas as pd
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.metrics.pairwise import cosine_similarity
 import nltk
 from nltk.corpus import stopwords
 import string
 import re
 
 # Download NLTK stopwords
 nltk.download('stopwords')
 
 def preprocess_text(text):
     # Convert text to lowercase
     text = text.lower()
     # Remove punctuation
     text = text.translate(str.maketrans('', '', string.punctuation))
     # Remove numbers
     text = re.sub(r'\d+', '', text)
     # Remove whitespace
     text = text.strip()
     # Remove stopwords
     stop_words = set(stopwords.words('english'))
     text = ' '.join([word for word in text.split() if word not in stop_words])
     return text
 
 def vectorize_text(texts):
     vectorizer = TfidfVectorizer()
     vectors = vectorizer.fit_transform(texts)
     return vectors, vectorizer
 
 def calculate_similarity(vector, vectors):
     cosine_similarities = cosine_similarity(vector, vectors)
     return cosine_similarities.flatten()
 
 # Sample dataset
 documents = [
     "The quick brown fox jumps over the lazy dog.",
     "Never jump over the lazy dog quickly.",
     "A quick brown dog outpaces a quick fox."
 ]
 
 # Preprocess documents
 preprocessed_docs = [preprocess_text(doc) for doc in documents]
 
 # Vectorize documents
 vectors, vectorizer = vectorize_text(preprocessed_docs)
 
 # Input text
 input_text = "The quick brown fox jumps over the lazy dog."
 preprocessed_input = preprocess_text(input_text)
 input_vector = vectorizer.transform([preprocessed_input])
 
 # Calculate similarity
 similarities = calculate_similarity(input_vector, vectors)
 
 # Display results
 for i, similarity in enumerate(similarities):
     print(f"Document {i + 1}: Similarity = {similarity:.4f}")

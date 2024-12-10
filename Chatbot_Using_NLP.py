import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import warnings


nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")
questions_list = data["Questions"]
answers_list = data["Answers"]


def preprocess_without_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = [stemmer.stem(token) for token in lemmatized]
    return " ".join(stemmed)


def preprocess_with_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = [stemmer.stem(token) for token in lemmatized]
    return " ".join(stemmed)


vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform([preprocess_without_stopwords(q) for q in questions_list])


def get_response(text):
    preprocessed_text = preprocess_with_stopwords(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    similarities = cosine_similarity(vectorized_text, X)
    max_similarity = np.max(similarities)
    if max_similarity >= 0.5:
        high_similarity_questions = [
            q for q, s in zip(questions_list, similarities[0]) if s >= 0.5
        ]

        target_answers = []
        for q in high_similarity_questions:
            target_answers.append(data[data["Questions"] == q]["Answers"].values[0])
        return f"Akash's {text} {target_answers[0]}"
    return f"Please try something else"


chat_history = []


def main():
    st.title("Know about Shivaprasad")
    if chat_history:
        st.text("Chat History:")
        for entry in chat_history:
            st.write(entry)

    user_input = st.text_input("Ask here")
    if user_input:
        response = get_response(user_input)
        chat_history.append(f"You: {user_input}")
        chat_history.append(f"Bot: {response}")

    st.text("Current Conversation:")
    for entry in chat_history:
        st.write(entry)


# Run the app
if __name__ == "__main__":
    main()

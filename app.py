import streamlit as st
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download('punkt')

ps = PorterStemmer()


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email / SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in stop_words]
    tokens = [ps.stem(i) for i in tokens]
    return " ".join(tokens)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error(" Spam Message")
        else:
            st.success(" Not Spam")

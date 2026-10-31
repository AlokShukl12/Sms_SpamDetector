import streamlit as st
import string
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(Message):
    Message = Message.lower()

    # Tokenization
    Message = nltk.word_tokenize(Message)

    y = []

    # Remove special characters
    for i in Message:
        if i.isalnum():
            y.append(i)

    Message = y[:]
    y.clear()

    # Remove stopwords
    for i in Message:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    Message = y[:]
    y.clear()

    # Stemming
    for i in Message:
        y.append(ps.stem(i))

    return " ".join(y)


# Load trained model and vectorizer
tfidf = pickle.load(open('vectorizer2.pkl', 'rb'))
mnb = pickle.load(open('model2.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter your SMS")

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter an SMS message.")

    else:
        # Preprocessing
        transformed_sms = transform_text(input_sms)

        # Vectorization
        vector_input = tfidf.transform([transformed_sms])

        # Prediction
        model_output = mnb.predict(vector_input)[0]

        # Display result
        if model_output == 1:
            st.error("Spam")
        else:
            st.success("Not a Spam")

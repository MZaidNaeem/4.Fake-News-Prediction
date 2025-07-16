import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ----------------- Setup -----------------
nltk.download('stopwords')
port_stem = PorterStemmer()

# ----------------- Stemming Function -----------------
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# ----------------- Load Model + Vectorizer -----------------
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")

st.markdown("Enter the news text and select its subject to check if it's **Real** or **Fake**.")

# Subject Dropdown
subject = st.selectbox("Select Subject", [
     'World News', 'News', 'Politics', 'Government News',
    'Left-News', 'US_News', 'Middle-east'
])

# News Content Input
news_input = st.text_area("Enter News Content Here", height=200)

# Predict Button
if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        # Attach subject to content
        full_text = subject + " " + news_input

        # Apply preprocessing
        preprocessed_text = stemming(full_text)

        # Transform using TF-IDF vectorizer
        transformed_text = vectorizer.transform([preprocessed_text])

        # Predict using model
        prediction = model.predict(transformed_text)

        # Output result
        result = "🟢 Real News" if prediction[0] == 0 else "🔴 Fake News"
        st.subheader("Prediction:")
        st.success(result)

📰 Fake News Detection Web App

![screenshot](gitimg.png)

A Machine Learning-powered NLP web application that classifies news articles as Real or Fake using logistic regression and TF-IDF features. Built with Streamlit for an interactive UI.

🚀 Overview
This project is part of my 25 Premium AI Projects in 25 Days Challenge, aiming to build production-ready ML solutions.
Despite working with noisy, imbalanced, and poorly labeled data, the model achieved an unexpected 80% accuracy — thanks to deep EDA, strong text preprocessing, and robust pipeline design.

🧠 How It Works
User inputs a news article and selects a relevant subject (e.g., Politics, US_News).

The app preprocesses the combined input (subject + content):

Lowercasing

Regex cleaning

Stopword removal

Stemming

The cleaned text is vectorized using a pre-trained TF-IDF model.

A trained Logistic Regression model predicts whether the news is Real or Fake.

The result is shown in an intuitive, real-time interface.

🔍 Features
✅ Logistic Regression model with TF-IDF features

✅ Robust text cleaning pipeline using nltk

✅ 80% accuracy on noisy, real-world data

✅ User-friendly interface with Streamlit

✅ Joblib model + vectorizer loading

✅ Supports 7 subject categories

🛠️ Tech Stack
Python

Streamlit

scikit-learn

nltk

joblib

TF-IDF Vectorizer

Regular Expressions


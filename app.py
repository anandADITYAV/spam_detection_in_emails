import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and the TfidfVectorizer
with open('spam_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit app layout
st.title("Email Spam Detection App")
st.write("Enter the email content to check if it's spam or not.")

# Text input for email content
email_content = st.text_area("Email Content")

# Predict button
if st.button("Check"):
    if email_content:
        # Preprocess the input email content
        email_vector = vectorizer.transform([email_content])

        # Make prediction
        prediction = model.predict(email_vector)

        # Show result
        if prediction[0] == 1:
            st.error("This email is spam.")
        else:
            st.success("This email is not spam.")

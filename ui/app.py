# ui/app.py
import streamlit as st
import requests
import os
import time

# The API URL is set from environment variables (or default to service container).
API_URL = os.getenv("API_URL", "http://service:8000/predict")
st.title("IMDB Sentiment Classifier")

# Let the user choose which framework (Autogen or LangChain) to use.
agent_choice = st.radio("Choose the processing framework:", ("Autogen", "LangChain"))

# Display a message based on the framework choice.
if agent_choice == "Autogen":
    st.write("The Autogen framework is chosen. The system will use the internal functions for preprocessing and classification.")
else:
    st.write("The LangChain framework is chosen. The system will use LangChain chains with a DistilGPT2-based LLM for processing.")

st.write("Enter a movie review below to predict its sentiment:")
review_text = st.text_area("Movie Review", height=200)

def call_api(review, agent_type, retries=3, delay=2):
    payload = {"review": review, "agent_type": agent_type.lower()}
    for i in range(retries):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.warning(f"Attempt {i+1}/{retries} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    st.error("Failed to contact the API service after several attempts.")
    return None

if st.button("Predict Sentiment"):
    if not review_text.strip():
        st.error("Review text is empty.")
    else:
        data = call_api(review_text, agent_choice)
        if data is not None:
            sentiment = data.get("sentiment", "unknown")
            st.success(f"Sentiment ({agent_choice}): {sentiment}")

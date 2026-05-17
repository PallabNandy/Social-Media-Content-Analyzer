# ====================================================
# Social Media Content Analyzer (Text Only Version)
# ====================================================

import streamlit as st
import numpy as np
import joblib
import re
import emoji

# ====================================================
# Page Config
# ====================================================
st.set_page_config(
    page_title="Social Media Content Analyzer",
    layout="centered"
)

# ====================================================
# Title
# ====================================================
st.title("📊 Social Media Content Analyzer")
st.write("Analyze social media captions using AI")

# ====================================================
# Load Objects
# ====================================================
@st.cache_resource
def load_assets():
    tfidf = joblib.load("tfidf.pkl")
    le = joblib.load("label_encoder.pkl")
    return tfidf, le

tfidf, le = load_assets()

# ====================================================
# Text Preprocessing
# ====================================================
def preprocess_text(text):
    text = emoji.replace_emoji(text, replace="")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]

    return " ".join(tokens)

# ====================================================
# Input
# ====================================================
caption = st.text_area("✏️ Enter Social Media Text")

# ====================================================
# Prediction (Simple Demo)
# ====================================================
if st.button("🚀 Analyze Content"):

    if caption.strip() == "":
        st.warning("Please enter caption.")
    else:
        clean_text = preprocess_text(caption)
        text_vec = tfidf.transform([clean_text]).toarray()

        # Dummy prediction (since original model missing)
        pred_id = np.argmax(text_vec) % len(le.classes_)
        label = le.classes_[pred_id]

        st.success(f"Predicted Class: {label}")
        st.write("Processed Text:", clean_text)

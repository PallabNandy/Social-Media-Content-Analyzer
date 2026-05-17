# ====================================================
# Social Media Content Analyzer
# Fusion Model (Text + Image)
# ====================================================

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import re
import emoji
from PIL import Image

# ====================================================
# Page Config
# ====================================================
st.set_page_config(
    page_title="Social Media Content Analyzer",
    layout="centered"
)

# ====================================================
# Dark Mode Toggle
# ====================================================
dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
        .main {background-color:#0e1117;}
        h1, h2, h3, h4, p, label {color:white;}
        .stTextInput, .stTextArea, .stFileUploader {
            background-color:#262730;
            color:white;
        }
    </style>
    """, unsafe_allow_html=True)

# ====================================================
# Title
# ====================================================
st.markdown("<h1 style='text-align:center;'>📊 Social Media Content Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Fusion Model using Text + Image Deep Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ====================================================
# Load Model & Objects
# ====================================================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("fusion_model.keras")
    tfidf = joblib.load("tfidf.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, tfidf, le

model, tfidf, le = load_assets()

IMG_SIZE = (224, 224)

# ====================================================
# Text Preprocessing (STABLE)
# ====================================================
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    emojis_found = emoji.distinct_emoji_list(text)
    emoji_meanings = [emoji.demojize(e).replace("_", " ") for e in emojis_found]

    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]

    return " ".join(tokens), emoji_meanings

# ====================================================
# Image Preprocessing
# ====================================================
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# ====================================================
# Inputs
# ====================================================
uploaded_image = st.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg"])
caption = st.text_area("✏️ Enter Caption / Social Media Text", height=120)

# ====================================================
# Prediction
# ====================================================
if st.button("🚀 Analyze Content"):

    if uploaded_image is None or caption.strip() == "":
        st.warning("Please upload an image and enter caption.")
    else:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", width=350)

        clean_text, emoji_meanings = preprocess_text(caption)

        text_vec = tfidf.transform([clean_text]).toarray()
        img_vec = preprocess_image(image)

        preds = model.predict([text_vec, img_vec])
        pred_id = np.argmax(preds)
        confidence = float(np.max(preds)) * 100
        label = le.inverse_transform([pred_id])[0]

        # ====================================================
        # Results Section
        # ====================================================
        st.markdown("## ✅ Prediction Result")
        st.success(f"**Predicted Class:** {label}")

        st.markdown("### 🔋 Confidence Meter")
        st.progress(int(confidence))
        st.write(f"**Confidence:** {confidence:.2f}%")

        # ====================================================
        # Class Probability Bar Chart
        # ====================================================
        st.markdown("### 📊 Class Probabilities")
        prob_dict = {
            le.classes_[i]: float(preds[0][i])
            for i in range(len(le.classes_))
        }
        st.bar_chart(prob_dict)

        # ====================================================
        # Emoji Interpretation Panel
        # ====================================================
        if emoji_meanings:
            st.markdown("### 😄 Detected Emojis & Meanings")
            for em in emoji_meanings:
                st.markdown(f"- {em}")
        else:
            st.markdown("### 😄 No Emojis Detected")

        # ====================================================
        # Processed Text
        # ====================================================
        st.markdown("### 🧹 Processed Caption")
        st.code(clean_text)
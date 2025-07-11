import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import os

# === CONFIG ===
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🐾 Animal Image Classifier")
st.write("Upload an image of an **Elephant, Buffalo, Rhino, or Zebra** to classify it using a trained CNN model.")

# === LOAD MODEL ===
@st.cache_resource
def load_cnn_model():
    return load_model("Model.keras")

model = load_cnn_model()

# === CLASS LABELS ===
class_names = ['Buffalo', 'Elephant', 'Rhino', 'Zebra']  # Same order used during training

# === IMAGE UPLOAD ===
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    image = image.resize((150, 150))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.markdown(f"### ✅ Prediction: `{predicted_class}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

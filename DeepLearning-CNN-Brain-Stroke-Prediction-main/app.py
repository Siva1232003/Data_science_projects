import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Set Streamlit page config at the top before any other Streamlit commands
st.set_page_config(page_title="Brain Stroke Detector", layout="centered")

# Load the trained model
model = tf.keras.models.load_model('BrainStrokeClassifier.keras')

# Preprocess function
def preprocess_image(image_data):
    image = Image.open(image_data).convert("RGB")
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    return img_array

# Streamlit UI
st.title("🧠 Brain Stroke Prediction")
st.markdown("Upload a brain CT scan to check for signs of **stroke**.")

uploaded_file = st.file_uploader("Choose a brain CT image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded CT Scan", use_column_width=True)

    img_array = preprocess_image(uploaded_file)

    prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]

    if prediction > 0.5:
        st.error(f"⚠️ Stroke Detected! (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ No Stroke Detected. (Confidence: {1 - prediction:.2f})")
